"""
Harmonic Centrality Analyzer — Advanced SEO Internal Link Graph Tool
Crawls a website, builds the internal link graph, computes harmonic centrality
and other graph metrics, and provides advanced visualizations + PDF export.
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from collections import deque, Counter
from urllib.parse import urlparse, urljoin, urldefrag
import requests
from bs4 import BeautifulSoup
import time
import io
import json
import tempfile
import os
from fpdf import FPDF
import base64
import re
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────── CONFIG ────────────────────────────────

PALETTE = ['#20808D', '#A84B2F', '#1B474D', '#BCE2E7', '#944454', '#FFC553', '#848456', '#6E522B']
TEAL = '#20808D'
DARK_TEAL = '#1B474D'
BG = '#F7F6F2'
TEXT_COLOR = '#28251D'
MUTED = '#7A7974'

st.set_page_config(
    page_title="Harmonic Centrality Analyzer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── CUSTOM CSS ────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { background-color: #F7F6F2; }
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: white;
        border: 1px solid #D4D1CA;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #20808D;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #7A7974;
        font-weight: 500;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-sublabel {
        font-size: 0.75rem;
        color: #BAB9B4;
        margin-top: 2px;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #20808D 0%, #1B474D 100%);
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        margin: 12px 0;
    }
    .insight-box h4 {
        color: #BCE2E7;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .insight-box p {
        color: #E8F4F6;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #28251D;
        border-bottom: 3px solid #20808D;
        padding-bottom: 8px;
        margin: 32px 0 16px 0;
    }

    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-high { background: #E8F5E9; color: #2E7D32; }
    .score-med { background: #FFF8E1; color: #F57F17; }
    .score-low { background: #FFEBEE; color: #C62828; }
    .score-zero { background: #F5F5F5; color: #616161; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── CRAWLER ───────────────────────────────────

def normalize_url(url):
    """Normalize URL by removing fragment, trailing slash, and lowering."""
    url, _ = urldefrag(url)
    url = url.rstrip('/')
    return url


def is_internal(url, base_domain):
    """Check if a URL belongs to the same domain."""
    try:
        parsed = urlparse(url)
        return parsed.netloc == base_domain or parsed.netloc == ''
    except Exception:
        return False


def crawl_website(start_url, max_pages=100, delay=0.3, progress_bar=None, status_text=None):
    """
    BFS crawler that discovers internal links and builds an edge list.
    Returns: list of edges (source, target), set of crawled URLs, dict of page info
    """
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc
    base_scheme = parsed_start.scheme

    visited = set()
    queue = deque([normalize_url(start_url)])
    edges = []
    page_info = {}

    headers = {
        'User-Agent': 'HarmonicCentralityBot/1.0 (SEO Analysis Tool)',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    crawled = 0
    while queue and crawled < max_pages:
        current_url = queue.popleft()
        if current_url in visited:
            continue

        try:
            resp = requests.get(current_url, headers=headers, timeout=10, allow_redirects=True)
            content_type = resp.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                continue

            visited.add(current_url)
            crawled += 1

            if progress_bar:
                progress_bar.progress(min(crawled / max_pages, 1.0))
            if status_text:
                status_text.text(f"Crawling page {crawled}/{max_pages}: {current_url[:80]}...")

            soup = BeautifulSoup(resp.text, 'lxml')

            title = soup.title.string.strip() if soup.title and soup.title.string else current_url
            meta_desc = ''
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag and meta_tag.get('content'):
                meta_desc = meta_tag['content']

            h1_tags = [h.get_text(strip=True) for h in soup.find_all('h1')]
            word_count = len(soup.get_text().split())

            page_info[current_url] = {
                'title': title[:100],
                'meta_description': meta_desc[:200],
                'h1': h1_tags[0] if h1_tags else '',
                'word_count': word_count,
                'status_code': resp.status_code,
            }

            links_found = 0
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # Skip non-http links
                if href.startswith(('mailto:', 'tel:', 'javascript:', '#', 'data:')):
                    continue

                full_url = urljoin(current_url, href)
                full_url = normalize_url(full_url)

                if is_internal(full_url, base_domain):
                    # Skip common non-page resources
                    path = urlparse(full_url).path.lower()
                    skip_ext = ('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg',
                                '.css', '.js', '.xml', '.json', '.zip', '.mp4', '.mp3')
                    if any(path.endswith(ext) for ext in skip_ext):
                        continue

                    edges.append((current_url, full_url))
                    links_found += 1

                    if full_url not in visited and full_url not in queue:
                        queue.append(full_url)

            page_info[current_url]['outbound_internal_links'] = links_found
            time.sleep(delay)

        except requests.exceptions.RequestException:
            continue
        except Exception:
            continue

    return edges, visited, page_info


# ─────────────────────── GRAPH ANALYSIS ────────────────────────────────

def build_graph(edges, visited):
    """Build a directed graph from edges, keeping only visited nodes."""
    G = nx.DiGraph()
    G.add_nodes_from(visited)
    for src, tgt in edges:
        if src in visited and tgt in visited:
            G.add_edge(src, tgt)
        elif src in visited:
            G.add_node(tgt)
            G.add_edge(src, tgt)
    return G


def compute_all_metrics(G):
    """Compute harmonic centrality and other graph metrics."""
    metrics = {}

    # Harmonic centrality (the star metric)
    hc = nx.harmonic_centrality(G)
    metrics['harmonic_centrality'] = hc

    # In-degree and out-degree
    metrics['in_degree'] = dict(G.in_degree())
    metrics['out_degree'] = dict(G.out_degree())

    # PageRank for comparison
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200)
        metrics['pagerank'] = pr
    except Exception:
        metrics['pagerank'] = {n: 0 for n in G.nodes()}

    # Betweenness centrality
    try:
        bc = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
        metrics['betweenness_centrality'] = bc
    except Exception:
        metrics['betweenness_centrality'] = {n: 0 for n in G.nodes()}

    # Closeness centrality (for comparison with harmonic)
    try:
        cc = nx.closeness_centrality(G)
        metrics['closeness_centrality'] = cc
    except Exception:
        metrics['closeness_centrality'] = {n: 0 for n in G.nodes()}

    # HITS
    try:
        hubs, authorities = nx.hits(G, max_iter=200)
        metrics['hub_score'] = hubs
        metrics['authority_score'] = authorities
    except Exception:
        metrics['hub_score'] = {n: 0 for n in G.nodes()}
        metrics['authority_score'] = {n: 0 for n in G.nodes()}

    return metrics


def classify_score(score, max_score):
    """Classify a harmonic centrality score into a tier."""
    if max_score == 0:
        return 'zero'
    ratio = score / max_score
    if ratio >= 0.6:
        return 'high'
    elif ratio >= 0.3:
        return 'medium'
    elif score > 0:
        return 'low'
    else:
        return 'zero'


def get_url_path(url):
    """Extract just the path from a URL for cleaner display."""
    parsed = urlparse(url)
    path = parsed.path
    if not path or path == '/':
        return '/'
    return path


# ─────────────────────── VISUALIZATIONS ────────────────────────────────

def create_top_pages_chart(df, n=20):
    """Horizontal bar chart of top pages by harmonic centrality."""
    top = df.nlargest(n, 'harmonic_centrality').sort_values('harmonic_centrality')

    fig = go.Figure()
    
    colors = []
    max_hc = top['harmonic_centrality'].max()
    for val in top['harmonic_centrality']:
        ratio = val / max_hc if max_hc > 0 else 0
        if ratio >= 0.6:
            colors.append('#20808D')
        elif ratio >= 0.3:
            colors.append('#FFC553')
        else:
            colors.append('#A84B2F')

    fig.add_trace(go.Bar(
        x=top['harmonic_centrality'],
        y=top['path'],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in top['harmonic_centrality']],
        textposition='outside',
        textfont=dict(size=11),
        hovertemplate='<b>%{y}</b><br>Harmonic Centrality: %{x:.4f}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text=f"Top {n} Pages by Harmonic Centrality",
            font=dict(size=18, color=TEXT_COLOR),
        ),
        xaxis_title="Harmonic Centrality Score",
        yaxis_title="",
        template='plotly_white',
        height=max(500, n * 28),
        margin=dict(l=20, r=80, t=60, b=40),
        font=dict(family='Inter', size=12),
        plot_bgcolor='white',
    )
    return fig


def create_distribution_chart(df):
    """Histogram + KDE of harmonic centrality distribution."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Distribution of Harmonic Centrality",
        "Log-Scale Distribution"
    ))

    # Linear scale
    fig.add_trace(go.Histogram(
        x=df['harmonic_centrality'],
        nbinsx=40,
        marker_color=TEAL,
        opacity=0.8,
        name='Count',
        hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra></extra>',
    ), row=1, col=1)

    # Log scale for better visibility of tail
    non_zero = df[df['harmonic_centrality'] > 0]['harmonic_centrality']
    if len(non_zero) > 0:
        fig.add_trace(go.Histogram(
            x=np.log10(non_zero + 1e-10),
            nbinsx=40,
            marker_color=DARK_TEAL,
            opacity=0.8,
            name='Count (log)',
            hovertemplate='Log10(Score): %{x:.2f}<br>Count: %{y}<extra></extra>',
        ), row=1, col=2)

    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False,
        font=dict(family='Inter', size=12),
    )
    fig.update_xaxes(title_text="Harmonic Centrality", row=1, col=1)
    fig.update_xaxes(title_text="Log10(Harmonic Centrality)", row=1, col=2)
    fig.update_yaxes(title_text="Number of Pages", row=1, col=1)
    fig.update_yaxes(title_text="Number of Pages", row=1, col=2)
    return fig


def create_hc_vs_pagerank(df):
    """Scatter plot comparing harmonic centrality vs PageRank."""
    tier_colors = {'high': '#20808D', 'medium': '#FFC553', 'low': '#A84B2F', 'zero': '#BAB9B4'}
    
    fig = px.scatter(
        df,
        x='harmonic_centrality',
        y='pagerank',
        color='tier',
        color_discrete_map=tier_colors,
        hover_data=['path', 'in_degree', 'out_degree'],
        size='in_degree',
        size_max=20,
        labels={
            'harmonic_centrality': 'Harmonic Centrality',
            'pagerank': 'PageRank',
            'tier': 'Tier',
        },
    )

    fig.update_layout(
        title=dict(
            text="Harmonic Centrality vs PageRank — Reachability vs Authority",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        template='plotly_white',
        height=550,
        font=dict(family='Inter', size=12),
    )
    return fig


def create_depth_analysis(G, start_url):
    """Analyze and visualize page depth from homepage."""
    try:
        lengths = nx.single_source_shortest_path_length(G, start_url)
    except Exception:
        return None, None

    depth_data = []
    for node, depth in lengths.items():
        depth_data.append({'url': node, 'path': get_url_path(node), 'depth': depth})

    depth_df = pd.DataFrame(depth_data)

    depth_counts = depth_df['depth'].value_counts().sort_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Depth {d}" for d in depth_counts.index],
        y=depth_counts.values,
        marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(depth_counts))],
        text=depth_counts.values,
        textposition='outside',
        hovertemplate='%{x}<br>Pages: %{y}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text="Pages by Click Depth from Homepage",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        xaxis_title="Click Depth",
        yaxis_title="Number of Pages",
        template='plotly_white',
        height=400,
        font=dict(family='Inter', size=12),
    )
    return fig, depth_df


def create_link_equity_heatmap(df):
    """Heatmap of in-degree vs out-degree colored by harmonic centrality."""
    fig = px.scatter(
        df,
        x='in_degree',
        y='out_degree',
        color='harmonic_centrality',
        color_continuous_scale=[[0, '#F7F6F2'], [0.3, '#BCE2E7'], [0.6, '#20808D'], [1.0, '#1B474D']],
        hover_data=['path', 'harmonic_centrality'],
        size='harmonic_centrality',
        size_max=18,
        labels={
            'in_degree': 'Inbound Internal Links',
            'out_degree': 'Outbound Internal Links',
            'harmonic_centrality': 'HC Score',
        },
    )
    fig.update_layout(
        title=dict(
            text="Link Profile — In-Degree vs Out-Degree (color = Harmonic Centrality)",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        template='plotly_white',
        height=500,
        font=dict(family='Inter', size=12),
    )
    return fig


def create_tier_breakdown(df):
    """Donut chart of page tier distribution."""
    tier_counts = df['tier'].value_counts()
    tier_order = ['high', 'medium', 'low', 'zero']
    tier_labels = {'high': 'High Centrality', 'medium': 'Medium Centrality', 'low': 'Low Centrality', 'zero': 'Orphan / Zero'}
    tier_colors_map = {'high': '#20808D', 'medium': '#FFC553', 'low': '#A84B2F', 'zero': '#BAB9B4'}

    labels = []
    values = []
    colors = []
    for t in tier_order:
        if t in tier_counts.index:
            labels.append(tier_labels[t])
            values.append(tier_counts[t])
            colors.append(tier_colors_map[t])

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textfont=dict(size=13),
        hovertemplate='%{label}<br>Pages: %{value}<br>%{percent}<extra></extra>',
    )])

    fig.update_layout(
        title=dict(
            text="Page Tier Distribution",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        height=420,
        font=dict(family='Inter', size=12),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
    )
    return fig


def create_centrality_comparison(df):
    """Radar/parallel coordinates showing multiple centrality metrics for top pages."""
    top10 = df.nlargest(10, 'harmonic_centrality')

    metrics_to_compare = ['harmonic_centrality', 'pagerank', 'betweenness_centrality', 'closeness_centrality', 'authority_score']
    metric_labels = {
        'harmonic_centrality': 'Harmonic',
        'pagerank': 'PageRank',
        'betweenness_centrality': 'Betweenness',
        'closeness_centrality': 'Closeness',
        'authority_score': 'Authority (HITS)',
    }

    # Normalize each metric to 0-1 for comparison
    norm_data = {}
    for m in metrics_to_compare:
        col = top10[m]
        max_val = col.max()
        if max_val > 0:
            norm_data[metric_labels[m]] = (col / max_val).values
        else:
            norm_data[metric_labels[m]] = col.values

    fig = go.Figure()
    categories = list(metric_labels.values())

    for i, (_, row) in enumerate(top10.iterrows()):
        values = []
        for m in metrics_to_compare:
            max_val = top10[m].max()
            values.append(row[m] / max_val if max_val > 0 else 0)
        values.append(values[0])  # close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=row['path'][:40],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            fill='toself',
            opacity=0.3,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05]),
        ),
        title=dict(
            text="Centrality Comparison — Top 10 Pages (Normalized)",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        height=550,
        font=dict(family='Inter', size=11),
        showlegend=True,
        legend=dict(font=dict(size=10)),
    )
    return fig


def create_network_graph(G, hc_scores, max_nodes=80):
    """Interactive Plotly network graph colored by harmonic centrality."""
    # Take top nodes by HC
    sorted_nodes = sorted(hc_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
    top_nodes = set(n for n, _ in sorted_nodes)

    subG = G.subgraph(top_nodes).copy()

    pos = nx.spring_layout(subG, k=2.5, iterations=60, seed=42)

    # Edges
    edge_x, edge_y = [], []
    for src, tgt in subG.edges():
        if src in pos and tgt in pos:
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.4, color='#D4D1CA'),
        hoverinfo='none',
        mode='lines',
        opacity=0.5,
    )

    # Nodes
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    max_hc = max(hc_scores.values()) if hc_scores else 1

    for node in subG.nodes():
        if node in pos:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            hc = hc_scores.get(node, 0)
            path = get_url_path(node)
            node_text.append(f"{path}<br>HC: {hc:.4f}<br>In: {G.in_degree(node)} Out: {G.out_degree(node)}")
            node_color.append(hc)
            node_size.append(max(6, (hc / max_hc) * 35) if max_hc > 0 else 8)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale=[[0, '#BCE2E7'], [0.4, '#20808D'], [1.0, '#1B474D']],
            colorbar=dict(title="HC Score", thickness=15),
            line=dict(width=1, color='white'),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(
            text=f"Internal Link Network Graph (Top {max_nodes} Pages by HC)",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white',
        height=650,
        font=dict(family='Inter'),
    )
    return fig


def create_orphan_analysis(G, all_nodes):
    """Identify and display orphan pages (zero inbound internal links)."""
    orphans = []
    weakly_linked = []
    for node in all_nodes:
        in_deg = G.in_degree(node) if node in G else 0
        if in_deg == 0:
            orphans.append(node)
        elif in_deg == 1:
            weakly_linked.append(node)
    return orphans, weakly_linked


def create_crawl_depth_treemap(depth_df, hc_dict):
    """Treemap showing pages grouped by depth, sized by HC."""
    if depth_df is None or len(depth_df) == 0:
        return None

    depth_df = depth_df.copy()
    depth_df['hc'] = depth_df['url'].map(hc_dict).fillna(0)
    depth_df['depth_label'] = 'Depth ' + depth_df['depth'].astype(str)
    depth_df['size'] = depth_df['hc'] + 0.01  # avoid zero-size

    fig = px.treemap(
        depth_df,
        path=['depth_label', 'path'],
        values='size',
        color='hc',
        color_continuous_scale=[[0, '#F7F6F2'], [0.3, '#BCE2E7'], [0.6, '#20808D'], [1.0, '#1B474D']],
        labels={'hc': 'HC Score'},
        hover_data=['hc'],
    )
    fig.update_layout(
        title=dict(
            text="Page Architecture Treemap — Depth vs Harmonic Centrality",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        height=500,
        font=dict(family='Inter', size=12),
    )
    return fig


def create_top_bottom_comparison(df, n=10):
    """Side-by-side comparison of top vs bottom pages."""
    top_n = df.nlargest(n, 'harmonic_centrality')
    non_zero = df[df['harmonic_centrality'] > 0]
    bottom_n = non_zero.nsmallest(n, 'harmonic_centrality') if len(non_zero) >= n else non_zero

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Top {n} Pages (Highest HC)", f"Bottom {min(n, len(bottom_n))} Pages (Lowest Non-Zero HC)"),
        horizontal_spacing=0.15,
    )

    fig.add_trace(go.Bar(
        y=top_n['path'],
        x=top_n['harmonic_centrality'],
        orientation='h',
        marker_color=TEAL,
        name='Top',
        text=[f"{v:.3f}" for v in top_n['harmonic_centrality']],
        textposition='outside',
    ), row=1, col=1)

    if len(bottom_n) > 0:
        fig.add_trace(go.Bar(
            y=bottom_n['path'],
            x=bottom_n['harmonic_centrality'],
            orientation='h',
            marker_color='#A84B2F',
            name='Bottom',
            text=[f"{v:.4f}" for v in bottom_n['harmonic_centrality']],
            textposition='outside',
        ), row=1, col=2)

    fig.update_layout(
        height=max(400, n * 30),
        template='plotly_white',
        showlegend=False,
        font=dict(family='Inter', size=11),
        title=dict(text="Top vs Bottom Pages — Harmonic Centrality Gap", font=dict(size=16, color=TEXT_COLOR)),
    )
    return fig


# ──────────────────────── PDF REPORT ───────────────────────────────────

def sanitize_pdf_text(text):
    """Replace Unicode characters that may cause issues with PDF fonts."""
    replacements = {
        '\u2014': '--',   # em dash
        '\u2013': '-',    # en dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...',  # ellipsis
        '\u2022': '*',    # bullet
        '\u00a0': ' ',    # non-breaking space
        '\u2032': "'",   # prime
        '\u2033': '"',   # double prime
        '\u2212': '-',    # minus sign
        '\u00d7': 'x',   # multiplication sign
        '\u2264': '<=',  # less than or equal
        '\u2265': '>=',  # greater than or equal
        '\u2260': '!=',  # not equal
        '\u03b1': 'alpha',  # alpha
        '\u03b2': 'beta',   # beta
        '\u2192': '->',     # right arrow
        '\u2190': '<-',     # left arrow
        '\u2211': 'SUM',    # summation
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Remove any remaining non-latin1 characters as a safety net
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text


class PDFReport(FPDF):
    """Custom PDF report with branding."""

    def __init__(self, site_url):
        super().__init__()
        self.site_url = sanitize_pdf_text(site_url)
        self.set_auto_page_break(auto=True, margin=20)

    def _safe(self, text):
        """Sanitize text before rendering in PDF."""
        return sanitize_pdf_text(str(text))

    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(32, 128, 141)
        self.cell(0, 8, 'Harmonic Centrality Report', align='L')
        self.cell(0, 8, self.site_url, align='R', new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(32, 128, 141)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(122, 121, 116)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def add_cover(self, stats):
        self.add_page()
        self.ln(40)
        self.set_font('Helvetica', 'B', 28)
        self.set_text_color(40, 37, 29)
        self.cell(0, 15, 'Harmonic Centrality', align='C', new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 15, 'Analysis Report', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(8)
        self.set_font('Helvetica', '', 14)
        self.set_text_color(122, 121, 116)
        self.cell(0, 10, self.site_url, align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_font('Helvetica', '', 11)
        self.cell(0, 8, f"Generated: {time.strftime('%B %d, %Y')}", align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(20)

        # Summary stats
        self.set_draw_color(32, 128, 141)
        self.set_line_width(0.3)
        y_start = self.get_y()
        col_w = 45
        x_start = (210 - col_w * 4) / 2

        for i, (label, value) in enumerate(stats.items()):
            x = x_start + i * col_w
            self.set_xy(x, y_start)
            self.set_font('Helvetica', 'B', 18)
            self.set_text_color(32, 128, 141)
            self.cell(col_w, 10, str(value), align='C', new_x="LMARGIN", new_y="NEXT")
            self.set_xy(x, y_start + 10)
            self.set_font('Helvetica', '', 8)
            self.set_text_color(122, 121, 116)
            self.cell(col_w, 6, label, align='C')

    def add_section(self, title):
        self.ln(8)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(32, 128, 141)
        self.cell(0, 10, self._safe(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(32, 128, 141)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 100, self.get_y())
        self.ln(4)

    def add_paragraph(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 37, 29)
        self.multi_cell(0, 5.5, self._safe(text))
        self.ln(2)

    def add_table(self, headers, data, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(32, 128, 141)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, self._safe(h), border=1, fill=True, align='C')
        self.ln()

        # Rows
        self.set_font('Helvetica', '', 8)
        self.set_text_color(40, 37, 29)
        for row_idx, row in enumerate(data):
            if row_idx % 2 == 0:
                self.set_fill_color(247, 246, 242)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, self._safe(str(cell)[:50]), border=1, fill=True, align='C' if i > 0 else 'L')
            self.ln()

    def add_chart_image(self, fig, title=""):
        """Save a plotly figure as image and embed in PDF."""
        try:
            img_bytes = fig.to_image(format="png", width=900, height=500, scale=2)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(img_bytes)
                f.flush()
                if self.get_y() > 180:
                    self.add_page()
                if title:
                    self.set_font('Helvetica', 'B', 11)
                    self.set_text_color(40, 37, 29)
                    self.cell(0, 8, self._safe(title), new_x="LMARGIN", new_y="NEXT")
                self.image(f.name, x=10, w=190)
                self.ln(4)
            os.unlink(f.name)
        except Exception:
            # If kaleido not available, skip image
            if title:
                self.add_paragraph(f"[Chart: {title} -- install kaleido for chart images in PDF]")


def generate_pdf_report(site_url, df, G, metrics, figures_dict, page_info, orphans, weakly_linked):
    """Generate a comprehensive PDF report."""
    pdf = PDFReport(site_url)
    pdf.alias_nb_pages()

    total_pages = len(df)
    max_hc = df['harmonic_centrality'].max()
    avg_hc = df['harmonic_centrality'].mean()
    orphan_count = len(orphans)

    # Cover
    pdf.add_cover({
        'Pages Analyzed': total_pages,
        'Max HC Score': f"{max_hc:.2f}",
        'Avg HC Score': f"{avg_hc:.3f}",
        'Orphan Pages': orphan_count,
    })

    # Executive Summary
    pdf.add_page()
    pdf.add_section("Executive Summary")

    high_count = len(df[df['tier'] == 'high'])
    med_count = len(df[df['tier'] == 'medium'])
    low_count = len(df[df['tier'] == 'low'])
    zero_count = len(df[df['tier'] == 'zero'])

    summary = (
        f"This report analyzes the internal link architecture of {site_url} using harmonic centrality, "
        f"a graph theory metric that measures how reachable each page is within the link network. "
        f"Unlike closeness centrality, harmonic centrality handles disconnected graphs correctly, "
        f"making it ideal for real websites where orphan pages and broken link clusters exist.\n\n"
        f"Key Findings:\n"
        f"- {total_pages} pages were analyzed with {G.number_of_edges()} internal links discovered.\n"
        f"- {high_count} pages ({high_count/total_pages*100:.1f}%) have HIGH centrality (well-connected, easily discoverable).\n"
        f"- {med_count} pages ({med_count/total_pages*100:.1f}%) have MEDIUM centrality.\n"
        f"- {low_count} pages ({low_count/total_pages*100:.1f}%) have LOW centrality (hard to discover).\n"
        f"- {zero_count} pages ({zero_count/total_pages*100:.1f}%) are ORPHAN pages with zero centrality (not reachable via links).\n"
        f"- Average graph density: {nx.density(G):.4f}"
    )
    pdf.add_paragraph(summary)

    # What is Harmonic Centrality
    pdf.add_section("What is Harmonic Centrality?")
    pdf.add_paragraph(
        "Harmonic centrality measures how reachable a page is from every other page in the link network. "
        "The formula is: H(v) = SUM of 1/d(v,u) for all nodes u != v, where d(v,u) is the shortest path distance. "
        "Pages close to many other pages score high; unreachable pages contribute zero. "
        "This metric directly correlates with crawl frequency, link equity flow, and AI citation probability.\n\n"
        "Reference: 'What Is Harmonic Centrality? Graph Theory, PageRank, and Modern SEO' — searchministry.au"
    )

    # Top pages table
    pdf.add_section("Top 30 Pages by Harmonic Centrality")
    top30 = df.nlargest(30, 'harmonic_centrality')
    headers = ['Page Path', 'HC Score', 'PageRank', 'In-Links', 'Out-Links', 'Tier']
    data = []
    for _, row in top30.iterrows():
        data.append([
            row['path'][:45],
            f"{row['harmonic_centrality']:.4f}",
            f"{row['pagerank']:.6f}",
            str(int(row['in_degree'])),
            str(int(row['out_degree'])),
            row['tier'].upper(),
        ])
    pdf.add_table(headers, data, col_widths=[65, 25, 25, 20, 20, 20])

    # Orphan pages
    if orphans:
        pdf.add_page()
        pdf.add_section(f"Orphan Pages ({len(orphans)} found)")
        pdf.add_paragraph(
            "These pages have ZERO inbound internal links and cannot be discovered through link traversal. "
            "They receive no link equity and are unlikely to be crawled regularly. "
            "Action: Add each orphan page to relevant cluster pages, or redirect/remove if no longer needed."
        )
        orphan_data = [[get_url_path(u)[:80]] for u in orphans[:50]]
        pdf.add_table(['Orphan Page URL'], orphan_data, col_widths=[190])

    # Weakly linked
    if weakly_linked:
        pdf.add_section(f"Weakly Linked Pages ({len(weakly_linked)} with only 1 inbound link)")
        pdf.add_paragraph(
            "These pages have only ONE inbound internal link. If that single link breaks, "
            "they become orphans. Consider adding additional internal links to strengthen their position."
        )
        weak_data = [[get_url_path(u)[:80]] for u in weakly_linked[:30]]
        pdf.add_table(['Weakly Linked Page'], weak_data, col_widths=[190])

    # Charts
    pdf.add_page()
    pdf.add_section("Visualizations")
    for chart_title, fig in figures_dict.items():
        pdf.add_chart_image(fig, chart_title)

    # Recommendations
    pdf.add_page()
    pdf.add_section("Recommendations")

    recs = [
        ("1. Fix Orphan Pages", 
         f"You have {len(orphans)} orphan pages with zero inbound internal links. "
         "Add contextual internal links from relevant cluster/pillar pages to each orphan. "
         "Pages that are no longer needed should be redirected (301) to relevant alternatives."),
        ("2. Strengthen Pillar Pages",
         "Ensure your highest-value pages (service pages, key landing pages) are linked from the homepage "
         "and persistent navigation. Aim for single-hop access from as many pages as possible to maximize their HC score."),
        ("3. Cross-Link Cluster Pages",
         "When a cluster page mentions a topic covered by a sibling page, add a contextual internal link. "
         "This reduces average path length and raises centrality across the entire cluster."),
        ("4. Reduce Click Depth",
         "Pages beyond depth 3 receive significantly less crawl budget and link equity. "
         "Flatten your architecture by adding hub pages or breadcrumb navigation."),
        ("5. Monitor Low-HC Pages",
         "Pages with low harmonic centrality are structurally disadvantaged. "
         "If they target important keywords, add more internal links pointing to them."),
    ]

    for title, text in recs:
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(32, 128, 141)
        pdf.cell(0, 8, sanitize_pdf_text(title), new_x="LMARGIN", new_y="NEXT")
        pdf.add_paragraph(text)

    # Methodology
    pdf.add_page()
    pdf.add_section("Methodology")
    pdf.add_paragraph(
        "Crawl Method: BFS (Breadth-First Search) starting from the provided URL. "
        "Only HTML pages on the same domain are included. External links, media files, "
        "and non-HTML resources are excluded.\n\n"
        "Graph Construction: A directed graph where each node is a page and each edge is an internal link. "
        "Self-loops are excluded.\n\n"
        "Metrics Computed:\n"
        "- Harmonic Centrality: H(v) = SUM(1/d(v,u)) for all u != v\n"
        "- PageRank: Google's original link authority algorithm (damping=0.85)\n"
        "- Betweenness Centrality: How often a page lies on shortest paths between other pages\n"
        "- Closeness Centrality: Inverse of average distance to all other nodes\n"
        "- HITS Authority & Hub Scores: Kleinberg's hyperlink-induced topic search\n\n"
        "Reference: Tharindu Gunawardana, 'What Is Harmonic Centrality? Graph Theory, PageRank, and Modern SEO', "
        "searchministry.au, March 2026."
    )

    # Sanitize all page titles/paths that may contain unicode from crawled pages
    # (already handled by _safe method in class, but ensure summary text is clean too)

    return bytes(pdf.output())


# ─────────────────────── MAIN APP ──────────────────────────────────────

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        
        url_input = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            help="Enter the full URL including https://",
        )

        max_pages = st.slider(
            "Max pages to crawl",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="More pages = better analysis but slower crawl",
        )

        crawl_delay = st.slider(
            "Crawl delay (seconds)",
            min_value=0.1,
            max_value=2.0,
            value=0.3,
            step=0.1,
            help="Be respectful — increase delay for smaller sites",
        )

        network_nodes = st.slider(
            "Network graph nodes",
            min_value=20,
            max_value=200,
            value=80,
            step=10,
            help="Number of top nodes to show in network visualization",
        )

        start_crawl = st.button("Analyze Website", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "Harmonic centrality measures how reachable a page is within an internal link network. "
            "High scores indicate pages that are close to many others — they get more crawl budget, "
            "link equity, and AI citation probability."
        )
        st.markdown(
            "[Reference: What Is Harmonic Centrality?](https://searchministry.au/guides/what-is-harmonic-centrality)"
        )

    # Header
    st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <h1 style="color:#28251D; font-weight:700; margin-bottom:0;">
                Harmonic Centrality Analyzer
            </h1>
            <p style="color:#7A7974; font-size:1.1rem; margin-top:4px;">
                Advanced Internal Link Architecture Analysis for SEO
            </p>
        </div>
    """, unsafe_allow_html=True)

    if not start_crawl:
        # Landing state
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:2rem; margin-bottom:8px;">📊</div>
                <h4 style="color:#28251D;">Graph Metrics</h4>
                <p style="color:#7A7974; font-size:0.9rem;">
                    Harmonic centrality, PageRank, betweenness, closeness, and HITS scores
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:2rem; margin-bottom:8px;">🕸️</div>
                <h4 style="color:#28251D;">Network Visualization</h4>
                <p style="color:#7A7974; font-size:0.9rem;">
                    Interactive graph, treemaps, heatmaps, and radar charts
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size:2rem; margin-bottom:8px;">📄</div>
                <h4 style="color:#28251D;">PDF Export</h4>
                <p style="color:#7A7974; font-size:0.9rem;">
                    Download a comprehensive report with all findings and recommendations
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box" style="margin-top:24px;">
            <h4>How It Works</h4>
            <p>
                <b>1.</b> Enter your website URL and click Analyze.<br>
                <b>2.</b> The crawler discovers internal links via BFS traversal.<br>
                <b>3.</b> A directed graph is built and harmonic centrality is computed: <code>H(v) = Σ 1/d(v,u)</code><br>
                <b>4.</b> Results are compared with PageRank, betweenness, and HITS scores.<br>
                <b>5.</b> Download the full report as PDF to share with your team.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Validate URL
    if not url_input:
        st.error("Please enter a website URL.")
        return

    if not url_input.startswith(('http://', 'https://')):
        url_input = 'https://' + url_input

    try:
        parsed = urlparse(url_input)
        if not parsed.netloc:
            st.error("Invalid URL. Please enter a valid website address.")
            return
    except Exception:
        st.error("Invalid URL format.")
        return

    # ─── CRAWLING ────
    st.markdown('<div class="section-header">Crawling Website</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()
    edges, visited, page_info = crawl_website(
        url_input, max_pages=max_pages, delay=crawl_delay,
        progress_bar=progress_bar, status_text=status_text,
    )
    crawl_time = time.time() - start_time

    progress_bar.progress(1.0)
    status_text.text(f"Crawl complete: {len(visited)} pages in {crawl_time:.1f}s")

    if len(visited) < 2:
        st.error("Could not crawl enough pages. Check the URL and try again.")
        return

    # ─── BUILD GRAPH ────
    st.markdown('<div class="section-header">Building Link Graph & Computing Metrics</div>', unsafe_allow_html=True)

    G = build_graph(edges, visited)
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    metrics = compute_all_metrics(G)

    # Build master DataFrame
    all_nodes = list(G.nodes())
    hc_scores = metrics['harmonic_centrality']
    max_hc = max(hc_scores.values()) if hc_scores else 0

    records = []
    for node in all_nodes:
        rec = {
            'url': node,
            'path': get_url_path(node),
            'harmonic_centrality': hc_scores.get(node, 0),
            'pagerank': metrics['pagerank'].get(node, 0),
            'betweenness_centrality': metrics['betweenness_centrality'].get(node, 0),
            'closeness_centrality': metrics['closeness_centrality'].get(node, 0),
            'hub_score': metrics['hub_score'].get(node, 0),
            'authority_score': metrics['authority_score'].get(node, 0),
            'in_degree': metrics['in_degree'].get(node, 0),
            'out_degree': metrics['out_degree'].get(node, 0),
            'tier': classify_score(hc_scores.get(node, 0), max_hc),
        }
        info = page_info.get(node, {})
        rec['title'] = info.get('title', '')
        rec['word_count'] = info.get('word_count', 0)
        rec['status_code'] = info.get('status_code', 0)
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.sort_values('harmonic_centrality', ascending=False).reset_index(drop=True)

    orphans, weakly_linked = create_orphan_analysis(G, all_nodes)

    # ─── KPI CARDS ────
    st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(visited)}</div>
            <div class="metric-label">Pages Crawled</div>
            <div class="metric-sublabel">in {crawl_time:.1f}s</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{G.number_of_edges():,}</div>
            <div class="metric-label">Internal Links</div>
            <div class="metric-sublabel">{G.number_of_edges()/len(visited):.1f} per page</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{max_hc:.2f}</div>
            <div class="metric-label">Highest HC</div>
            <div class="metric-sublabel">{df.iloc[0]['path'][:25] if len(df) > 0 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        avg_hc = df['harmonic_centrality'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_hc:.3f}</div>
            <div class="metric-label">Avg HC Score</div>
            <div class="metric-sublabel">median: {df['harmonic_centrality'].median():.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#A84B2F;">{len(orphans)}</div>
            <div class="metric-label">Orphan Pages</div>
            <div class="metric-sublabel">{len(orphans)/len(all_nodes)*100:.1f}% of total</div>
        </div>
        """, unsafe_allow_html=True)

    with k6:
        density = nx.density(G)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{density:.4f}</div>
            <div class="metric-label">Graph Density</div>
            <div class="metric-sublabel">{'Sparse' if density < 0.05 else 'Moderate' if density < 0.2 else 'Dense'}</div>
        </div>
        """, unsafe_allow_html=True)

    # ─── INSIGHT BOX ────
    top_page = df.iloc[0] if len(df) > 0 else None
    if top_page is not None:
        high_pct = len(df[df['tier'] == 'high']) / len(df) * 100
        orphan_pct = len(orphans) / len(all_nodes) * 100

        st.markdown(f"""
        <div class="insight-box">
            <h4>Key Insight</h4>
            <p>
                <b>{top_page['path']}</b> is the most reachable page with an HC score of <b>{top_page['harmonic_centrality']:.4f}</b>.
                {high_pct:.0f}% of pages have high centrality, while <b>{orphan_pct:.0f}% are orphaned</b> with no inbound links.
                {'The orphan rate is concerning — these pages get zero link equity and minimal crawl budget.' if orphan_pct > 10 else 'The orphan rate is acceptable but could be improved.'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ─── TAB-BASED VISUALIZATIONS ────
    st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)

    figures_dict = {}

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Top Pages", "Distribution", "HC vs PageRank",
        "Network Graph", "Click Depth", "Link Profile",
        "Tier Breakdown", "Centrality Radar"
    ])

    with tab1:
        fig_top = create_top_pages_chart(df, n=min(25, len(df)))
        st.plotly_chart(fig_top, use_container_width=True)
        figures_dict["Top Pages by Harmonic Centrality"] = fig_top

        # Top vs Bottom comparison
        fig_tb = create_top_bottom_comparison(df, n=10)
        st.plotly_chart(fig_tb, use_container_width=True)
        figures_dict["Top vs Bottom Pages"] = fig_tb

    with tab2:
        fig_dist = create_distribution_chart(df)
        st.plotly_chart(fig_dist, use_container_width=True)
        figures_dict["HC Distribution"] = fig_dist

        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribution Statistics**")
            stats_df = df['harmonic_centrality'].describe()
            st.dataframe(stats_df.to_frame('Harmonic Centrality'), use_container_width=True)
        with col2:
            st.markdown("**Percentiles**")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            pct_data = {f"P{p}": [df['harmonic_centrality'].quantile(p/100)] for p in percentiles}
            st.dataframe(pd.DataFrame(pct_data), use_container_width=True)

    with tab3:
        fig_scatter = create_hc_vs_pagerank(df)
        st.plotly_chart(fig_scatter, use_container_width=True)
        figures_dict["HC vs PageRank"] = fig_scatter

        st.markdown("""
        <div class="insight-box">
            <h4>Understanding the Comparison</h4>
            <p>
                <b>Harmonic Centrality</b> measures structural reachability — how easy a page is to discover through link traversal.<br>
                <b>PageRank</b> measures accumulated authority — how much link equity flows to a page.<br>
                Pages in the <b>top-right</b> are both highly reachable AND authoritative (your strongest pages).<br>
                Pages with <b>high HC but low PR</b> are easy to find but lack authority — consider building more links to them.<br>
                Pages with <b>low HC but high PR</b> receive authority but are hard to find — improve their internal linking.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        fig_network = create_network_graph(G, hc_scores, max_nodes=network_nodes)
        st.plotly_chart(fig_network, use_container_width=True)
        figures_dict["Network Graph"] = fig_network

    with tab5:
        fig_depth, depth_df = create_depth_analysis(G, normalize_url(url_input))
        if fig_depth:
            st.plotly_chart(fig_depth, use_container_width=True)
            figures_dict["Click Depth Distribution"] = fig_depth

            # Treemap
            fig_treemap = create_crawl_depth_treemap(depth_df, hc_scores)
            if fig_treemap:
                st.plotly_chart(fig_treemap, use_container_width=True)
                figures_dict["Depth Treemap"] = fig_treemap

            # Depth stats
            if depth_df is not None and len(depth_df) > 0:
                st.markdown("**Pages by Depth Level**")
                depth_summary = depth_df.groupby('depth').agg(
                    pages=('url', 'count'),
                ).reset_index()
                depth_summary.columns = ['Depth', 'Pages']
                st.dataframe(depth_summary, use_container_width=True)
        else:
            st.warning("Could not compute click depth — homepage may not be in the crawled set.")

    with tab6:
        fig_link = create_link_equity_heatmap(df)
        st.plotly_chart(fig_link, use_container_width=True)
        figures_dict["Link Profile"] = fig_link

    with tab7:
        fig_tier = create_tier_breakdown(df)
        st.plotly_chart(fig_tier, use_container_width=True)
        figures_dict["Tier Breakdown"] = fig_tier

    with tab8:
        if len(df) >= 3:
            fig_radar = create_centrality_comparison(df)
            st.plotly_chart(fig_radar, use_container_width=True)
            figures_dict["Centrality Radar"] = fig_radar
        else:
            st.info("Need at least 3 pages to generate radar comparison.")

    # ─── ORPHAN & WEAK PAGES ────
    st.markdown('<div class="section-header">Structural Issues</div>', unsafe_allow_html=True)

    issue_col1, issue_col2 = st.columns(2)

    with issue_col1:
        st.markdown(f"**Orphan Pages ({len(orphans)})**")
        if orphans:
            orphan_df = pd.DataFrame({
                'Page': [get_url_path(u) for u in orphans],
                'URL': orphans,
            })
            st.dataframe(orphan_df, use_container_width=True, height=300)
        else:
            st.success("No orphan pages found.")

    with issue_col2:
        st.markdown(f"**Weakly Linked Pages ({len(weakly_linked)}) — Only 1 inbound link**")
        if weakly_linked:
            weak_df = pd.DataFrame({
                'Page': [get_url_path(u) for u in weakly_linked],
                'In-Degree': [1] * len(weakly_linked),
            })
            st.dataframe(weak_df, use_container_width=True, height=300)
        else:
            st.success("No weakly linked pages found.")

    # ─── FULL DATA TABLE ────
    st.markdown('<div class="section-header">Full Data Table</div>', unsafe_allow_html=True)

    display_df = df[['path', 'title', 'harmonic_centrality', 'pagerank',
                      'betweenness_centrality', 'closeness_centrality',
                      'in_degree', 'out_degree', 'authority_score', 'hub_score',
                      'word_count', 'tier']].copy()
    display_df.columns = ['Path', 'Title', 'Harmonic Centrality', 'PageRank',
                          'Betweenness', 'Closeness', 'In-Links', 'Out-Links',
                          'Authority', 'Hub Score', 'Words', 'Tier']

    st.dataframe(
        display_df.style.background_gradient(subset=['Harmonic Centrality'], cmap='YlGnBu'),
        use_container_width=True,
        height=500,
    )

    # CSV download
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Full Data (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"harmonic_centrality_{parsed.netloc}_{time.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    # ─── PDF REPORT ────
    st.markdown('<div class="section-header">Download Full Report</div>', unsafe_allow_html=True)

    try:
        pdf_bytes = generate_pdf_report(
            url_input, df, G, metrics, figures_dict,
            page_info, orphans, weakly_linked,
        )
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"harmonic_centrality_report_{parsed.netloc}_{time.strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary",
        )
    except Exception as e:
        st.warning(f"PDF generation encountered an issue: {str(e)}. CSV download is still available above.")

    # ─── RECOMMENDATIONS ────
    st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)

    if len(orphans) > 0:
        st.markdown(f"""
        <div class="insight-box">
            <h4>1. Fix {len(orphans)} Orphan Pages</h4>
            <p>
                These pages have zero inbound internal links and cannot be discovered through link traversal.
                They receive no link equity and are unlikely to be crawled regularly.
                Add each orphan to relevant cluster pages or redirect/remove if no longer needed.
            </p>
        </div>
        """, unsafe_allow_html=True)

    if len(weakly_linked) > 0:
        st.markdown(f"""
        <div class="insight-box">
            <h4>2. Strengthen {len(weakly_linked)} Weakly Linked Pages</h4>
            <p>
                These pages depend on a single internal link. If that link breaks, they become orphans.
                Add at least 2-3 contextual internal links from related content.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # General recommendations
    st.markdown("""
    <div class="insight-box">
        <h4>3. Cross-Link Cluster Pages</h4>
        <p>
            When a cluster page mentions a topic covered by a sibling page, add a contextual internal link.
            This reduces average path length and raises centrality scores across the entire cluster.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <h4>4. Strengthen Pillar Page Centrality</h4>
        <p>
            Link to key service and category pages from the homepage, related cluster pages, and persistent navigation.
            Aim for single-hop access from as many pages as possible.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#BAB9B4; font-size:0.85rem;">'
        'Based on <a href="https://searchministry.au/guides/what-is-harmonic-centrality" target="_blank">'
        '"What Is Harmonic Centrality? Graph Theory, PageRank, and Modern SEO"</a> by Tharindu Gunawardana'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
