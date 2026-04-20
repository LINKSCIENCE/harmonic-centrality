"""
Harmonic Centrality Analyzer — Advanced SEO Internal Link Graph Tool
Crawls a website, builds the internal link graph, computes harmonic centrality
and other graph metrics, provides advanced visualizations + PDF export.
Session-state persists all crawl results so no progress is ever lost.
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

PALETTE = ['#20808D','#A84B2F','#1B474D','#BCE2E7','#944454',
           '#FFC553','#848456','#6E522B']
TEAL = '#20808D'
DARK_TEAL = '#1B474D'
BG = '#F7F6F2'
TEXT_COLOR = '#28251D'
MUTED = '#7A7974'

CATEGORY_COLORS = [  # FIX 1: was missing closing ]
    '#20808D','#A84B2F','#FFC553','#944454','#1B474D',
    '#6E522B','#848456','#BCE2E7','#D95F5F','#5F9EA0',
    '#8B6914','#3D7A5E','#C47B3A','#5A5F8C','#8E3A59',
    '#4A7C59','#7B4F9E','#C4823A','#3A6B8C','#9E6B4F',
]  # <-- was missing

st.set_page_config(
    page_title="Harmonic Centrality Analyzer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)  # FIX 2: was missing closing )

# ─────────────────────────── CUSTOM CSS ────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: white; border-radius: 12px; padding: 20px;
    border: 1px solid #E8E6E0; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    text-align: center; height: 100%;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #20808D; line-height: 1.2; }
.metric-label { font-size: 0.85rem; color: #7A7974; font-weight: 500; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-sublabel { font-size: 0.75rem; color: #BAB9B4; margin-top: 2px; }
.section-header { font-size: 1.3rem; font-weight: 700; color: #28251D; padding: 16px 0 8px 0; border-bottom: 2px solid #20808D; margin-bottom: 20px; }
.insight-box { background: linear-gradient(135deg, #F0F8F9 0%, #E8F4F5 100%); border-left: 4px solid #20808D; border-radius: 8px; padding: 16px 20px; margin: 12px 0; }
.insight-box h4 { color: #1B474D; margin-bottom: 8px; font-size: 1rem; }
.insight-box p { color: #28251D; margin: 0; font-size: 0.9rem; line-height: 1.6; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #20808D, #1B474D); color: white; border: none; border-radius: 8px; font-weight: 600; padding: 0.6rem 1.2rem; }
.stButton > button[kind="primary"]:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SESSION-STATE HELPERS
# ══════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = dict(
        crawl_done=False,
        crawl_url="",
        crawl_df=None,
        crawl_G=None,
        crawl_metrics=None,
        crawl_page_info=None,
        crawl_edges=None,
        crawl_visited=None,
        crawl_time=0.0,
    )  # FIX 3: was missing closing )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _clear_state():
    for k in ["crawl_done","crawl_url","crawl_df","crawl_G",
              "crawl_metrics","crawl_page_info","crawl_edges",
              "crawl_visited","crawl_time"]:
        st.session_state[k] = None if k not in ("crawl_done",) else False
    st.session_state["crawl_time"] = 0.0

# ─────────────────────────── CRAWLER ───────────────────────────────────

def normalize_url(url):
    url, _ = urldefrag(url)
    return url.rstrip('/')

def is_internal(url, base_domain):
    try:
        return urlparse(url).netloc == base_domain
    except Exception:
        return False

def crawl_website(start_url, max_pages=100, delay=0.3,
                  progress_bar=None, status_text=None):
    start_url = normalize_url(start_url)
    base_domain = urlparse(start_url).netloc
    visited = set()
    edges = []
    queue = deque([start_url])
    page_info = {}
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; HarmonicCentralityBot/1.0)'}

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            if 'text/html' not in resp.headers.get('Content-Type', ''):
                continue
            visited.add(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            title_tag = soup.find('title')
            title_text = title_tag.get_text(strip=True) if title_tag else ''
            words = len(soup.get_text().split())
            page_info[url] = {
                'title': title_text,
                'word_count': words,
                'status_code': resp.status_code,
            }  # FIX 4: was missing closing }
            for a in soup.find_all('a', href=True):
                href = a['href'].strip()
                if not href or href.startswith(('#','mailto:','tel:','javascript:')):
                    continue
                abs_url = normalize_url(urljoin(url, href))
                if is_internal(abs_url, base_domain):
                    edges.append((url, abs_url))
                    if abs_url not in visited:
                        queue.append(abs_url)
            pct = len(visited) / max_pages
            if progress_bar:
                progress_bar.progress(min(pct, 0.99))
            if status_text:
                status_text.text(f"Crawling: {url[:80]}… ({len(visited)}/{max_pages})")
            time.sleep(delay)
        except Exception:
            visited.add(url)

    return edges, visited, page_info

# ─────────────────────── GRAPH ANALYSIS ────────────────────────────────

def build_graph(edges, visited):
    G = nx.DiGraph()
    G.add_nodes_from(visited)
    for src, dst in edges:
        if src in visited and dst in visited and src != dst:
            G.add_edge(src, dst)
    return G

def compute_all_metrics(G):
    UG = G.to_undirected()
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {k:{} for k in ['harmonic_centrality','pagerank',
                                'betweenness_centrality','closeness_centrality',
                                'hub_score','authority_score','in_degree','out_degree']}
    hc = nx.harmonic_centrality(UG)
    max_hc = max(hc.values()) if hc else 1
    hc = {k: v/max_hc for k, v in hc.items()}
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        pr = {nd: 1/len(nodes) for nd in nodes}
    try:
        bc = nx.betweenness_centrality(G, normalized=True, k=min(100, n))
    except Exception:
        bc = {nd: 0 for nd in nodes}
    try:
        cc = nx.closeness_centrality(G)
    except Exception:
        cc = {nd: 0 for nd in nodes}
    try:
        hubs, auths = nx.hits(G, max_iter=200, normalized=True)
    except Exception:
        hubs = auths = {nd: 0 for nd in nodes}
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    return dict(harmonic_centrality=hc, pagerank=pr,
                betweenness_centrality=bc, closeness_centrality=cc,
                hub_score=hubs, authority_score=auths,
                in_degree=in_deg, out_degree=out_deg)

def classify_score(score, max_score):
    if max_score == 0:
        return 'low'
    ratio = score / max_score
    if ratio >= 0.7: return 'high'
    if ratio >= 0.4: return 'medium'
    return 'low'

def get_url_path(url):
    try:
        p = urlparse(url).path
        return p if p else '/'
    except Exception:
        return url

# ─────────────────────── VISUALIZATIONS ────────────────────────────────

def create_top_pages_chart(df, n=20):
    top = df.head(n).copy()
    top['path_short'] = top['path'].apply(lambda x: (x[:45]+'…') if len(x)>45 else x)
    colors = [TEAL if t=='high' else ('#A84B2F' if t=='medium' else MUTED) for t in top['tier']]
    fig = go.Figure(go.Bar(
        x=top['harmonic_centrality'],
        y=top['path_short'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.4f}" for v in top['harmonic_centrality']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>HC Score: %{x:.4f}<extra></extra>',  # FIX 5: was garbled
    ))
    fig.update_layout(
        title=dict(text=f'Top {n} Pages by Harmonic Centrality', font=dict(size=16, color=TEXT_COLOR)),
        xaxis_title='Harmonic Centrality Score',
        yaxis=dict(autorange='reversed', tickfont=dict(size=10), automargin=True),
        height=max(420, n*30),
        margin=dict(l=160, r=80, t=50, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig

def create_distribution_chart(df):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('HC Score Distribution', 'Cumulative Distribution'))
    fig.add_trace(go.Histogram(
        x=df['harmonic_centrality'], nbinsx=30,
        marker_color=TEAL, opacity=0.8, name='HC Distribution',
    ), row=1, col=1)
    sorted_hc = df['harmonic_centrality'].sort_values().values
    pct = np.arange(1, len(sorted_hc)+1) / len(sorted_hc) * 100
    fig.add_trace(go.Scatter(
        x=sorted_hc, y=pct, mode='lines',
        line=dict(color=TEAL, width=2),
        fill='tozeroy', fillcolor='rgba(32,128,141,0.15)',
        name='Cumulative',
    ), row=1, col=2)
    fig.update_layout(height=380, showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
                      margin=dict(l=20, r=20, t=50, b=40))
    fig.update_xaxes(showgrid=True, gridcolor='#F0EDE8')
    fig.update_yaxes(showgrid=True, gridcolor='#F0EDE8')
    return fig

def create_hc_vs_pagerank(df):
    fig = px.scatter(
        df, x='pagerank', y='harmonic_centrality', color='tier',
        size='in_degree', hover_data={'path': True, 'harmonic_centrality': ':.4f',
                                       'pagerank': ':.6f', 'in_degree': True},
        color_discrete_map={'high': TEAL, 'medium': '#A84B2F', 'low': MUTED},
        size_max=20,
    )
    fig.update_layout(
        title='HC Score vs PageRank', xaxis_title='PageRank Score',
        yaxis_title='Harmonic Centrality', height=450,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=70, r=20, t=50, b=50),
    )
    fig.update_yaxes(title_standoff=12)
    fig.update_xaxes(title_standoff=10, showgrid=True, gridcolor='#F0EDE8')
    fig.update_yaxes(showgrid=True, gridcolor='#F0EDE8')
    return fig

def create_depth_analysis(G, start_url):
    if start_url not in G.nodes:
        candidates = [n for n in G.nodes if '/' not in n.rstrip('/').replace('https://','').replace('http://','')]
        if not candidates:
            return None, None
        start_url = candidates[0]
    try:
        lengths = nx.single_source_shortest_path_length(G, start_url)
    except Exception:
        return None, None
    depth_data = [{'url': u, 'depth': d, 'path': get_url_path(u)} for u, d in lengths.items()]
    depth_df = pd.DataFrame(depth_data)
    depth_counts = depth_df.groupby('depth').size().reset_index(name='count')
    fig = go.Figure(go.Bar(
        x=depth_counts['depth'], y=depth_counts['count'],
        marker_color=TEAL, opacity=0.85,
        text=depth_counts['count'], textposition='outside',
    ))
    fig.update_layout(
        title='Pages by Click Depth from Homepage',
        xaxis_title='Click Depth', yaxis_title='Number of Pages',
        height=380, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig, depth_df

def create_link_equity_heatmap(df):
    df2 = df.head(30).copy()
    fig = go.Figure(go.Bar(
        x=df2['path'].apply(lambda x: x[:30] if len(x)>30 else x),
        y=df2['in_degree'],
        marker=dict(
            color=df2['harmonic_centrality'],
            colorscale='Teal', showscale=True,
            colorbar=dict(title='HC Score'),
        ),
        hovertemplate='<b>%{x}</b><br>In-Links: %{y}<br>HC: %{marker.color:.4f}<extra></extra>',
    ))
    fig.update_layout(
        title='Link Profile — Inbound Links Coloured by HC Score',
        xaxis_title='Page Path', yaxis_title='Inbound Links', height=420,
        xaxis_tickangle=-45, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=100),
    )
    return fig

def create_tier_breakdown(df):
    counts = df['tier'].value_counts()
    colors = {'high': TEAL, 'medium': '#A84B2F', 'low': MUTED}
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker_colors=[colors.get(l, MUTED) for l in counts.index],
        hole=0.45, textinfo='label+percent',
    ))
    fig.update_layout(title='Page Tier Distribution', height=380,
                      margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white')
    return fig

def create_centrality_comparison(df):
    top10 = df.head(10)
    metrics_list = ['harmonic_centrality','pagerank','betweenness_centrality',
                    'closeness_centrality','authority_score']
    labels = ['HC','PageRank','Betweenness','Closeness','Authority']
    fig = go.Figure()
    colors = list(PALETTE[:len(top10)])
    for i, (_, row) in enumerate(top10.iterrows()):
        vals = [row[m] for m in metrics_list]
        max_vals = [df[m].max() or 1 for m in metrics_list]
        norm = [v/mv for v, mv in zip(vals, max_vals)]
        norm.append(norm[0])
        theta = labels + [labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=norm, theta=theta, fill='toself', opacity=0.25,
            name=row['path'][:30],
            line=dict(color=colors[i % len(colors)], width=1.5),
        ))
    fig.update_layout(
        title='Centrality Radar — Top 10 Pages',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500, paper_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

def create_network_graph(G, hc_scores, max_nodes=80):
    nodes_sorted = sorted(hc_scores.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [n for n, _ in nodes_sorted[:max_nodes]]
    SG = G.subgraph(top_nodes)
    try:
        pos = nx.spring_layout(SG, seed=42, k=1.5)
    except Exception:
        pos = nx.random_layout(SG)
    max_hc = max(hc_scores.values()) if hc_scores else 1
    ex, ey = [], []
    for u, v in SG.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    edge_trace = go.Scatter(
        x=ex, y=ey, mode='lines',
        line=dict(width=0.5, color='#CCCCCC'), hoverinfo='none',
    )
    nx_vals = [hc_scores.get(n, 0) for n in SG.nodes()]
    node_trace = go.Scatter(
        x=[pos[n][0] for n in SG.nodes()],
        y=[pos[n][1] for n in SG.nodes()],
        mode='markers',
        hovertext=[f"<b>{get_url_path(n)}</b><br>HC: {hc_scores.get(n,0):.4f}" for n in SG.nodes()],  # FIX 6: was generator, not list
        hoverinfo='text',
        marker=dict(
            size=[max(6, min(30, hc_scores.get(n,0)/max_hc*28+4)) for n in SG.nodes()],
            color=nx_vals, colorscale='Teal', showscale=True,
            colorbar=dict(title='HC Score'),
            line=dict(width=1, color='white'),
        ),
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Internal Link Network (Top {max_nodes} pages)",
        showlegend=False, hovermode='closest', height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='white', plot_bgcolor='white',
    )
    return fig

def create_orphan_analysis(G, all_nodes):
    in_deg = dict(G.in_degree())
    orphans = [n for n in all_nodes if in_deg.get(n, 0) == 0]
    weakly_linked = [n for n in all_nodes if in_deg.get(n, 0) == 1]
    return orphans, weakly_linked

def create_crawl_depth_treemap(depth_df, hc_dict):
    if depth_df is None or depth_df.empty:
        return None
    depth_df = depth_df.copy()
    depth_df['hc'] = depth_df['url'].map(lambda u: hc_dict.get(u, 0))
    depth_df['depth_label'] = depth_df['depth'].apply(lambda d: f"Depth {d}")
    depth_df['path_short'] = depth_df['path'].apply(lambda x: (x[:35]+'…') if len(x)>35 else x)
    fig = px.treemap(
        depth_df, path=['depth_label','path_short'], values='hc', color='hc',
        color_continuous_scale='Teal', title="Crawl Depth Treemap — Sized by HC Score",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor='white', plot_bgcolor='white')
    return fig

def create_top_bottom_comparison(df, n=10):
    top_n = df.head(n).copy()
    bot_n = df.tail(n).copy()
    top_n['label'] = top_n['path'].apply(lambda x: x[:30]+'…' if len(x)>30 else x)
    bot_n['label'] = bot_n['path'].apply(lambda x: x[:30]+'…' if len(x)>30 else x)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Top {n} Pages", f"Bottom {n} Pages"))
    fig.add_trace(go.Bar(y=top_n['label'], x=top_n['harmonic_centrality'],
                         orientation='h', marker_color=TEAL, name='Top Pages'), row=1, col=1)
    fig.add_trace(go.Bar(y=bot_n['label'], x=bot_n['harmonic_centrality'],
                         orientation='h', marker_color='#A84B2F', name='Bottom Pages'), row=1, col=2)
    fig.update_layout(height=400, showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
                      margin=dict(l=20, r=20, t=50, b=40))
    fig.update_xaxes(showgrid=True, gridcolor='#F0EDE8')
    fig.update_yaxes(autorange='reversed')
    return fig

# ──────────────────────── PDF REPORT ──────────────────────────────────

def sanitize_pdf_text(text):
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '\u2019':"'",' \u2018':"'", '\u201c':'"', '\u201d':'"',
        '\u2013':'-', '\u2014':'--', '\u2026':'...', '\u00b0':'deg',
        '\u00e9':'e', '\u00e8':'e', '\u00ea':'e', '\u00eb':'e',
        '\u00e0':'a', '\u00e2':'a', '\u00e4':'a', '\u00f4':'o',
        '\u00f6':'o', '\u00fc':'u', '\u00fb':'u', '\u00e7':'c',
        '\u00f1':'n', '\u00ed':'i', '\u00f3':'o', '\u00fa':'u',
        '\u00e1':'a', '\u2022':'-',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return ''.join(c if ord(c) < 128 else '?' for c in text)


class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self._teal = (32, 128, 141)
        self._dark = (27, 71, 77)
        self._text = (40, 37, 29)
        self._muted = (122, 121, 116)
        self._bg = (247, 246, 242)

    def _safe(self, text, width=0, align='L', ln=True):
        try:
            s = sanitize_pdf_text(text)
            if ln:
                self.cell(width, self.font_size*1.5, s, ln=True, align=align)
            else:
                self.cell(width, self.font_size*1.5, s, align=align)
        except Exception:
            pass

    def header(self):
        self.set_fill_color(*self._teal)
        self.rect(0, 0, 220, 14, 'F')
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 3)
        self.cell(0, 8, 'Harmonic Centrality Analysis Report', align='L')
        self.set_xy(0, 3)
        self.cell(200, 8, 'Confidential', align='R')
        self.set_text_color(*self._text)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(*self._muted)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_cover(self, site_url, crawl_time, n_pages, n_links):
        self.add_page()
        self.set_fill_color(*self._bg)
        self.rect(10, 20, 190, 80, 'F')
        self.set_xy(10, 30)
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(*self._dark)
        self._safe('Harmonic Centrality', 190, 'C')
        self.set_font('Helvetica', '', 16)
        self.set_text_color(*self._teal)
        self._safe('Internal Link Architecture Analysis', 190, 'C')
        self.ln(6)
        self.set_font('Helvetica', '', 11)
        self.set_text_color(*self._muted)
        self._safe(sanitize_pdf_text(site_url), 190, 'C')
        self.ln(4)
        self._safe(f"Generated: {time.strftime('%Y-%m-%d %H:%M')} | Pages: {n_pages} | Links: {n_links}", 190, 'C')
        self.ln(20)

    def ensure_space(self, needed=24):
        if self.get_y() + needed > 270:
            self.add_page()

    def add_section(self, title):
        self.ensure_space(18)
        self.ln(4)
        y = self.get_y()
        self.set_fill_color(*self._teal)
        self.rect(10, y, 190, 10, 'F')
        self.set_xy(15, y + 1.2)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(255, 255, 255)
        self.cell(180, 7, sanitize_pdf_text(title), ln=False)
        self.set_text_color(*self._text)
        self.ln(10)

    def add_paragraph(self, text):
        self.ensure_space(14)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(*self._text)
        self.set_x(10)
        self.multi_cell(190, 6, sanitize_pdf_text(text))
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)
        self.ensure_space(16 + min(len(rows), 20) * 7)
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(*self._dark)
        self.set_text_color(255, 255, 255)
        self.set_x(10)
        for h, w in zip(headers, col_widths):
            self.cell(w, 8, sanitize_pdf_text(str(h)), border=0, fill=True, align='C')
        self.ln()
        self.set_font('Helvetica', '', 8)
        self.set_text_color(*self._text)
        for ri, row in enumerate(rows):
            if self.get_y() + 9 > 270:
                self.add_page()
                self.set_font('Helvetica', 'B', 9)
                self.set_fill_color(*self._dark)
                self.set_text_color(255, 255, 255)
                self.set_x(10)
                for h, w in zip(headers, col_widths):
                    self.cell(w, 8, sanitize_pdf_text(str(h)), border=0, fill=True, align='C')
                self.ln()
                self.set_font('Helvetica', '', 8)
                self.set_text_color(*self._text)
            if ri % 2 == 0:  # FIX 7: was ternary expression that could silently fail
                self.set_fill_color(240, 248, 249)
            else:
                self.set_fill_color(255, 255, 255)
            self.set_x(10)
            for val, w in zip(row, col_widths):
                self.cell(w, 7, sanitize_pdf_text(str(val))[:40], border=0, fill=True, align='L')
            self.ln()
        self.ln(4)

    def add_chart_image(self, fig, title=""):
        try:
            self.ensure_space(110)
            fig_pdf = go.Figure(fig)
            fig_pdf.update_layout(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(color='#28251D', family='Arial', size=12),
                margin=dict(l=90, r=40, t=35, b=70),
                title=None, showlegend=True,
            )
            try:
                fig_pdf.update_xaxes(showgrid=True, gridcolor='#EAE7E1', zeroline=False, automargin=True)
                fig_pdf.update_yaxes(showgrid=True, gridcolor='#EAE7E1', zeroline=False, automargin=True)
            except Exception:
                pass
            try:
                fig_pdf.update_traces(textposition='outside', cliponaxis=False)
            except Exception:
                pass
            img_bytes = fig_pdf.to_image(format='png', width=1200, height=700, scale=2)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            if title:
                self.set_font('Helvetica', 'B', 10)
                self.set_text_color(*self._dark)
                self.set_x(10)
                self.cell(190, 6, sanitize_pdf_text(title), ln=True)
            img_h = 100
            if self.get_y() + img_h > 270:
                self.add_page()
            self.image(tmp_path, x=10, y=self.get_y(), w=190, h=img_h)
            self.ln(img_h + 4)
            os.unlink(tmp_path)
        except Exception as e:
            self.add_paragraph(f"[Chart could not be rendered: {str(e)[:80]}]")


def generate_pdf_report(site_url, df, G, metrics, figures_dict, page_info, orphans, weakly_linked):
    pdf = PDFReport()
    hc_scores = metrics['harmonic_centrality']
    n_pages = len(df)
    n_links = G.number_of_edges()
    pdf.add_cover(site_url, 0, n_pages, n_links)

    # Executive summary
    pdf.add_section("Executive Summary")
    top = df.iloc[0] if len(df) > 0 else None
    max_hc = df['harmonic_centrality'].max() if len(df) > 0 else 0
    avg_hc = df['harmonic_centrality'].mean() if len(df) > 0 else 0
    high_pct = len(df[df['tier']=='high'])/len(df)*100 if len(df) > 0 else 0
    density = nx.density(G)
    pdf.add_paragraph(
        f"Analysis of {site_url} reveals a link graph with {n_pages} pages and "
        f"{n_links} internal links (density: {density:.4f}). "
        f"The most central page is '{get_url_path(top['url']) if top is not None else 'N/A'}' "
        f"with an HC score of {max_hc:.4f}. Average HC across the site is {avg_hc:.4f}. "
        f"{high_pct:.0f}% of pages hold high-centrality status. "
        f"{len(orphans)} orphan pages and {len(weakly_linked)} weakly-linked pages require attention."
    )

    # Top pages table
    pdf.add_section("Top 20 Pages by Harmonic Centrality")
    top20 = df.head(20)
    table_rows = [(get_url_path(r['url'])[:45], f"{r['harmonic_centrality']:.4f}",  # FIX 8: renamed rows -> table_rows
                   f"{r['pagerank']:.6f}", str(r['in_degree']), r['tier'])
                  for _, r in top20.iterrows()]
    pdf.add_table(['Path','HC Score','PageRank','In-Links','Tier'], table_rows, [70,25,30,25,20])

    # Charts  FIX 9: renamed loop variable title -> chart_title to avoid shadowing
    for chart_title, fig in figures_dict.items():
        pdf.add_section(chart_title)
        pdf.add_chart_image(fig, "")

    # Orphans
    pdf.add_section("Structural Issues")
    if orphans:
        pdf.add_paragraph(f"Orphan pages ({len(orphans)}) — no inbound internal links:")
        orphan_rows = [(get_url_path(u),) for u in orphans[:20]]
        pdf.add_table(['Orphan Page URL'], orphan_rows, [190])
    if weakly_linked:
        pdf.add_paragraph(f"Weakly linked pages ({len(weakly_linked)}) — only 1 inbound link:")
        weak_rows = [(get_url_path(u),) for u in weakly_linked[:20]]
        pdf.add_table(['Weakly Linked Page URL'], weak_rows, [190])

    # Recommendations
    pdf.add_section("Recommendations")
    recs = []
    if len(orphans) > 0:
        recs.append(("Fix Orphan Pages", f"Add contextual internal links to the {len(orphans)} orphaned pages."))
    if len(weakly_linked) > 0:
        recs.append(("Strengthen Weak Pages", f"Add 2-3 more internal links to the {len(weakly_linked)} weakly-linked pages."))
    recs += [
        ("Cross-Link Cluster Pages", "When a cluster page mentions a sibling topic, add a contextual link."),
        ("Strengthen Pillar Pages", "Link pillar pages from homepage and persistent navigation."),
        ("Reduce Click Depth", "Ensure key pages are reachable within 3 clicks from the homepage."),
    ]
    for title_r, body_r in recs:
        pdf.add_paragraph(f"- {title_r}: {body_r}")

    return bytes(pdf.output())

# ══════════════════════════════════════════════════════════════════════
# 3-D HC VISUALIZER
# ══════════════════════════════════════════════════════════════════════

def _assign_category_colors(targets: list) -> dict:
    unique = sorted(set(targets))
    return {cat: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, cat in enumerate(unique)}

def _url_to_category(url: str) -> str:
    try:
        path = urlparse(url).path.strip('/')
        if not path:
            return 'homepage'
        seg = path.split('/')[0]
        return seg[:30] if seg else 'homepage'
    except Exception:
        return 'other'

def crawl_df_to_3d(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame()
    d['Domain'] = df['url'].apply(get_url_path)
    d['Full_URL'] = df['url']
    d['Target'] = df['url'].apply(_url_to_category)
    d['HC_Score'] = df['harmonic_centrality'].fillna(0)
    d['DR'] = df['in_degree'].fillna(0)
    d['Page_Traffic'] = df['out_degree'].fillna(0)
    d['Volume'] = (df['betweenness_centrality'].fillna(0) * 1000).round(2)
    d['PageRank'] = df['pagerank'].fillna(0)
    d['Tier'] = df['tier']
    return d.sort_values('HC_Score', ascending=False).reset_index(drop=True)

def _build_bipartite_graph(df: pd.DataFrame):
    G = nx.Graph()
    for _, row in df.iterrows():
        domain = str(row['Domain'])
        target = str(row['Target'])
        hc = float(row.get('HC_Score', 1) or 1)
        dr = float(row.get('DR', 0) or 0)
        traf = float(row.get('Page_Traffic', 0) or 0)
        vol = float(row.get('Volume', 0) or 0)
        if not G.has_node(domain):
            G.add_node(domain, kind='domain', hc=hc, dr=dr, traffic=traf, volume=vol, target=target)
        else:
            G.nodes[domain]['hc'] += hc
        if not G.has_node(target):
            G.add_node(target, kind='category', hc=0, dr=0, traffic=0, volume=0, target=target)
        G.add_edge(domain, target, weight=hc)
    return G

def _spring_layout_3d(G: nx.Graph, seed: int = 42) -> dict:
    pos2d = nx.spring_layout(G, seed=seed, weight='weight', k=1.2)
    dc = nx.degree_centrality(G)
    return {node: (float(x), float(y), float(dc.get(node, 0) * 4))
            for node, (x, y) in pos2d.items()}

def create_3d_network(df: pd.DataFrame) -> go.Figure:
    G = _build_bipartite_graph(df)
    pos = _spring_layout_3d(G)
    cat_colors = _assign_category_colors(list(df['Target'].unique()))
    ex, ey, ez = [], [], []
    for u, v in G.edges():
        x0,y0,z0 = pos[u]; x1,y1,z1 = pos[v]
        ex += [x0,x1,None]; ey += [y0,y1,None]; ez += [z0,z1,None]
    edge_trace = go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(color='rgba(32,128,141,0.18)', width=1),
        hoverinfo='none', showlegend=False,
    )
    domains = [n for n, d in G.nodes(data=True) if d.get('kind') == 'domain']
    max_hc = max((G.nodes[n]['hc'] for n in domains), default=1)
    dx, dy, dz, dsize, dcolor, dtext = [], [], [], [], [], []
    for n in domains:
        nd = G.nodes[n]
        x, y, z = pos[n]
        hc = nd['hc']
        dx.append(x); dy.append(y); dz.append(z)
        dsize.append(max(5, min(40, (hc/max_hc)*38+4)))
        dcolor.append(cat_colors.get(nd['target'], '#20808D'))
        dtext.append(
            f"<b>{n}</b><br>HC: {hc:.4f}<br>"
            f"In-Links (DR): {nd['dr']:.0f}<br>"
            f"Out-Links: {nd['traffic']:.0f}<br>"
            f"Cluster: {nd['target']}"
        )
    domain_trace = go.Scatter3d(
        x=dx, y=dy, z=dz, mode='markers',
        marker=dict(size=dsize, color=dcolor, opacity=0.88, line=dict(width=0.5, color='white')),
        text=dtext, hoverinfo='text', name='Pages',
    )
    categories = [n for n, d in G.nodes(data=True) if d.get('kind') == 'category']
    cx, cy, cz, ccolor, ctext = [], [], [], [], []
    for n in categories:
        x, y, z = pos[n]
        cx.append(x); cy.append(y); cz.append(z)
        ccolor.append(cat_colors.get(n, '#1B474D'))
        ctext.append(f"<b>Cluster: {n}</b><br>Pages: {G.degree(n)}")
    cat_trace = go.Scatter3d(
        x=cx, y=cy, z=cz, mode='markers+text',
        marker=dict(size=18, color=ccolor, symbol='diamond', opacity=0.95, line=dict(width=1, color='white')),
        text=[n for n in categories], textposition='top center',
        textfont=dict(size=9, color='white'),
        hovertext=ctext, hoverinfo='text', name='Clusters',
    )
    fig = go.Figure(data=[edge_trace, domain_trace, cat_trace])
    fig.update_layout(
        title=dict(text="3D HC Network — Pages clustered by URL segment", font=dict(size=16, color='#CDCCCA')),
        scene=dict(
            bgcolor='#0E1117',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=True, zeroline=False, showticklabels=False, title='Centrality Depth'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        paper_bgcolor='#0E1117', font=dict(family='Inter', color='#CDCCCA'),
        margin=dict(l=0, r=0, t=50, b=0), height=700,
    )
    return fig

def create_3d_hc_scatter(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2['log_traffic'] = np.log1p(df2['Page_Traffic'].fillna(0))
    df2['log_volume'] = np.log1p(df2['Volume'].fillna(0))
    df2['DR'] = df2['DR'].fillna(0)
    df2['HC_Score'] = df2['HC_Score'].fillna(0)
    max_hc = df2['HC_Score'].max() or 1
    sizes = (df2['HC_Score']/max_hc*30+4).clip(4, 36).tolist()
    hover = [
        f"<b>{row['Domain']}</b><br>"
        f"HC Score: {row['HC_Score']:.4f}<br>"
        f"In-Links (DR): {row['DR']:.0f}<br>"
        f"Out-Links: {row.get('Page_Traffic',0):.0f}<br>"
        f"Betweenness×1k: {row.get('Volume',0):.2f}<br>"
        f"Cluster: {row['Target']}"
        for _, row in df2.iterrows()
    ]
    fig = go.Figure(data=[go.Scatter3d(
        x=df2['DR'], y=df2['log_traffic'], z=df2['log_volume'],
        mode='markers',
        marker=dict(
            size=sizes, color=df2['HC_Score'],
            colorscale=[[0,'#BCE2E7'],[0.4,'#20808D'],[0.7,'#1B474D'],[1.0,'#FFC553']],
            opacity=0.85, line=dict(width=0.4, color='white'),
            colorbar=dict(title='HC Score'),
        ),
        text=hover, hoverinfo='text',
    )])
    fig.update_layout(
        title=dict(text="3D Quality Scatter — In-Links × Out-Links × Betweenness",
                   font=dict(size=16, color='#CDCCCA')),
        scene=dict(
            bgcolor='#0E1117',
            xaxis=dict(title='In-Links (DR proxy)'),
            yaxis=dict(title='log(Out-Links)'),
            zaxis=dict(title='log(Betweenness×1k)'),
            camera=dict(eye=dict(x=1.6, y=1.4, z=0.9)),
        ),
        paper_bgcolor='#0E1117', font=dict(family='Inter', color='#CDCCCA'),
        margin=dict(l=0, r=0, t=50, b=0), height=650,
    )
    return fig

def create_3d_hc_skyline(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy().sort_values('HC_Score', ascending=False)
    cat_colors = _assign_category_colors(list(df2['Target'].unique()))
    categories = df2['Target'].unique().tolist()
    cat_idx = {c: i for i, c in enumerate(categories)}
    traces = []
    for cat in categories:
        sub = df2[df2['Target']==cat].head(15).reset_index(drop=True)
        color = cat_colors[cat]
        for rank, row in sub.iterrows():
            traces.append(go.Scatter3d(
                x=[cat_idx[cat], cat_idx[cat]], y=[rank, rank], z=[0, row['HC_Score']],
                mode='lines', line=dict(color=color, width=6),
                hoverinfo='skip', showlegend=False,
            ))
        traces.append(go.Scatter3d(
            x=[cat_idx[cat]] * len(sub), y=list(range(len(sub))),
            z=sub['HC_Score'].tolist(), mode='markers', name=cat,
            marker=dict(size=5, color=color, opacity=0.9),
            text=[f"<b>{r['Domain']}</b><br>HC: {r['HC_Score']:.4f}" for _, r in sub.iterrows()],
            hoverinfo='text',
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text="3D HC Skyline — Top Pages per Cluster", font=dict(size=16, color='#CDCCCA')),
        scene=dict(
            bgcolor='#0E1117',
            xaxis=dict(title='Cluster', tickvals=list(range(len(categories))),
                       ticktext=[c[:18] for c in categories]),
            yaxis=dict(title='Page Rank'),
            zaxis=dict(title='HC Score'),
            camera=dict(eye=dict(x=2.0, y=-1.8, z=1.2)),
        ),
        paper_bgcolor='#0E1117', font=dict(family='Inter', color='#CDCCCA'),
        margin=dict(l=0, r=0, t=50, b=0), height=680,
    )
    return fig

def render_3d_tab(df_3d: pd.DataFrame):
    st.markdown("""  # FIX 10: was garbled with non-tab content mixed in
    <div style="background:#1B474D22;border-left:3px solid #20808D;
                padding:14px 18px;border-radius:6px;margin-bottom:18px">
    <b>3D Harmonic Centrality Network</b> built directly from your crawl.
    Each <b>sphere</b> = a page, sized by HC score.
    <b>Diamond hubs</b> = URL-segment clusters (blog, services, products, etc.).
    Rotate by dragging &bull; Zoom with scroll &bull; Hover for full metrics.
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Pages", f"{len(df_3d)}")
    k2.metric("Clusters", f"{df_3d['Target'].nunique()}")
    k3.metric("Max HC", f"{df_3d['HC_Score'].max():.4f}")
    k4.metric("Avg HC", f"{df_3d['HC_Score'].mean():.4f}")
    k5.metric("Avg In-Links", f"{df_3d['DR'].mean():.1f}")

    with st.expander("Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            min_hc = st.slider("Min HC Score", 0.0, float(df_3d['HC_Score'].max()), 0.0, 0.01, key='3d_min_hc')
        with fc2:
            min_in = st.slider("Min In-Links", 0, int(df_3d['DR'].max()), 0, key='3d_min_in')
        with fc3:
            all_cats = sorted(df_3d['Target'].unique())
            sel_cats = st.multiselect("Clusters", all_cats, default=all_cats, key='3d_cats')

    filt = df_3d[
        (df_3d['HC_Score'] >= min_hc) &
        (df_3d['DR'] >= min_in) &
        (df_3d['Target'].isin(sel_cats))
    ].copy()

    if filt.empty:
        st.warning("No pages match the current filters.")
        return

    st.caption(f"Showing {len(filt)} pages across {filt['Target'].nunique()} clusters")

    t1, t2, t3 = st.tabs(["🌐 Network (Screaming Frog style)", "📊 Quality Scatter", "🏙️ HC Skyline"])
    with t1:
        st.markdown("**How to read:** Sphere size = HC Score · Colour = cluster · Diamond = cluster hub")
        with st.spinner("Rendering 3D network…"):
            st.plotly_chart(create_3d_network(filt), use_container_width=True)
    with t2:
        st.markdown("**Axes:** X = in-links · Y = log(out-links) · Z = log(betweenness) · Colour gradient = HC")
        with st.spinner("Rendering 3D scatter…"):
            st.plotly_chart(create_3d_hc_scatter(filt), use_container_width=True)
    with t3:
        st.markdown("**Bars:** Each vertical bar = one page · Height = HC Score · Grouped by URL cluster")
        with st.spinner("Rendering 3D skyline…"):
            st.plotly_chart(create_3d_hc_skyline(filt), use_container_width=True)

    st.markdown("**All Pages — HC Scored**")
    show_cols = ['Domain','Target','HC_Score','DR','Page_Traffic','Volume','Tier']
    tbl = filt[show_cols].copy().reset_index(drop=True)
    tbl.index += 1
    st.dataframe(
        tbl.style
           .background_gradient(subset=['HC_Score'], cmap='YlGnBu')
           .format({'HC_Score': '{:.4f}', 'DR': '{:.0f}', 'Page_Traffic': '{:.0f}', 'Volume': '{:.2f}'}),
        use_container_width=True, height=440,
    )
    csv_buf = io.StringIO()
    filt.to_csv(csv_buf, index=False)
    st.download_button("Download 3D Data CSV", csv_buf.getvalue(), "hc_3d_data.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════

def main():
    _init_state()

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        url_input = st.text_input(
            "Website URL", placeholder="https://example.com",
            key="url_input_field", help="Enter the full URL including https://",
        )
        max_pages = st.slider("Max pages to crawl", 10, 500, 100, 10, key="max_pages_slider")
        crawl_delay = st.slider("Crawl delay (s)", 0.1, 2.0, 0.3, 0.1, key="crawl_delay_slider")
        network_nodes = st.slider("Network graph nodes", 20, 200, 80, 10, key="network_nodes_slider")
        start_crawl = st.button("🔍 Analyze Website", type="primary", use_container_width=True, key="start_crawl_btn")
        if st.session_state.crawl_done:
            if st.button("🗑️ New Crawl (clear results)", use_container_width=True, key="clear_btn"):
                _clear_state()
                st.rerun()
            st.success(f"✅ Results loaded: {st.session_state.crawl_url[:40]}")
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "Harmonic centrality measures how reachable a page is within an internal link network. "
            "High scores indicate pages that are close to many others — they capture more crawl budget, "
            "link equity, and AI citation probability."
        )

    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px 0">
      <h1 style="color:#28251D;font-weight:700;margin-bottom:0">🔗 Harmonic Centrality Analyzer</h1>
      <p style="color:#7A7974;font-size:1.1rem;margin-top:4px">Advanced Internal Link Architecture Analysis for SEO</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.crawl_done and not start_crawl:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        cards = [
            ("📈", "Graph Metrics", "Harmonic centrality, PageRank, betweenness, closeness, and HITS scores"),
            ("🕸️", "Network Visualization", "Interactive 2D graph, 3D network, treemaps, heatmaps, radar charts"),
            ("📄", "PDF Export", "Download a full report with all findings and recommendations"),
        ]
        for col, (icon, title_c, desc) in zip([c1, c2, c3], cards):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:2rem;margin-bottom:8px">{icon}</div>
                  <h4 style="color:#28251D">{title_c}</h4>
                  <p style="color:#7A7974;font-size:0.9rem">{desc}</p>
                </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box" style="margin-top:24px">
          <h4>How It Works</h4>
          <p>
            <b>1.</b> Enter your website URL and click Analyze.<br>
            <b>2.</b> The crawler discovers internal links via BFS traversal.<br>
            <b>3.</b> A directed graph is built and harmonic centrality is computed: <code>H(v) = &Sigma; 1/d(v,u)</code><br>
            <b>4.</b> Results compare HC vs PageRank, betweenness, HITS &amp; closeness.<br>
            <b>5.</b> The <b>3D HC Network tab</b> shows a Screaming-Frog-style cluster view.<br>
            <b>6.</b> All results persist &mdash; switching tabs never loses your data.
          </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── TRIGGER NEW CRAWL ──────────────────────────────────────────────
    if start_crawl:
        if not url_input:
            st.error("Please enter a website URL.")
            return
        raw_url = url_input.strip()
        if not raw_url.startswith(('http://', 'https://')):
            raw_url = 'https://' + raw_url
        try:
            if not urlparse(raw_url).netloc:
                raise ValueError
        except Exception:
            st.error("Invalid URL. Please enter a valid website address.")
            return
        _clear_state()
        st.markdown('<div class="section-header">Crawling Website…</div>', unsafe_allow_html=True)
        pb = st.progress(0)
        stx = st.empty()
        t0 = time.time()
        edges, visited, page_info = crawl_website(
            raw_url, max_pages=max_pages, delay=crawl_delay,
            progress_bar=pb, status_text=stx,
        )
        elapsed = time.time() - t0
        pb.progress(1.0)
        stx.text(f"Done — {len(visited)} pages in {elapsed:.1f}s")
        if len(visited) < 2:
            st.error("Could not crawl enough pages. Check the URL and try again.")
            return
        G = build_graph(edges, visited)
        G.remove_edges_from(nx.selfloop_edges(G))
        metrics = compute_all_metrics(G)
        hc_scores = metrics['harmonic_centrality']
        max_hc = max(hc_scores.values()) if hc_scores else 0
        records = []
        for node in G.nodes():
            rec = dict(
                url=node, path=get_url_path(node),
                harmonic_centrality=hc_scores.get(node, 0),
                pagerank=metrics['pagerank'].get(node, 0),
                betweenness_centrality=metrics['betweenness_centrality'].get(node, 0),
                closeness_centrality=metrics['closeness_centrality'].get(node, 0),
                hub_score=metrics['hub_score'].get(node, 0),
                authority_score=metrics['authority_score'].get(node, 0),
                in_degree=metrics['in_degree'].get(node, 0),
                out_degree=metrics['out_degree'].get(node, 0),
                tier=classify_score(hc_scores.get(node, 0), max_hc),
            )
            info = page_info.get(node, {})
            rec['title'] = info.get('title', '')
            rec['word_count'] = info.get('word_count', 0)
            rec['status_code'] = info.get('status_code', 0)
            records.append(rec)
        df = pd.DataFrame(records).sort_values('harmonic_centrality', ascending=False).reset_index(drop=True)
        st.session_state.crawl_done = True
        st.session_state.crawl_url = raw_url
        st.session_state.crawl_df = df
        st.session_state.crawl_G = G
        st.session_state.crawl_metrics = metrics
        st.session_state.crawl_page_info = page_info
        st.session_state.crawl_edges = edges
        st.session_state.crawl_visited = visited
        st.session_state.crawl_time = elapsed
        st.rerun()

    if not st.session_state.crawl_done:
        return

    # ── RENDER RESULTS ─────────────────────────────────────────────────
    df = st.session_state.crawl_df
    G = st.session_state.crawl_G
    metrics = st.session_state.crawl_metrics
    page_info = st.session_state.crawl_page_info
    parsed = urlparse(st.session_state.crawl_url)
    crawl_time = st.session_state.crawl_time
    hc_scores = metrics['harmonic_centrality']
    max_hc = df['harmonic_centrality'].max() if len(df) > 0 else 0
    all_nodes = list(G.nodes())
    orphans, weakly_linked = create_orphan_analysis(G, all_nodes)

    st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    avg_hc = df['harmonic_centrality'].mean()
    density = nx.density(G)
    for col, val, label, sub in [
        (k1, len(st.session_state.crawl_visited), "Pages Crawled", f"in {crawl_time:.1f}s"),
        (k2, f"{G.number_of_edges():,}", "Internal Links", f"{G.number_of_edges()/len(all_nodes):.1f} per page"),
        (k3, f"{max_hc:.2f}", "Highest HC", df.iloc[0]['path'][:25] if len(df)>0 else ''),
        (k4, f"{avg_hc:.3f}", "Avg HC Score", f"median {df['harmonic_centrality'].median():.3f}"),
        (k5, len(orphans), "Orphan Pages", f"{len(orphans)/len(all_nodes)*100:.1f}% of total"),
        (k6, f"{density:.4f}", "Graph Density", "Sparse" if density<0.05 else "Moderate" if density<0.2 else "Dense"),
    ]:
        with col:
            color = '#A84B2F' if label == "Orphan Pages" and len(orphans) > 0 else '#20808D'
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:{color}">{val}</div>
              <div class="metric-label">{label}</div>
              <div class="metric-sublabel">{sub}</div>
            </div>""", unsafe_allow_html=True)

    top_page = df.iloc[0] if len(df) > 0 else None
    if top_page is not None:
        high_pct = len(df[df['tier']=='high'])/len(df)*100
        orphan_pct = len(orphans)/len(all_nodes)*100
        st.markdown(f"""
        <div class="insight-box">
          <h4>Key Insight</h4>
          <p><b>{top_page['path']}</b> is your most reachable page (HC = <b>{top_page['harmonic_centrality']:.4f}</b>).
          {high_pct:.0f}% of pages carry high centrality.
          {'⚠️ ' if orphan_pct>10 else ''}<b>{orphan_pct:.0f}% are orphaned</b> —
          {'this is critical and needs immediate attention.' if orphan_pct>10 else 'the orphan rate is acceptable but can be improved.'}
          </p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)
    figures_dict = {}
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🏆 Top Pages", "📊 Distribution", "⚡ HC vs PageRank",
        "🕸️ Network Graph", "🔽 Click Depth", "🔥 Link Profile",
        "🎯 Tier Breakdown", "🕷️ Centrality Radar", "🌐 3D HC Network",
    ])

    with tab1:
        fig_top = create_top_pages_chart(df, n=min(25, len(df)))
        st.plotly_chart(fig_top, use_container_width=True)
        figures_dict['Top Pages by Harmonic Centrality'] = fig_top
        fig_tb = create_top_bottom_comparison(df, n=10)
        st.plotly_chart(fig_tb, use_container_width=True)
        figures_dict['Top vs Bottom Pages'] = fig_tb

    with tab2:
        fig_dist = create_distribution_chart(df)
        st.plotly_chart(fig_dist, use_container_width=True)
        figures_dict['HC Distribution'] = fig_dist
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribution Statistics**")
            st.dataframe(df['harmonic_centrality'].describe().to_frame('HC'), use_container_width=True)
        with col2:
            st.markdown("**Percentiles**")
            pcts = {f"P{int(p*100)}": df['harmonic_centrality'].quantile(p) for p in [0.1,0.25,0.5,0.75,0.9,0.95,0.99]}
            st.dataframe(pd.DataFrame(pcts, index=['value']).T, use_container_width=True)

    with tab3:
        fig_scatter = create_hc_vs_pagerank(df)
        st.plotly_chart(fig_scatter, use_container_width=True)
        figures_dict['HC vs PageRank'] = fig_scatter
        st.markdown("""
        <div class="insight-box">
          <h4>Reading This Chart</h4>
          <p><b>Top-right</b> = highly reachable AND authoritative (your strongest pages).<br>
          <b>High HC, low PR</b> = easy to find but lacks authority — build more links to these.<br>
          <b>Low HC, high PR</b> = receives authority but hard to discover — improve internal linking.</p>
        </div>""", unsafe_allow_html=True)

    with tab4:
        fig_net = create_network_graph(G, hc_scores, max_nodes=network_nodes)
        st.plotly_chart(fig_net, use_container_width=True)
        figures_dict['Network Graph'] = fig_net

    with tab5:
        fig_depth, depth_df = create_depth_analysis(G, normalize_url(st.session_state.crawl_url))
        if fig_depth:
            st.plotly_chart(fig_depth, use_container_width=True)
            figures_dict['Click Depth Distribution'] = fig_depth
            fig_tree = create_crawl_depth_treemap(depth_df, hc_scores)
            if fig_tree:
                st.plotly_chart(fig_tree, use_container_width=True)
                figures_dict['Depth Treemap'] = fig_tree
        else:
            st.warning("Could not compute click depth.")

    with tab6:
        fig_link = create_link_equity_heatmap(df)
        st.plotly_chart(fig_link, use_container_width=True)
        figures_dict['Link Profile'] = fig_link

    with tab7:
        fig_tier = create_tier_breakdown(df)
        st.plotly_chart(fig_tier, use_container_width=True)
        figures_dict['Tier Breakdown'] = fig_tier

    with tab8:
        if len(df) >= 3:
            fig_radar = create_centrality_comparison(df)
            st.plotly_chart(fig_radar, use_container_width=True)
            figures_dict['Centrality Radar'] = fig_radar
        else:
            st.info("Need at least 3 pages for radar comparison.")

    with tab9:
        df_3d = crawl_df_to_3d(df)
        render_3d_tab(df_3d)

    # ── STRUCTURAL ISSUES ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Structural Issues</div>', unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown(f"**Orphan Pages** ({len(orphans)})")
        if orphans:
            st.dataframe(pd.DataFrame({'Page': [get_url_path(u) for u in orphans]}),
                         use_container_width=True, height=300)
        else:
            st.success("No orphan pages found.")
    with ic2:
        st.markdown(f"**Weakly Linked Pages** ({len(weakly_linked)})")
        if weakly_linked:
            st.dataframe(pd.DataFrame({'Page': [get_url_path(u) for u in weakly_linked],
                                        'In-Degree': [1]*len(weakly_linked)}),
                         use_container_width=True, height=300)
        else:
            st.success("No weakly linked pages found.")

    # ── FULL DATA TABLE ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Full Data Table</div>', unsafe_allow_html=True)
    disp = df[['path','title','harmonic_centrality','pagerank',
               'betweenness_centrality','closeness_centrality',
               'in_degree','out_degree','authority_score','hub_score',
               'word_count','tier']].copy()
    disp.columns = ['Path','Title','HC','PageRank','Betweenness','Closeness',
                    'In-Links','Out-Links','Authority','Hub','Words','Tier']
    st.dataframe(
        disp.style.background_gradient(subset=['HC'], cmap='YlGnBu'),
        use_container_width=True, height=500,
    )
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download Full Data CSV", csv_buf.getvalue(),
        f"hc_{parsed.netloc}_{time.strftime('%Y%m%d')}.csv", "text/csv",
    )

    # ── RECOMMENDATIONS ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)
    recs = []
    if orphans:
        recs.append((f"1. Fix {len(orphans)} Orphan Pages",
                     "These pages have zero inbound internal links — they receive no link equity and are rarely crawled. "
                     "Add each orphan to at least one relevant cluster page."))
    if weakly_linked:
        recs.append((f"2. Strengthen {len(weakly_linked)} Weakly Linked Pages",
                     "Single-link dependency means one broken link makes a page an orphan. "
                     "Add 2–3 contextual internal links from related content."))
    recs += [
        ("3. Cross-Link Within Clusters",
         "When a cluster page mentions a topic covered by a sibling page, add a contextual link. "
         "This shortens average path lengths and raises HC across the cluster."),
        ("4. Strengthen Pillar Pages",
         "Link pillar/category pages from the homepage, related clusters, and persistent nav. "
         "Aim for single-hop access from as many pages as possible."),
        ("5. Reduce Click Depth",
         "Every additional hop from the homepage halves crawl probability. "
         "Key money pages should sit at depth ≤ 3."),
    ]
    for title_r, body_r in recs:
        st.markdown(f"""
        <div class="insight-box">
          <h4>{title_r}</h4>
          <p>{body_r}</p>
        </div>""", unsafe_allow_html=True)

    # ── PDF DOWNLOAD ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Download Full Report</div>', unsafe_allow_html=True)
    try:
        pdf_bytes = generate_pdf_report(
            st.session_state.crawl_url, df, G, metrics,
            figures_dict, page_info, orphans, weakly_linked,
        )
        st.download_button(
            "📄 Download PDF Report", pdf_bytes,
            f"hc_report_{parsed.netloc}_{time.strftime('%Y%m%d')}.pdf",
            "application/pdf", type="primary",
        )
    except Exception as e:
        st.warning(f"PDF issue: {str(e)[:100]}. CSV is still available above.")

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#BAB9B4;font-size:0.85rem">'
        'Based on <a href="https://searchministry.au/guides/what-is-harmonic-centrality" target="_blank">'
        '"What Is Harmonic Centrality?"</a> by Tharindu Gunawardana</p>',
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()