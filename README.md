# Harmonic Centrality Analyzer

**LinkScience by WLDM** — internal link graph SEO tool.

Crawls a website, builds the internal link graph, computes harmonic centrality and other graph metrics, and provides advanced visualizations + PDF export.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Deployed on Railway. Push to `main` triggers auto-redeploy.

## Credits

Originally based on [TAHASHAH12/Harmonic_centrality_tool](https://github.com/TAHASHAH12/Harmonic_centrality_tool). Brand-styled and extended for LinkScience by WLDM.
