# Movie Recommender

This is a small Flask web app that returns similar movies using pre-built artifacts.
It's intended for local development and demos.

Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Rebuild artifacts if you changed the source CSVs:

```bash
python web_app/build_artifacts.py
```

3. Start the app:

```bash
python web_app/app.py
```

4. Open http://127.0.0.1:5001/ in your browser.

Files that matter
- `web_app/app.py` — Flask app and endpoints
- `web_app/artifacts.joblib` — precomputed artifacts (features, neighbors, title map, dataframe)
- `web_app/build_artifacts.py` — script to regenerate `artifacts.joblib` from CSVs
- `web_app/templates/` — Jinja2 templates for the UI
- `web_app/static/` — CSS, images and other static assets
- `requirements.txt` — Python dependencies
