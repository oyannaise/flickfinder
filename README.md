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

Notes
- The app serves static files from `web_app/static/`. If you change templates or CSS, hard-refresh your browser (Cmd+Shift+R) or open a private window to avoid cached assets.
- The logo file currently used is `web_app/static/title_remote.png` (a local copy of the image). If you prefer a different logo, replace that file.
- For production use, run this behind a proper WSGI server (gunicorn/uvicorn) and avoid using the built-in Flask dev server.

Ready to push
- I can prepare a git commit with these changes and a suggested .gitignore (if you want to exclude large data files) — tell me if you want me to create the commit commands.
