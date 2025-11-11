from flask import Flask, render_template, request, jsonify, make_response
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import joblib
import os

app = Flask(__name__, template_folder="templates")
# During development prefer not to cache static files so browser shows updates immediately
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def no_cache_headers(response):
    """Ensure development responses are not cached by the browser.

    This helps when modifying templates/static files frequently during development.
    """
    try:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    except Exception:
        pass
    return response

# --- Daten / Artefakte laden (schneller Start mit Cache) ---
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / 'artifacts.joblib'

# A tiny in-memory recorder to inspect the last request the server received (useful for debugging client issues)
last_request_info = {}
LAST_REQUEST_LOG = BASE_DIR / 'last_requests.log'

def build_artifacts_and_save():
    """Build artifacts by delegating to the separate build script and load the resulting joblib.

    This is a lightweight wrapper so we don't duplicate the heavy preprocessing here.
    It will call `web_app/build_artifacts.py` with --out pointing to ARTIFACT_PATH.
    If a Downloads/archive/movies.csv exists, pass it as --data to prefer the archive dataset.
    """
    import subprocess
    import sys

    script = Path(__file__).resolve().parent / 'build_artifacts.py'
    cmd = [sys.executable, str(script), '--out', str(ARTIFACT_PATH)]
    home_archive = Path.home() / 'Downloads' / 'archive' / 'movies.csv'
    if home_archive.exists():
        cmd = [sys.executable, str(script), '--data', str(home_archive), '--out', str(ARTIFACT_PATH)]

    # Run the builder; let any exceptions surface so the caller sees errors
    subprocess.check_call(cmd)
    # load and return the created artifacts
    return joblib.load(ARTIFACT_PATH)


if ARTIFACT_PATH.exists():
    data = joblib.load(ARTIFACT_PATH)
    features = data['features']
    scaled = data['scaled']
    scaler = data['scaler']
    kmeans = data['kmeans']
    nn = data['nn']
    title_to_pos = data['title_to_pos']
    df = data.get('original_df')
    available_genres = data.get('available_genres', [])
    available_actors = data.get('available_actors', [])
else:
    # Build and save artifacts (this may take a bit the first time)
    data = build_artifacts_and_save()
    features = data['features']
    scaled = data['scaled']
    scaler = data['scaler']
    kmeans = data['kmeans']
    nn = data['nn']
    title_to_pos = data['title_to_pos']
    df = data.get('original_df')
    available_genres = data.get('available_genres', [])
    available_actors = data.get('available_actors', [])


def recommend_movies(title: str, n: int = 6, genre: 'list|str' = None, actor: 'list|str' = None):
    key = title.lower()
    if key not in title_to_pos:
        return None
    pos = title_to_pos[key]
    vec = scaled[pos].reshape(1, -1)
    distances, indices = nn.kneighbors(vec, n_neighbors=10)
    candidates = []
    target_cluster = int(features.loc[pos, 'Cluster'])
    for idx in indices[0]:
        if int(features.loc[idx, 'Cluster']) != target_cluster:
            continue
        if idx == pos:
            continue
        candidates.append(idx)
        if len(candidates) >= n:
            break

    # If not enough candidates are in the same cluster, fill with next nearest neighbors
    if len(candidates) < n:
        for idx in indices[0]:
            if idx == pos or idx in candidates:
                continue
            candidates.append(idx)
            if len(candidates) >= n:
                break

    results = []
    # apply optional filters (genre, actor) when building results
    def row_matches_filters(orig_row, genre_filter, actor_filter):
        # genre_filter may be a list of genres or a single genre string
        if genre_filter:
            g = str(orig_row.get('Genre', '')).lower()
            if isinstance(genre_filter, (list, tuple)):
                # match if any provided genre appears in the movie's genre string
                matches = any(gf.strip().lower() in g for gf in genre_filter if gf)
                if not matches:
                    return False
            else:
                if genre_filter.strip().lower() not in g:
                    return False
        if actor_filter:
            # actor_filter may be list or string; check Star1..Star4
            actor_cols = [c for c in orig_row.index if str(c).lower().startswith('star')]
            qlist = actor_filter if isinstance(actor_filter, (list, tuple)) else [actor_filter]
            found_any = False
            for q in (a for a in qlist if a):
                ql = q.strip().lower()
                for c in actor_cols:
                    val = orig_row.get(c, '')
                    if pd.notna(val) and ql in str(val).strip().lower():
                        found_any = True
                        break
                if found_any:
                    break
            if not found_any:
                return False
        return True

    for c in candidates:
        if len(results) >= n:
            break
        row = features.loc[c]
        orig_idx = int(row['orig_index'])
        orig_row = df.loc[orig_idx]
        if not row_matches_filters(orig_row, genre, actor):
            continue
        results.append({
            'Series_Title': orig_row['Series_Title'],
            'Genre': orig_row['Genre'],
            'IMDB_Rating': orig_row['IMDB_Rating'],
            'Poster_Link': orig_row.get('Poster_Link', ''),
            'Overview': orig_row.get('Overview', ''),
            'Tagline': orig_row.get('Tagline', '')
        })
    # Return only candidates that match the optional filters. If filters are strict and
    # eliminate candidates, fewer than `n` may be returned.
    return results


def list_movies_by_filter(genre: str = None, actor: str = None, limit: int = 500):
    """Return a list of movie dicts that match the given genre and/or actor.

    If both filters are provided, both must match. Results are sorted by IMDB rating
    (descending) then Released_Year (descending). Limit caps the number of returned items.
    """
    if genre is None and actor is None:
        return []
    # work on a copy of the original dataframe
    mdf = df.copy()
    # normalize genre matching: accept a list of genres or a single genre string
    if genre:
        if isinstance(genre, (list, tuple)):
            # build a combined mask where any of the provided genres match
            mask = pd.Series(False, index=mdf.index)
            for g in genre:
                if not g:
                    continue
                gq = str(g).strip().lower()
                mask = mask | mdf['Genre'].fillna('').str.lower().str.contains(gq)
            mdf = mdf[mask]
        else:
            gq = str(genre).strip().lower()
            mdf = mdf[mdf['Genre'].fillna('').str.lower().str.contains(gq)]
    if actor:
        # actor may be a list of names or a single string
        actor_cols = [c for c in mdf.columns if str(c).lower().startswith('star')]
        if actor_cols:
            # build boolean mask across star columns using substring matching
            mask = pd.Series(False, index=mdf.index)
            if isinstance(actor, (list, tuple)):
                for a in actor:
                    if not a:
                        continue
                    aq = str(a).strip().lower()
                    for c in actor_cols:
                        mask = mask | mdf[c].fillna('').astype(str).str.strip().str.lower().str.contains(aq)
            else:
                aq = str(actor).strip().lower()
                for c in actor_cols:
                    mask = mask | mdf[c].fillna('').astype(str).str.strip().str.lower().str.contains(aq)
            mdf = mdf[mask]
        else:
            # no actor columns available; try searching in any text fields as a fallback
            if isinstance(actor, (list, tuple)):
                mask = pd.Series(False, index=mdf.index)
                for a in actor:
                    if not a:
                        continue
                    aq = str(a).strip().lower()
                    mask = mask | mdf.apply(lambda r: aq in str(r.get('Series_Title','')).lower() or aq in str(r.get('Overview','')).lower(), axis=1)
                mdf = mdf[mask]
            else:
                aq = str(actor).strip().lower()
                mdf = mdf[mdf.apply(lambda r: aq in str(r.get('Series_Title','')).lower() or aq in str(r.get('Overview','')).lower(), axis=1)]

    # ensure numeric columns for sorting
    try:
        mdf['IMDB_Rating'] = pd.to_numeric(mdf.get('IMDB_Rating', pd.Series(np.nan)), errors='coerce')
    except Exception:
        mdf['IMDB_Rating'] = pd.Series(np.nan)
    try:
        mdf['Released_Year'] = pd.to_numeric(mdf.get('Released_Year', pd.Series(np.nan)), errors='coerce')
    except Exception:
        mdf['Released_Year'] = pd.Series(np.nan)

    # For actor-only searches we return all matching movies (no sampling) when limit is None.
    # For genre-only searches we return a random sample of up to `limit` (default 20 in caller).
    mdf = mdf.sort_values(by=['IMDB_Rating', 'Released_Year'], ascending=[False, False])

    # Decide which rows to output based on whether actor/genre are provided
    if actor and not genre:
        # actor-only: return all matches if limit is None, otherwise apply numeric limit
        rows = mdf if limit is None else mdf.head(limit)
    elif genre and not actor:
        # genre-only: return a random sample of up to `limit` rows
        sample_n = limit if limit is not None else 20
        if len(mdf) > sample_n:
            rows = mdf.sample(n=sample_n, random_state=42)
        else:
            rows = mdf
    else:
        # both provided or neither: apply the numeric limit if present
        rows = mdf if limit is None else mdf.head(limit)

    out = []
    for _, row in rows.iterrows():
        out.append({
            'Series_Title': row.get('Series_Title', ''),
            'Genre': row.get('Genre', ''),
            'IMDB_Rating': row.get('IMDB_Rating', ''),
            'Poster_Link': row.get('Poster_Link', ''),
            'Overview': row.get('Overview', ''),
            'Tagline': row.get('Tagline', '')
        })
    return out


@app.route('/autocomplete')
def autocomplete():
    """Simple autocomplete endpoint for actors and genres.

    Query params:
      - type: 'actor' or 'genre'
      - q: query string
    Returns JSON list of up to 12 matching strings.
    """
    typ = request.args.get('type', '')
    q = request.args.get('q', '').strip().lower()
    out = []
    if not q:
        return jsonify([])
    if typ == 'actor':
        # case-insensitive prefix match in available_actors
        for a in available_actors:
            if a and a.lower().startswith(q):
                out.append(a)
                if len(out) >= 12:
                    break
    elif typ == 'genre':
        for g in available_genres:
            if g and g.lower().startswith(q):
                out.append(g)
                if len(out) >= 12:
                    break
    return jsonify(out)


@app.route('/debug_count')
def debug_count():
    """Return JSON with counts for given genre/actor for quick debugging.

    Usage: /debug_count?genre=Comedy&actor=Margot+Robbie
    """
    genre = request.args.get('genre')
    actor = request.args.get('actor')
    # count movies by filters
    matches = list_movies_by_filter(genre=genre or None, actor=actor or None, limit=1000000)
    return jsonify({
        'genre': genre,
        'actor': actor,
        'count': len(matches),
        'sample': [m['Series_Title'] for m in matches[:10]]
    })


@app.route('/api/recommend')
def api_recommend():
    """Return JSON recommendations for quick testing.

    Query params:
      - title: movie title (required)
      - genre: optional, comma-separated or multiple params allowed
      - actor: optional, comma-separated or multiple params allowed
      - n: optional int, number of neighbors (default 6)
    """
    title = request.args.get('title', '').strip()
    if not title:
        return jsonify({'error': 'title query param required'}), 400
    # parse genres and actors
    genres = request.args.getlist('genre') or []
    genre_raw = request.args.get('genre')
    if genre_raw and ',' in genre_raw and not genres:
        genres = [g.strip() for g in genre_raw.split(',') if g.strip()]
    actors = request.args.getlist('actor') or []
    actor_raw = request.args.get('actor')
    if actor_raw and ',' in actor_raw and not actors:
        actors = [a.strip() for a in actor_raw.split(',') if a.strip()]
    try:
        n = int(request.args.get('n', 6))
    except Exception:
        n = 6
    recs = recommend_movies(title, n=n, genre=genres or None, actor=actors or None)
    if recs is None:
        return jsonify({'error': 'title not found'}), 404
    return jsonify({'title': title, 'n': n, 'genre': genres, 'actor': actors, 'results': recs})


@app.route('/last_request')
def get_last_request():
    """Return the last request the server recorded (for debugging client issues)."""
    try:
        return jsonify(last_request_info)
    except Exception:
        return jsonify({}), 200


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    error = None
    query = ''
    selected_genre = ''
    selected_actor = ''
    # Support form POST and GET query params so users can test via URL like /?genre=Comedy or /?actor=Margot+Robbie
    if request.method == 'POST' or request.args:
        # request.values merges args and form, prefer form (POST) values when present
        query = request.values.get('title', '').strip()
        # support genre selection; template currently uses a single-select but accept either
        raw_genres = request.values.getlist('genre') or []
        # normalize: remove empty strings and whitespace-only values
        selected_genres = [g.strip() for g in raw_genres if g and g.strip()]
        # preserve a display string for the actor input (comma-separated)
        selected_actor_text = request.values.get('actor', '').strip()
        # parse actors as comma-separated list (allow multiple)
        selected_actors = [a.strip() for a in selected_actor_text.split(',') if a.strip()]
        # record last request for debugging (write a small JSON line to a logfile)
        try:
            last_request_info.clear()
            last_request_info.update({
                'method': request.method,
                'title': query,
                'selected_genres': selected_genres,
                'selected_actors': selected_actors,
                'values': {k: request.values.getlist(k) for k in request.values.keys()}
            })
            # append to log file so we have a persistent trace
            with open(LAST_REQUEST_LOG, 'a', encoding='utf-8') as fh:
                import json
                fh.write(json.dumps(last_request_info, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # If the user provided neither title nor filters, ask for input
        if not query and not (selected_genres or selected_actors):
            error = 'Please enter a movie title or select a.genre/actor.'
        # If no title but a filter was provided, return matching movies:
        elif not query and (selected_genres or selected_actors):
            # actor-only -> return all actor's films; genre-only -> return 20 random films
            if selected_actors and not selected_genres:
                recommendations = list_movies_by_filter(actor=selected_actors or None, genre=None, limit=None)
            elif selected_genres and not selected_actors:
                recommendations = list_movies_by_filter(genre=selected_genres or None, actor=None, limit=20)
            else:
                # both provided: return actor's movies filtered by genre (no cap)
                recommendations = list_movies_by_filter(genre=selected_genres or None, actor=selected_actors or None, limit=None)
            if not recommendations:
                error = 'Keine Filme gefunden für die gewählten Filter.'
        else:
            # pass filter params to recommendation
            recs = recommend_movies(query, n=6, genre=selected_genres or None, actor=selected_actors or None)
            if recs is None:
                # if the title wasn't found but filters were provided, fall back to listing by filter
                if selected_genres or selected_actors:
                    if selected_actors and not selected_genres:
                        recommendations = list_movies_by_filter(actor=selected_actors or None, genre=None, limit=None)
                    elif selected_genres and not selected_actors:
                        recommendations = list_movies_by_filter(genre=selected_genres or None, actor=None, limit=20)
                    else:
                        recommendations = list_movies_by_filter(genre=selected_genres or None, actor=selected_actors or None, limit=None)
                    if not recommendations:
                        error = 'Movie not found. Keine Filme für die angegebenen Filter.'
                else:
                    error = 'Movie not found. Please check the spelling.'
            else:
                recommendations = recs
    # Compute a simple cache-busting version based on artifact mtime (or current time)
    # Use a time-based cache-busting token so browsers fetch fresh assets on each request.
    # This forces the static asset URLs to change per-request during development.
    static_version = int(time.time())

    rendered = render_template(
        'index.html',
        recommendations=recommendations,
        error=error,
        query=query,
        available_genres=available_genres,
        available_actors=available_actors,
        selected_genres=selected_genres if 'selected_genres' in locals() else [],
        selected_actor_text=selected_actor_text if 'selected_actor_text' in locals() else '',
    static_version=static_version
    )
    # Ensure browsers do not aggressively cache the HTML page
    resp = make_response(rendered)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


if __name__ == '__main__':
    # Run on port 5001 to avoid macOS system services that bind 5000.
    # Disable debug/reloader for a stable single-process dev run.
    app.run(debug=False, host='127.0.0.1', port=5001)
