"""Precompute and save preprocessing artifacts for faster app startup.

This script mirrors the preprocessing in `app.py` and stores a joblib file
with: features (DataFrame), scaled (ndarray), scaler, kmeans, nn, title_to_pos.

Run once (or when the CSV changes) to speed up Flask startup.
"""
from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent


def resolve_csv_path(arg_path: str = None) -> Path:
    """Resolve which CSV to use.

    Priority: --data CLI arg > DATA_CSV env var > project root imdb_top_1000.csv > cwd imdb_top_1000.csv
    """
    candidates = []
    if arg_path:
        candidates.append(Path(arg_path))
    env_path = os.environ.get('DATA_CSV')
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(BASE_DIR / 'imdb_top_1000.csv')
    candidates.append(Path.cwd() / 'imdb_top_1000.csv')

    for p in candidates:
        if p and p.exists():
            return p

    raise FileNotFoundError(
        f"Kein Datensatz gefunden. Versuchte Pfade: {', '.join(str(p) for p in candidates)}"
    )


def build_and_save(out_path: Path, csv_path: Path):
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Detect schema: older imdb_top_1000.csv contains 'Series_Title' and 'Genre'
    if 'Series_Title' in df.columns and 'Genre' in df.columns:
        features = df[['Genre', 'Series_Title', 'IMDB_Rating', 'Released_Year', 'Runtime', 'Meta_score']].copy()
    else:
        # Attempt to adapt archive schema (movies.csv with id,name,date,minute,rating,description)
        print('Detected alternate schema; attempting to merge genres and actors from archive folder')
        movies = df
        # map fields
        movies = movies.rename(columns={
            'name': 'Series_Title',
            'date': 'Released_Year',
            'minute': 'Runtime',
            'rating': 'IMDB_Rating',
            'description': 'Overview',
            'tagline': 'Tagline',
            'id': 'id'
        })

        # build Genre by joining genres.csv if available (same parent folder as csv_path)
        archive_dir = Path(csv_path).parent
        genres_path = archive_dir / 'genres.csv'
        if genres_path.exists():
            # read only id and genre columns to avoid loading huge files into memory
            try:
                hdr = pd.read_csv(genres_path, nrows=0)
                cols = hdr.columns.tolist()
                usecols = [c for c in ('id', 'genre') if c in cols]
                if not usecols:
                    # fallback to first two columns
                    usecols = cols[:2]
                gdf = pd.read_csv(genres_path, usecols=usecols)
                # ensure we have 'id' and 'genre' named columns
                if 'id' not in gdf.columns and gdf.columns.size >= 1:
                    gdf = gdf.rename(columns={gdf.columns[0]: 'id'})
                if 'genre' not in gdf.columns and gdf.columns.size >= 2:
                    gdf = gdf.rename(columns={gdf.columns[1]: 'genre'})
                gg = gdf.groupby('id')['genre'].apply(lambda s: ', '.join(sorted(set(s.dropna().astype(str)))))
                movies = movies.merge(gg.rename('Genre'), how='left', left_on='id', right_index=True)
            except Exception:
                movies['Genre'] = ''
        else:
            movies['Genre'] = ''

        # build actor columns Star1..Star4 from actors.csv if available
        actors_path = archive_dir / 'actors.csv'
        if actors_path.exists():
            try:
                hdr = pd.read_csv(actors_path, nrows=0)
                cols = hdr.columns.tolist()
                usecols = [c for c in ('id', 'name') if c in cols]
                if not usecols:
                    usecols = cols[:2]
                adf = pd.read_csv(actors_path, usecols=usecols)
                # normalize column names
                if 'id' not in adf.columns and adf.columns.size >= 1:
                    adf = adf.rename(columns={adf.columns[0]: 'id'})
                if 'name' not in adf.columns and adf.columns.size >= 2:
                    adf = adf.rename(columns={adf.columns[1]: 'name'})
                # adf: id,name,role -> group names
                actor_lists = adf.groupby('id')['name'].apply(lambda s: list(s.dropna().astype(str)))
            except Exception:
                actor_lists = pd.Series(dtype=object)
            # convert list-of-lists into a rectangular DataFrame, missing entries become NaN
            stars_df = pd.DataFrame(actor_lists.tolist(), index=actor_lists.index)
            # keep only first 4 actors and rename cols
            stars_df = stars_df.loc[:, :3]
            rename_map = {i: f'Star{i+1}' for i in range(4)}
            stars_df = stars_df.rename(columns=rename_map)
            # merge into movies on id (actors index are ids)
            movies = movies.merge(stars_df, how='left', left_on='id', right_index=True)
            # fill missing star columns with empty string
            for i in range(1,5):
                col = f'Star{i}'
                if col not in movies.columns:
                    movies[col] = ''
                movies[col] = movies[col].fillna('')
        else:
            for i in range(1,5):
                movies[f'Star{i}'] = ''

        # attempt to merge poster links from posters.csv (if present)
        posters_path = archive_dir / 'posters.csv'
        if posters_path.exists():
            try:
                hdr = pd.read_csv(posters_path, nrows=0)
                cols = hdr.columns.tolist()
                # try to pick id + one url-like column
                usecols = [c for c in cols if c.lower() in ('id', 'movie_id', 'movieid')]
                # add any poster/url-like column
                url_col = next((c for c in cols if any(k in c.lower() for k in ('poster', 'url', 'image', 'path', 'link'))), None)
                if url_col:
                    usecols.append(url_col)
                if not usecols:
                    usecols = cols[:2]
                pdf = pd.read_csv(posters_path, usecols=usecols)
                # try to detect id column (common names) and a poster/url column
                id_col = None
                for cand in ('id', 'movie_id', 'movieId', 'movieid', 'film_id'):
                    if cand in pdf.columns:
                        id_col = cand
                        break
                if id_col is None:
                    id_col = pdf.columns[0]

                if 'url_col' not in locals():
                    url_col = next((c for c in pdf.columns if any(k in c.lower() for k in ('poster', 'url', 'image', 'path', 'link')) and c != id_col), None)

                if url_col:
                    psub = pdf[[id_col, url_col]].drop_duplicates(subset=id_col).set_index(id_col)
                    psub = psub.rename(columns={url_col: 'Poster_Link'})
                    movies = movies.merge(psub['Poster_Link'], how='left', left_on='id', right_index=True)
                else:
                    movies['Poster_Link'] = ''
            except Exception:
                # if anything goes wrong reading a large posters file, continue without posters
                movies['Poster_Link'] = ''
        else:
            movies['Poster_Link'] = ''

        # ensure Meta_score exists (not available in archive) -> fill with 0
        if 'Meta_score' not in movies.columns:
            movies['Meta_score'] = 0

        # create features frame expected by rest of pipeline
        features = movies[['Genre', 'Series_Title', 'IMDB_Rating', 'Released_Year', 'Runtime', 'Meta_score']].copy()
        # keep original df as movies (with id)
        df = movies
    # Normalize runtime text like '142 min' -> numeric minutes
    if features['Runtime'].dtype == object:
        features['Runtime'] = features['Runtime'].astype(str).str.replace(' min', '', regex=False)
    features['Runtime'] = pd.to_numeric(features['Runtime'], errors='coerce')
    features['IMDB_Rating'] = pd.to_numeric(features['IMDB_Rating'], errors='coerce')
    features['Released_Year'] = pd.to_numeric(features['Released_Year'], errors='coerce')
    features['Meta_score'] = pd.to_numeric(features['Meta_score'], errors='coerce')
    # fill missing meta scores with 0 to avoid dropping too many rows
    features['Meta_score'] = features['Meta_score'].fillna(0)
    # drop rows missing critical numeric info
    features = features.dropna(subset=['IMDB_Rating', 'Released_Year', 'Runtime'])
    genres = features['Genre'].str.get_dummies(sep=', ')
    features = pd.concat([features.drop('Genre', axis=1), genres], axis=1)
    features = features.reset_index().rename(columns={'index': 'orig_index'})

    numeric_cols = features.select_dtypes(include=['number']).columns.tolist()
    if 'orig_index' in numeric_cols:
        numeric_cols.remove('orig_index')
    numeric_features = features[numeric_cols]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_features)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    features['Cluster'] = clusters

    nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
    nn.fit(scaled)

    title_to_pos = {t.lower(): pos for pos, t in enumerate(features['Series_Title'].astype(str).str.lower())}
    # derive available genres (unique single genres) and actors from the original dataframe
    # Genre column contains comma-separated genres; split and get unique
    genre_set = set()
    for g in df['Genre'].dropna().astype(str):
        for part in [p.strip() for p in g.split(',')]:
            if part:
                genre_set.add(part)

    # actors are in Star1..Star4 columns in the provided CSV (if present)
    actor_cols = [c for c in df.columns if c.lower().startswith('star')]
    actor_set = set()
    for c in actor_cols:
        actor_set.update(df[c].dropna().astype(str).str.strip().tolist())

    payload = {
        'features': features,
        'scaled': scaled,
        'scaler': scaler,
        'kmeans': kmeans,
        'nn': nn,
        'title_to_pos': title_to_pos,
        'original_df': df,
        'available_genres': sorted(list(genre_set)),
        'available_actors': sorted([a for a in actor_set if a and a.lower() != 'nan'])
    }
    joblib.dump(payload, out_path)
    print(f"Artifacts saved to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build precomputed artifacts for the recommender')
    parser.add_argument('--data', '-d', help='Path to dataset CSV to use (overrides default)')
    parser.add_argument('--out', '-o', help='Output joblib path', default=str(Path(__file__).resolve().parent / 'artifacts.joblib'))
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.data)
    out = Path(args.out)
    build_and_save(out, csv_path)
