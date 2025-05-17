import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# --- Global Configurations and Resource Loading ---
BASE_PATH = '/Users/chenwang/Documents/flask-song-recommender/'
MASTER_SONG_DATA_PATH = os.path.join(BASE_PATH, 'song_data_new.csv')
SCALER_FILE_PATH   = os.path.join(BASE_PATH, 'scaler_autoencoder.pkl')
MODEL_FILE_PATH    = os.path.join(BASE_PATH, 'autoencoder_trained.pth')

_sp = Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

FEATURE_COLUMNS = [
    'year', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature'
]
AUTOENCODER_INPUT_DIM    = len(FEATURE_COLUMNS)
AUTOENCODER_ENCODING_DIM = 5


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=AUTOENCODER_ENCODING_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


# Load resources
master_df_songs = pd.read_csv(MASTER_SONG_DATA_PATH).drop(columns=['Unnamed: 0'], errors='ignore')
scaler = joblib.load(SCALER_FILE_PATH)
autoencoder_model = Autoencoder(input_dim=AUTOENCODER_INPUT_DIM)
autoencoder_model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location='cpu'))
autoencoder_model.eval()
RESOURCES_LOADED_SUCCESSFULLY = True


def fetch_playlist_df(playlist_url: str) -> pd.DataFrame:
    """
    Fetch Spotify playlist tracks (name, artist, id),
    then match against master_df_songs on song_title & artist_name.
    Returns DataFrame with columns: ['id', 'name', 'artist'] + FEATURE_COLUMNS.
    """
    pid = playlist_url.rstrip('/').split('/')[-1].split('?')[0]
    results = _sp.playlist_tracks(pid, fields="items(track(name,artists(name),id)),next")
    items = results.get('items', [])
    spotify_info = []
    while items:
        for it in items:
            t = it['track']
            spotify_info.append({
                'name_spotify': t['name'],
                'artist_spotify': t['artists'][0]['name'] if t['artists'] else None,
                'id_spotify': t.get('id')
            })
        if results.get('next'):
            results = _sp.next(results)
            items = results.get('items', [])
        else:
            break

    if not spotify_info:
        return pd.DataFrame()

    # Prepare master DataFrame for case-insensitive merge
    df_master = master_df_songs.copy()
    df_master['song_title_lc']  = df_master['song_title'].str.lower()
    df_master['artist_name_lc'] = df_master['artist_name'].str.lower()

    df_sp = pd.DataFrame(spotify_info)
    df_sp['song_title_lc']  = df_sp['name_spotify'].str.lower()
    df_sp['artist_name_lc'] = df_sp['artist_spotify'].str.lower()

    # Merge on lowercase keys
    df_merged = pd.merge(
        df_master,
        df_sp[['song_title_lc', 'artist_name_lc', 'id_spotify']],
        on=['song_title_lc', 'artist_name_lc'],
        how='inner'
    )

    if df_merged.empty:
        return pd.DataFrame()

    # Rename and select columns
    df_merged = df_merged.rename(columns={
        'song_title': 'name',
        'artist_name': 'artist',
        'id_spotify': 'id'
    })
    cols = ['id', 'name', 'artist'] + FEATURE_COLUMNS
    existing_cols = [c for c in cols if c in df_merged.columns]
    return df_merged[existing_cols].reset_index(drop=True)


def get_song_embeddings(df_subset: pd.DataFrame) -> np.ndarray:
    if df_subset.empty:
        return np.array([])
    feats = df_subset[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)
    scaled = scaler.transform(feats)
    with torch.no_grad():
        emb, _ = autoencoder_model(torch.from_numpy(scaled))
    return emb.numpy()


def recommend_songs(df_seed: pd.DataFrame, emotion: str, n_recs: int = 10) -> pd.DataFrame:
    """
    Recommend songs of the same emotion, excluding seed IDs.
    Returns DataFrame with ['name', 'artist', 'id'].
    """
    if not RESOURCES_LOADED_SUCCESSFULLY:
        print("ERROR: Resources not loaded.")
        return pd.DataFrame()
    if df_seed is None or df_seed.empty:
        print("WARNING: Seed playlist is empty.")
        return pd.DataFrame()

    # Filter candidates by emotion
    candidates = master_df_songs[master_df_songs['emotion'].str.lower() == emotion.lower()].copy()
    if candidates.empty:
        print(f"INFO: No songs for emotion '{emotion}'.")
        return pd.DataFrame()

    # Compute embeddings
    seed_emb = get_song_embeddings(df_seed)
    if seed_emb.size == 0:
        print("ERROR: No embeddings for seed.")
        return pd.DataFrame()
    query_emb = seed_emb.mean(axis=0, keepdims=True)

    cand_emb = get_song_embeddings(candidates)
    if cand_emb.size == 0:
        print("ERROR: No embeddings for candidates.")
        return pd.DataFrame()

    # Similarity
    sims = cosine_similarity(cand_emb, query_emb).flatten()
    candidates['similarity'] = sims

    # Determine seed IDs
    if 'id' in df_seed.columns and isinstance(df_seed['id'], pd.Series):
        seed_ids = df_seed['id'].dropna().unique().tolist()
    else:
        seed_ids = []

    # Exclude seeds
    if seed_ids:
        pool = candidates[~candidates['id'].isin(seed_ids)]
    else:
        pool = candidates

    # Top k
    topk = pool.sort_values('similarity', ascending=False).head(n_recs)

    # Build output
    out = topk.rename(columns={'song_title': 'name', 'artist_name': 'artist'})
    cols = [c for c in ['name', 'artist', 'id'] if c in out.columns]
    result = out[cols].reset_index(drop=True)
    print(f"Returning {len(result)} recommendations.")
    return result
