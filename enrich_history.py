"""
enrich_history.py  —  One-time history enrichment script
=========================================================
Run from Terminal:
    python3 enrich_history.py \
        --csv  /path/to/scrobbles.csv \
        --profile  marlonrando

What it does (four phases):
  1. Load CSV → unique (artist, track) pairs with play counts, MBIDs, albums
  2. Search Spotify for track IDs → batch-fetch Audio Features
  3. MusicBrainz release year  (uses track_mbid from CSV where present)
  4. Last.fm tag fetch          (reuses shared disk cache)

Output:
  ~/.music-agent/{profile}_track_metadata.json   — enriched track store
  ~/.music-agent/{profile}_clusters.json         — k-means assignments
  ~/.music-agent/{profile}_cluster_top30.txt     — top-30 per cluster
                                                   (ready for AllMusic lookup)
Spotify auth uses Client Credentials — no browser/login required.
Reads SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET from either:
  - .streamlit/secrets.toml  (standard project location)
  - Environment variables
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import re
import sys
import time
import threading
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── 22-lane taxonomy ──────────────────────────────────────────────────────────
LANES: list[dict] = [
    {"name": "Melancholy Balladry",          "energy": (0.0, 0.40), "valence": (0.0, 0.35),  "acousticness": (0.50, 1.0), "danceability": (0.0, 0.50), "mode": 0,    "tempo": (40, 90)},
    {"name": "Introspective Songcraft",      "energy": (0.15,0.55), "valence": (0.20,0.60),  "acousticness": (0.35, 1.0), "danceability": (0.20,0.60), "mode": None,  "tempo": (60,130)},
    {"name": "Hook-Perfected Songs",         "energy": (0.30,0.85), "valence": (0.45,1.0),   "acousticness": (0.0,  0.80),"danceability": (0.40,0.80), "mode": None,  "tempo": (80,180)},
    {"name": "Atmospheric / Texture-First",  "energy": (0.0, 0.40), "valence": (0.0, 0.50),  "acousticness": (0.0,  0.80),"danceability": (0.0, 0.40), "mode": None,  "tempo": (40,100)},
    {"name": "Dark Textured Rock / Pop",     "energy": (0.30,0.65), "valence": (0.0, 0.40),  "acousticness": (0.0,  0.35),"danceability": (0.30,0.60), "mode": 0,    "tempo": (80,145)},
    {"name": "Roots and Traditional",        "energy": (0.15,0.70), "valence": (0.20,0.85),  "acousticness": (0.55, 1.0), "danceability": (0.30,0.75), "mode": None,  "tempo": (60,185)},
    {"name": "Propulsive Guitar Rock",       "energy": (0.55,0.90), "valence": (0.25,0.75),  "acousticness": (0.0,  0.30),"danceability": (0.35,0.70), "mode": None,  "tempo": (110,185)},
    {"name": "Groove-First Rock",            "energy": (0.55,0.90), "valence": (0.50,0.90),  "acousticness": (0.0,  0.30),"danceability": (0.60,0.90), "mode": 1,    "tempo": (90,150)},
    {"name": "Energy-First Rock",            "energy": (0.65,0.95), "valence": (0.20,1.0),   "acousticness": (0.0,  0.35),"danceability": (0.30,0.65), "mode": None,  "tempo": (100,175)},
    {"name": "Adrenaline Rock",              "energy": (0.82,1.0),  "valence": (0.0, 0.85),  "acousticness": (0.0,  0.25),"danceability": (0.20,0.55), "mode": None,  "tempo": (130,220)},
    {"name": "Melody-Anchored Hip-Hop",      "energy": (0.30,0.70), "valence": (0.30,0.70),  "acousticness": (0.0,  0.35),"danceability": (0.55,0.80), "mode": None,  "tempo": (70,110)},
    {"name": "Groove-First Hip-Hop",         "energy": (0.55,0.85), "valence": (0.40,0.80),  "acousticness": (0.0,  0.25),"danceability": (0.70,0.95), "mode": None,  "tempo": (80,115)},
    {"name": "Textural / Atmospheric Hip-Hop","energy":(0.15,0.55), "valence": (0.10,0.50),  "acousticness": (0.0,  0.30),"danceability": (0.45,0.75), "mode": None,  "tempo": (60,100)},
    {"name": "Melody-Led Jazz",              "energy": (0.20,0.65), "valence": (0.40,0.85),  "acousticness": (0.30, 0.90),"danceability": (0.30,0.65), "mode": None,  "tempo": (80,200)},
    {"name": "Harmonic-First Jazz",          "energy": (0.20,0.70), "valence": (0.25,0.70),  "acousticness": (0.20, 0.80),"danceability": (0.25,0.60), "mode": None,  "tempo": (80,220)},
    {"name": "Expansive Jazz",               "energy": (0.10,0.60), "valence": (0.10,0.60),  "acousticness": (0.10, 0.70),"danceability": (0.10,0.45), "mode": None,  "tempo": (40,180)},
    {"name": "Rhythm-Forward Jazz",          "energy": (0.45,0.85), "valence": (0.45,0.85),  "acousticness": (0.10, 0.60),"danceability": (0.65,0.90), "mode": None,  "tempo": (90,180)},
    {"name": "Ambient Jazz",                 "energy": (0.0, 0.35), "valence": (0.20,0.65),  "acousticness": (0.50, 1.0), "danceability": (0.10,0.45), "mode": None,  "tempo": (40,90)},
    {"name": "Modern Pop",                   "energy": (0.55,0.90), "valence": (0.50,0.95),  "acousticness": (0.0,  0.40),"danceability": (0.60,0.90), "mode": None,  "tempo": (100,160)},
    {"name": "Urban / Contemporary R&B",     "energy": (0.40,0.80), "valence": (0.30,0.75),  "acousticness": (0.0,  0.30),"danceability": (0.65,0.90), "mode": None,  "tempo": (70,130)},
    {"name": "Dance / Electronic",           "energy": (0.65,0.98), "valence": (0.40,0.90),  "acousticness": (0.0,  0.15),"danceability": (0.75,0.98), "mode": None,  "tempo": (120,175)},
    {"name": "Soul / Classic R&B",           "energy": (0.40,0.80), "valence": (0.50,0.90),  "acousticness": (0.10, 0.50),"danceability": (0.55,0.85), "mode": None,  "tempo": (80,150)},
]

FEATURE_COLS = ["energy", "valence", "acousticness", "danceability",
                "instrumentalness", "tempo_norm", "mode"]

# ── Paths ─────────────────────────────────────────────────────────────────────
MUSIC_AGENT_DIR = Path.home() / ".music-agent"
LASTFM_CACHE    = MUSIC_AGENT_DIR / "lastfm_tag_cache.json"
SPOTIFY_CACHE   = MUSIC_AGENT_DIR / "spotify_id_cache.json"   # artist+track → spotify_id

# ── Spotify client credentials ────────────────────────────────────────────────

def _load_secrets() -> dict:
    """Read from .streamlit/secrets.toml in parent directories, or env."""
    import os
    secrets: dict = {}
    for candidate in [
        Path(__file__).parent.parent / ".streamlit" / "secrets.toml",
        Path(__file__).parent       / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit"  / "secrets.toml",
    ]:
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    secrets[k.strip()] = v.strip().strip('"').strip("'")
            break
    # env overrides
    for key in ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET",
                "LASTFM_API_KEY", "ANTHROPIC_API_KEY"):
        if key in os.environ:
            secrets[key] = os.environ[key]
    return secrets


def _spotify_client_token(client_id: str, client_secret: str) -> str:
    """Exchange client credentials for a bearer token (valid ~1 hour)."""
    creds   = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    payload = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=payload,
        headers={"Authorization": f"Basic {creds}",
                 "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())["access_token"]


# ── Generic HTTP helpers ──────────────────────────────────────────────────────

def _get_json(url: str, headers: dict | None = None,
              timeout: int = 10, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
    return None


# ── Phase 1: Load CSV ─────────────────────────────────────────────────────────

def load_history(csv_path: Path) -> dict[tuple, dict]:
    """
    Returns {(artist, track): {plays, albums: set, track_mbid, album_mbid}}
    Handles both Last.fm export format (uts,utc_time,artist,...) and
    Spotify streaming history format.
    """
    tracks: dict[tuple, dict] = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        is_lastfm  = "artist" in headers and "track" in headers and "uts" in headers
        is_spotify = "artistName" in headers or "master_metadata_track_name" in headers

        for row in reader:
            if is_lastfm:
                artist = row.get("artist", "").strip()
                track  = row.get("track",  "").strip()
                album  = row.get("album",  "").strip()
                t_mbid = row.get("track_mbid", "").strip()
                a_mbid = row.get("album_mbid", "").strip()
                # derive rough year from unix timestamp
                uts    = row.get("uts", "").strip()
                year   = int(uts[:4]) if uts.isdigit() and len(uts) >= 10 else None
                # actually uts is epoch seconds — convert
                if uts.isdigit():
                    import datetime
                    year = datetime.datetime.utcfromtimestamp(int(uts)).year
            elif is_spotify:
                artist = (row.get("artistName") or
                          row.get("master_metadata_album_artist_name") or "").strip()
                track  = (row.get("trackName") or
                          row.get("master_metadata_track_name") or "").strip()
                album  = ""
                t_mbid = ""
                a_mbid = ""
                year   = None
                ts     = (row.get("endTime") or row.get("ts") or "").strip()
                if ts:
                    try:
                        year = int(ts[:4])
                    except Exception:
                        pass
            else:
                continue

            if not artist or not track:
                continue
            key = (artist, track)
            if key not in tracks:
                tracks[key] = {"plays": 0, "albums": set(),
                               "track_mbid": t_mbid, "album_mbid": a_mbid,
                               "listen_year": year}
            tracks[key]["plays"] += 1
            if album:
                tracks[key]["albums"].add(album)
            # prefer non-empty mbid
            if t_mbid and not tracks[key]["track_mbid"]:
                tracks[key]["track_mbid"] = t_mbid

    # Convert album set → sorted list for JSON serialisation
    for v in tracks.values():
        v["albums"] = sorted(v["albums"])
    return tracks


# ── Phase 2: Spotify search + audio features ──────────────────────────────────

_spotify_rate_lock = threading.Lock()
_spotify_last_call: list[float] = [0.0]
_SPOTIFY_DELAY = 0.08   # ~12 req/s, well under 180/30s limit


def _spotify_rate_sleep():
    with _spotify_rate_lock:
        wait = _SPOTIFY_DELAY - (time.time() - _spotify_last_call[0])
        if wait > 0:
            time.sleep(wait)
        _spotify_last_call[0] = time.time()


def search_spotify_id(artist: str, track: str,
                      token: str, cache: dict) -> str | None:
    ck = f"{artist.lower().strip()}|||{track.lower().strip()}"
    if ck in cache:
        return cache[ck]
    _spotify_rate_sleep()
    q   = urllib.parse.quote(f'artist:{artist} track:{track}')
    url = f"https://api.spotify.com/v1/search?q={q}&type=track&limit=1"
    data = _get_json(url, headers={"Authorization": f"Bearer {token}"})
    sid  = None
    if data:
        items = data.get("tracks", {}).get("items", [])
        if items:
            sid = items[0]["id"]
    cache[ck] = sid   # cache None too (avoid re-searching misses)
    return sid


def fetch_audio_features_batch(ids: list[str], token: str) -> dict[str, dict]:
    """Fetch audio features for up to 100 Spotify IDs. Returns {id: features}."""
    if not ids:
        return {}
    _spotify_rate_sleep()
    url  = f"https://api.spotify.com/v1/audio-features?ids={','.join(ids)}"
    data = _get_json(url, headers={"Authorization": f"Bearer {token}"})
    if not data:
        return {}
    result = {}
    for feat in (data.get("audio_features") or []):
        if feat and feat.get("id"):
            result[feat["id"]] = feat
    return result


# ── Phase 3: MusicBrainz year ─────────────────────────────────────────────────

_mb_rate_lock = threading.Lock()
_mb_last_call: list[float] = [0.0]
_MB_DELAY = 1.05   # MusicBrainz: 1 req/s


def _mb_rate_sleep():
    with _mb_rate_lock:
        wait = _MB_DELAY - (time.time() - _mb_last_call[0])
        if wait > 0:
            time.sleep(wait)
        _mb_last_call[0] = time.time()


def _mb_year_from_mbid(mbid: str) -> int | None:
    """Look up recording by MBID, return earliest release year."""
    _mb_rate_sleep()
    url  = (f"https://musicbrainz.org/ws/2/recording/{mbid}"
            f"?inc=releases&fmt=json")
    data = _get_json(url, headers={"User-Agent": "MusicDiscoveryAgent/1.0 (contact@example.com)"})
    if not data:
        return None
    years = []
    for rel in data.get("releases", []):
        date = rel.get("date", "")
        if date and len(date) >= 4 and date[:4].isdigit():
            years.append(int(date[:4]))
    return min(years) if years else None


def _mb_year_search(artist: str, track: str) -> int | None:
    """Search MusicBrainz by artist + recording title."""
    _mb_rate_sleep()
    q    = urllib.parse.quote(f'recording:"{track}" AND artist:"{artist}"')
    url  = f"https://musicbrainz.org/ws/2/recording/?query={q}&limit=1&fmt=json"
    data = _get_json(url, headers={"User-Agent": "MusicDiscoveryAgent/1.0 (contact@example.com)"})
    if not data:
        return None
    recs = data.get("recordings", [])
    if not recs:
        return None
    years = []
    for rel in recs[0].get("releases", []):
        date = rel.get("date", "")
        if date and len(date) >= 4 and date[:4].isdigit():
            years.append(int(date[:4]))
    return min(years) if years else None


# ── Phase 4: Last.fm tags ─────────────────────────────────────────────────────

_lfm_rate_lock = threading.Lock()
_lfm_last_call: list[float] = [0.0]
_LFM_DELAY     = 0.22


def _lfm_rate_sleep():
    with _lfm_rate_lock:
        wait = _LFM_DELAY - (time.time() - _lfm_last_call[0])
        if wait > 0:
            time.sleep(wait)
        _lfm_last_call[0] = time.time()


def _fetch_lastfm(params: dict) -> list[tuple[str, int]]:
    params["format"] = "json"
    url = "https://ws.audioscrobbler.com/2.0/?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read().decode("utf-8"))
        tags_raw = data.get("toptags", {}).get("tag", [])
        if isinstance(tags_raw, dict):
            tags_raw = [tags_raw]
        return [(t["name"].lower().strip(), int(t["count"]))
                for t in tags_raw if int(t["count"]) > 0]
    except Exception:
        return []


def fetch_lastfm_tags(artist: str, track: str,
                      api_key: str, cache: dict) -> list[tuple[str, int]]:
    tk = f"t:{artist.lower().strip()}|||{track.lower().strip()}"
    if tk in cache:
        return cache[tk]
    _lfm_rate_sleep()
    tags = _fetch_lastfm({"method": "track.getTopTags", "artist": artist,
                           "track": track, "api_key": api_key, "autocorrect": "1"})
    if len(tags) < 3:
        ak = f"a:{artist.lower().strip()}"
        if ak not in cache:
            _lfm_rate_sleep()
            cache[ak] = _fetch_lastfm({"method": "artist.getTopTags",
                                        "artist": artist, "api_key": api_key,
                                        "autocorrect": "1"})
        if len(cache[ak]) > len(tags):
            tags = cache[ak]
    cache[tk] = tags
    return tags


# ── Lane scoring ──────────────────────────────────────────────────────────────

def score_lane(feat: dict, lane: dict) -> float:
    """
    Score a track's audio features against a lane definition.
    Returns 0.0–1.0: fraction of active dimensions the track falls within,
    weighted so centre-of-range scores higher than edge.
    """
    dims   = ["energy", "valence", "acousticness", "danceability"]
    score  = 0.0
    checks = 0

    for dim in dims:
        lo, hi = lane[dim]
        v = feat.get(dim)
        if v is None:
            continue
        checks += 1
        if lo <= v <= hi:
            # bonus for being near centre
            centre   = (lo + hi) / 2
            half_w   = (hi - lo) / 2
            score   += 1.0 - 0.3 * abs(v - centre) / (half_w + 1e-9)

    # tempo
    t_lo, t_hi = lane["tempo"]
    tempo = feat.get("tempo", 0)
    checks += 1
    if t_lo <= tempo <= t_hi:
        score += 1.0

    # mode (if lane specifies)
    if lane["mode"] is not None:
        checks += 1
        if feat.get("mode") == lane["mode"]:
            score += 1.0

    return score / max(checks, 1)


def assign_lane(feat: dict) -> str:
    best_lane  = "Unclassified"
    best_score = 0.0
    for lane in LANES:
        s = score_lane(feat, lane)
        if s > best_score:
            best_score = s
            best_lane  = lane["name"]
    return best_lane


# ── k-means clustering ────────────────────────────────────────────────────────

def cluster_tracks(metadata: dict[str, dict], n_clusters: int = 22) -> dict[str, int]:
    """
    Run k-means on Spotify audio features.
    Returns {track_key: cluster_id}.
    Features: energy, valence, acousticness, danceability,
              instrumentalness, tempo_norm (0-1), mode.
    """
    keys, vectors = [], []
    for k, v in metadata.items():
        af = v.get("audio_features")
        if not af:
            continue
        tempo_norm = min(af.get("tempo", 120) / 220.0, 1.0)
        vec = [
            af.get("energy",           0.5),
            af.get("valence",          0.5),
            af.get("acousticness",     0.5),
            af.get("danceability",     0.5),
            af.get("instrumentalness", 0.0),
            tempo_norm,
            float(af.get("mode", 0.5)),
        ]
        keys.append(k)
        vectors.append(vec)

    if len(vectors) < n_clusters:
        print(f"  Only {len(vectors)} enriched tracks — reducing clusters to {len(vectors)//2}")
        n_clusters = max(2, len(vectors) // 2)

    X      = np.array(vectors, dtype=float)
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    km.fit(Xs)

    return {k: int(c) for k, c in zip(keys, km.labels_)}


# ── Output helpers ────────────────────────────────────────────────────────────

def _format_cluster_report(metadata: dict, clusters: dict[str, int],
                            n_clusters: int) -> str:
    lines = []
    for cid in range(n_clusters):
        members = [(k, metadata[k]) for k, c in clusters.items()
                   if c == cid and k in metadata]
        if not members:
            continue
        members.sort(key=lambda x: -x[1]["plays"])
        top30 = members[:30]

        # Representative audio features (median of cluster)
        afs = [m[1]["audio_features"] for m in members if m[1].get("audio_features")]
        medians: dict = {}
        if afs:
            for dim in ["energy","valence","acousticness","danceability","tempo","mode"]:
                vals = [af[dim] for af in afs if dim in af]
                if vals:
                    medians[dim] = round(float(np.median(vals)), 3)

        # Top Last.fm tags across cluster
        tag_counter: Counter = Counter()
        for _, v in members:
            for tag, count in (v.get("lastfm_tags") or [])[:10]:
                tag_counter[tag] += count
        top_tags = [t for t, _ in tag_counter.most_common(15)]

        # Top-down lane assignment for this cluster
        if afs:
            lane_votes: Counter = Counter()
            for af in afs:
                lane_votes[assign_lane(af)] += 1
            dominant_lane = lane_votes.most_common(1)[0][0]
        else:
            dominant_lane = "Unknown"

        lines.append(f"\n{'='*70}")
        lines.append(f"CLUSTER {cid}  |  {len(members)} tracks  "
                     f"|  Dominant lane: {dominant_lane}")
        lines.append(f"  Audio features (median): {medians}")
        lines.append(f"  Top tags: {', '.join(top_tags)}")
        lines.append(f"\n  Top 30 by plays (for AllMusic / manual lookup):")
        for rank, (key, v) in enumerate(top30, 1):
            artist, track = key.split("|||", 1)
            af = v.get("audio_features") or {}
            lane = assign_lane(af) if af else "—"
            lines.append(f"  {rank:2d}. [{v['plays']:3d}x] {artist} — {track}")
            lines.append(f"       lane={lane}  "
                         f"E={af.get('energy','?'):.2f}  "
                         f"V={af.get('valence','?'):.2f}  "
                         f"A={af.get('acousticness','?'):.2f}  "
                         f"D={af.get('danceability','?'):.2f}" if af else "")
    return "\n".join(lines)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Enrich listening history with Spotify audio features.")
    ap.add_argument("--csv",     required=True, help="Path to Last.fm or Spotify history CSV")
    ap.add_argument("--profile", required=True, help="Profile name (e.g. marlonrando)")
    ap.add_argument("--skip-mb",   action="store_true", help="Skip MusicBrainz year lookup")
    ap.add_argument("--skip-lfm",  action="store_true", help="Skip Last.fm tag fetch")
    ap.add_argument("--clusters",  type=int, default=22, help="k-means cluster count (default 22)")
    ap.add_argument("--limit",     type=int, default=0,
                    help="Only enrich first N unique tracks (0=all, useful for testing)")
    args = ap.parse_args()

    csv_path    = Path(args.csv)
    profile     = args.profile
    meta_path   = MUSIC_AGENT_DIR / f"{profile}_track_metadata.json"
    cluster_path= MUSIC_AGENT_DIR / f"{profile}_clusters.json"
    report_path = MUSIC_AGENT_DIR / f"{profile}_cluster_top30.txt"

    secrets = _load_secrets()
    spotify_id  = secrets.get("SPOTIFY_CLIENT_ID")
    spotify_sec = secrets.get("SPOTIFY_CLIENT_SECRET")
    lfm_key     = secrets.get("LASTFM_API_KEY")

    if not spotify_id or not spotify_sec:
        sys.exit("ERROR: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET not found in secrets.toml")

    # ── Phase 1: Load history ─────────────────────────────────────────────────
    print(f"\n[1/4] Loading history from {csv_path.name} ...")
    history = load_history(csv_path)
    all_keys = sorted(history.keys(), key=lambda k: -history[k]["plays"])
    if args.limit:
        all_keys = all_keys[:args.limit]
    print(f"      {len(history):,} unique tracks total  |  "
          f"processing {len(all_keys):,}")

    # Load existing metadata (checkpoint resume)
    metadata: dict[str, dict] = _load_json(meta_path)
    sp_cache: dict             = _load_json(SPOTIFY_CACHE)
    lfm_cache: dict            = _load_json(LASTFM_CACHE)

    # ── Phase 2: Spotify IDs + Audio Features ─────────────────────────────────
    print(f"\n[2/4] Fetching Spotify audio features ...")
    token = _spotify_client_token(spotify_id, spotify_sec)
    token_time = time.time()

    def _refresh_token_if_needed():
        nonlocal token, token_time
        if time.time() - token_time > 3000:   # refresh before 3600s expiry
            token      = _spotify_client_token(spotify_id, spotify_sec)
            token_time = time.time()

    # Step 2a: resolve Spotify IDs for tracks that don't have them yet
    need_id = [(artist, track) for (artist, track) in all_keys
               if f"{artist.lower().strip()}|||{track.lower().strip()}" not in sp_cache]

    print(f"      Searching Spotify IDs for {len(need_id):,} tracks "
          f"({len(all_keys)-len(need_id):,} cached) ...")

    def _search_one(pair):
        artist, track = pair
        _refresh_token_if_needed()
        return (artist, track), search_spotify_id(artist, track, token, sp_cache)

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_search_one, pair): pair for pair in need_id}
        for fut in as_completed(futs):
            done += 1
            if done % 500 == 0 or done == len(need_id):
                print(f"      ... searched {done:,}/{len(need_id):,}", end="\r")
    print()

    _save_json(SPOTIFY_CACHE, sp_cache)

    # Step 2b: batch-fetch audio features for tracks with a Spotify ID
    keys_needing_features = [
        (artist, track) for (artist, track) in all_keys
        if (f"{artist}|||{track}" not in metadata or
            not metadata[f"{artist}|||{track}"].get("audio_features"))
        and sp_cache.get(f"{artist.lower().strip()}|||{track.lower().strip()}")
    ]
    sid_batches = []
    batch: list[tuple] = []
    for artist, track in keys_needing_features:
        ck  = f"{artist.lower().strip()}|||{track.lower().strip()}"
        sid = sp_cache.get(ck)
        if sid:
            batch.append((f"{artist}|||{track}", sid))
        if len(batch) == 100:
            sid_batches.append(batch); batch = []
    if batch:
        sid_batches.append(batch)

    print(f"      Fetching audio features in {len(sid_batches)} batches ...")
    feat_count = 0
    for i, batch in enumerate(sid_batches):
        _refresh_token_if_needed()
        ids   = [sid for _, sid in batch]
        feats = fetch_audio_features_batch(ids, token)
        for (key, sid), feat in zip(batch, [feats.get(s) for s in ids]):
            if feat:
                if key not in metadata:
                    metadata[key] = {}
                metadata[key]["audio_features"] = {
                    k: feat[k] for k in
                    ["energy","valence","acousticness","danceability",
                     "instrumentalness","tempo","mode","loudness",
                     "speechiness","liveness","key","time_signature"]
                    if k in feat
                }
                feat_count += 1
        if (i+1) % 10 == 0 or (i+1) == len(sid_batches):
            print(f"      ... batch {i+1}/{len(sid_batches)}  "
                  f"({feat_count} features stored)", end="\r")
    print()

    # ── Phase 3: MusicBrainz year ─────────────────────────────────────────────
    if not args.skip_mb:
        need_year = [(artist, track) for (artist, track) in all_keys
                     if not (metadata.get(f"{artist}|||{track}") or {}).get("year")]
        print(f"\n[3/4] MusicBrainz year lookup for {len(need_year):,} tracks ...")
        print(f"      (sequential at 1 req/s — this takes ~{len(need_year)//60} min)")
        done = 0
        for artist, track in need_year:
            key  = f"{artist}|||{track}"
            mbid = history[(artist, track)].get("track_mbid", "")
            year = None
            if mbid:
                year = _mb_year_from_mbid(mbid)
            if not year:
                year = _mb_year_search(artist, track)
            if key not in metadata:
                metadata[key] = {}
            metadata[key]["year"] = year
            done += 1
            if done % 100 == 0 or done == len(need_year):
                print(f"      ... {done:,}/{len(need_year):,}", end="\r")
            # checkpoint every 500
            if done % 500 == 0:
                _save_json(meta_path, metadata)
        print()
    else:
        print(f"\n[3/4] MusicBrainz skipped (--skip-mb)")

    # ── Phase 4: Last.fm tags ─────────────────────────────────────────────────
    if not args.skip_lfm and lfm_key:
        need_tags = [(artist, track) for (artist, track) in all_keys
                     if not (metadata.get(f"{artist}|||{track}") or {}).get("lastfm_tags")]
        print(f"\n[4/4] Last.fm tags for {len(need_tags):,} tracks ...")
        print(f"      (4 workers at 4.5 req/s — ~{len(need_tags)//900} min)")

        lock    = threading.Lock()
        counter = [0]

        def _fetch_tags_one(pair):
            artist, track = pair
            tags = fetch_lastfm_tags(artist, track, lfm_key, lfm_cache)
            key  = f"{artist}|||{track}"
            with lock:
                if key not in metadata:
                    metadata[key] = {}
                metadata[key]["lastfm_tags"] = tags
                counter[0] += 1
                n = counter[0]
            return n

        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_fetch_tags_one, pair) for pair in need_tags]
            for fut in as_completed(futs):
                n = fut.result()
                if n % 500 == 0 or n == len(need_tags):
                    print(f"      ... {n:,}/{len(need_tags):,}", end="\r")
                if n % 1000 == 0:
                    _save_json(LASTFM_CACHE, lfm_cache)
                    _save_json(meta_path, metadata)
        print()
        _save_json(LASTFM_CACHE, lfm_cache)
    else:
        print(f"\n[4/4] Last.fm tags skipped")

    # Merge play counts and album data into metadata
    for (artist, track), hist in history.items():
        key = f"{artist}|||{track}"
        if key not in metadata:
            metadata[key] = {}
        metadata[key].update({
            "artist":      artist,
            "track":       track,
            "plays":       hist["plays"],
            "albums":      hist["albums"],
            "album_count": len(hist["albums"]),
            "track_mbid":  hist.get("track_mbid", ""),
        })

    _save_json(meta_path, metadata)
    print(f"\n  Metadata saved → {meta_path}")

    # ── k-means clustering ────────────────────────────────────────────────────
    print(f"\n[Clustering] Running k-means with k={args.clusters} ...")
    clusters = cluster_tracks(metadata, n_clusters=args.clusters)
    _save_json(cluster_path, clusters)
    print(f"  Clusters saved → {cluster_path}")

    # ── Cluster report ────────────────────────────────────────────────────────
    print(f"\n[Report] Generating cluster top-30 report ...")
    report = _format_cluster_report(metadata, clusters, args.clusters)
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved → {report_path}")
    print(report[:3000])   # preview first 3000 chars

    # ── Summary ───────────────────────────────────────────────────────────────
    enriched = sum(1 for v in metadata.values() if v.get("audio_features"))
    print(f"\n{'='*50}")
    print(f"  Total tracks processed : {len(all_keys):,}")
    print(f"  Audio features fetched : {enriched:,}")
    print(f"  Clusters created       : {args.clusters}")
    print(f"  Metadata file          : {meta_path}")
    print(f"  Cluster report         : {report_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
