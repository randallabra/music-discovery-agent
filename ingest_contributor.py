#!/usr/bin/env python3
"""
ingest_contributor.py
=====================
Recursive contributor ingestion pipeline for the music taste graph.

Usage:
  python3 ingest_contributor.py --username teddxn \
      --data-dir "~/.music-agent/teddxn Spotify Extended Streaming History"

  python3 ingest_contributor.py --migrate-marlonrando   # one-time seed

Phases:
  0  — Initialise track_universe.db (if needed) / migrate marlonrando seed
  1  — Parse contributor Spotify export → {username}_profile.json
  2  — Identify net-new tracks not yet in universe
  3  — Enrich net-new tracks:
         3a  Last.fm track.getTopTags
         3b  Last.fm track.getSimilar  (≥6 plays seeds)
         3c  MusicBrainz Phase A  (artist-level genres/tags)
         3d  MusicBrainz Phase B  (recording-level tags + release title)
         3e  Discogs  (release genres/styles)
         3f  AcousticBrainz dump scan  (MBID-indexed audio features)
         3g  Million Song Dataset match
         3h  Label propagation  (AB feature inference from similar tracks)
  4  — Write enriched tracks to track_universe.db
  5  — Re-cluster universe (silhouette sweep, min cluster size ≥160)
  6  — Compute + store cluster affinity scores for all user profiles
"""

import argparse, json, os, re, sqlite3, sys, time, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
AGENT_DIR   = Path(os.path.expanduser("~/.music-agent"))
PROJECT_DIR = Path("/Users/randallabra/Marlonrando/Business/Assorted/AI Play/Spotify/music-agent")
DB_PATH     = AGENT_DIR / "track_universe.db"
SECRETS     = PROJECT_DIR.parent / ".streamlit" / "secrets.toml"

MARLONRANDO_META = AGENT_DIR / "marlonrando_track_metadata.json"
AB_DB            = AGENT_DIR / "acousticbrainz.db"
MSD_H5           = AGENT_DIR / "msd_summary_file.h5"

# ── logging helper ─────────────────────────────────────────────────────────────
import datetime
def log(msg, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0  —  Initialise / migrate
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    track_key        TEXT PRIMARY KEY,
    artist           TEXT,
    title            TEXT,
    track_mbid       TEXT,
    lastfm_tags      TEXT,
    mb_rec_genres    TEXT,
    mb_rec_tags      TEXT,
    mb_release_title TEXT,
    mb_artist_genres TEXT,
    mb_artist_tags   TEXT,
    discogs_genres   TEXT,
    discogs_styles   TEXT,
    ab               TEXT,
    msd              TEXT,
    cluster_v2       INTEGER,
    cluster_affinity TEXT,
    contributors     TEXT,
    added_ts         TEXT,
    enriched_ts      TEXT
);

CREATE TABLE IF NOT EXISTS user_profiles (
    username    TEXT,
    track_key   TEXT,
    plays       INTEGER,
    ms_played   INTEGER,
    albums      TEXT,
    spotify_uris TEXT,
    PRIMARY KEY (username, track_key)
);

CREATE TABLE IF NOT EXISTS cluster_meta (
    cluster_id   INTEGER PRIMARY KEY,
    size         INTEGER,
    top_tags     TEXT,
    acoustic_centroid TEXT,
    label        TEXT
);
"""

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    con.commit()
    return con

def migrate_marlonrando(con):
    """Seed track_universe.db from marlonrando_track_metadata.json."""
    log("Phase 0 — Migrating marlonrando seed data …")
    with open(MARLONRANDO_META) as f:
        meta = json.load(f)

    profile = {}
    now = datetime.datetime.utcnow().isoformat()
    batch_tracks = []
    batch_profile = []

    for key, v in meta.items():
        plays = v.get("plays", 0)
        # user profile row
        if plays > 0:
            profile[key] = plays
            batch_profile.append((
                "marlonrando", key,
                plays,
                0,  # ms_played not in marlonrando meta
                json.dumps(v.get("albums", [])),
                json.dumps([]),
            ))
        # track universe row
        batch_tracks.append((
            key,
            v.get("artist",""), v.get("track",""),
            v.get("track_mbid",""),
            json.dumps(v.get("lastfm_tags",[])),
            json.dumps(v.get("mb_rec_genres",[])),
            json.dumps(v.get("mb_rec_tags",[])),
            v.get("mb_release_title",""),
            json.dumps(v.get("mb_artist_genres",[])),
            json.dumps(v.get("mb_artist_tags",[])),
            json.dumps(v.get("discogs_genres",[])),
            json.dumps(v.get("discogs_styles",[])),
            json.dumps(v.get("ab")) if v.get("ab") else None,
            json.dumps(v.get("msd")) if v.get("msd") else None,
            v.get("cluster_v2"),
            None,  # cluster_affinity computed later
            json.dumps(["marlonrando"]),
            now, now,
        ))

    con.executemany("""
        INSERT OR REPLACE INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, batch_tracks)
    con.executemany("""
        INSERT OR REPLACE INTO user_profiles VALUES (?,?,?,?,?,?)
    """, batch_profile)
    con.commit()

    # save marlonrando_profile.json
    profile_path = AGENT_DIR / "marlonrando_profile.json"
    with open(profile_path, "w") as f:
        json.dump({"username": "marlonrando", "plays": profile}, f)

    log(f"    Migrated {len(batch_tracks):,} tracks  {len(batch_profile):,} play records")
    log(f"    marlonrando_profile.json saved → {profile_path}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1  —  Parse contributor Spotify export
# ══════════════════════════════════════════════════════════════════════════════

def parse_spotify_export(data_dir: Path, username: str, min_plays: int = 3,
                         block_artists: set = None):
    log(f"Phase 1 — Parsing Spotify export for {username} …")
    files = sorted(data_dir.glob("Streaming_History_Audio*.json"))
    if not files:
        sys.exit(f"No Streaming_History_Audio*.json files found in {data_dir}")

    tracks = defaultdict(lambda: {"plays":0,"ms_played":0,"albums":set(),"uris":set()})
    skipped = podcasts = blocked = 0
    block_artists = block_artists or set()

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        for r in data:
            artist = r.get("master_metadata_album_artist_name")
            track  = r.get("master_metadata_track_name")
            album  = r.get("master_metadata_album_album_name","")
            uri    = r.get("spotify_track_uri","")
            ms     = r.get("ms_played", 0)
            if not artist or not track: podcasts += 1; continue
            if ms < 30000: skipped += 1; continue
            if artist.lower() in block_artists: blocked += 1; continue
            key = f"{artist}|||{track}"
            tracks[key]["plays"]    += 1
            tracks[key]["ms_played"] += ms
            tracks[key]["artist"]    = artist
            tracks[key]["track"]     = track
            if album: tracks[key]["albums"].add(album)
            if uri:   tracks[key]["uris"].add(uri)

    profile = {k: v for k, v in tracks.items() if v["plays"] >= min_plays}
    log(f"    Raw events: {sum(v['plays'] for v in tracks.values()):,}")
    log(f"    Skipped <30s: {skipped:,}  Podcasts: {podcasts:,}  Blocked artists: {blocked:,}")
    log(f"    Unique tracks total: {len(tracks):,}")
    log(f"    ≥{min_plays} plays (corpus): {len(profile):,}")

    out = AGENT_DIR / f"{username}_profile.json"
    serialisable = {k: {**v, "albums": list(v["albums"]), "uris": list(v["uris"])}
                    for k, v in profile.items()}
    with open(out, "w") as f:
        json.dump({"username": username, "plays": serialisable}, f)
    log(f"    Profile saved → {out}")
    return profile

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2  —  Net-new track identification
# ══════════════════════════════════════════════════════════════════════════════

def find_net_new(con, profile: dict):
    log("Phase 2 — Identifying net-new tracks …")
    existing = {r[0] for r in con.execute("SELECT track_key FROM tracks")}
    net_new  = {k: v for k, v in profile.items() if k not in existing}
    log(f"    Profile tracks : {len(profile):,}")
    log(f"    Already in DB  : {len(profile) - len(net_new):,}")
    log(f"    Net-new        : {len(net_new):,}")
    return net_new

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3a  —  Last.fm track.getTopTags
# ══════════════════════════════════════════════════════════════════════════════

def enrich_lastfm_tags(net_new: dict, api_key: str):
    log(f"Phase 3a — Last.fm getTopTags ({len(net_new):,} tracks) …")
    import urllib.request, urllib.parse

    cache_path = AGENT_DIR / "lastfm_tag_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    done = hits = 0
    for key, v in net_new.items():
        artist, title = v["artist"], v["track"]
        ck = f"{artist}|||{title}"
        if ck in cache:
            v["lastfm_tags"] = cache[ck]
            done += 1
            continue
        try:
            params = urllib.parse.urlencode({
                "method": "track.getTopTags", "artist": artist,
                "track": title, "api_key": api_key, "format": "json"
            })
            req = urllib.request.urlopen(
                f"http://ws.audioscrobbler.com/2.0/?{params}", timeout=10)
            data = json.loads(req.read())
            tags = [t["name"] for t in data.get("toptags",{}).get("tag",[])
                    if int(t.get("count",0)) >= 10]
            v["lastfm_tags"] = tags
            cache[ck] = tags
            if tags: hits += 1
        except Exception:
            v["lastfm_tags"] = []
        done += 1
        if done % 500 == 0:
            log(f"    {done:,}/{len(net_new):,}  hits={hits:,}")
            cache_path.write_text(json.dumps(cache))
        time.sleep(0.25)

    cache_path.write_text(json.dumps(cache))
    log(f"    Done — {hits:,}/{len(net_new):,} returned tags")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3b  —  Last.fm track.getSimilar
# ══════════════════════════════════════════════════════════════════════════════

def enrich_similarity_graph(profile: dict, net_new: dict, api_key: str):
    seeds = {k: v for k, v in profile.items() if v["plays"] >= 6}
    log(f"Phase 3b — Last.fm getSimilar ({len(seeds):,} seeds ≥6 plays) …")
    import urllib.request, urllib.parse

    cache_path = AGENT_DIR / "similarity_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    graph_path = AGENT_DIR / "similarity_graph.json"
    graph = json.loads(graph_path.read_text()) if graph_path.exists() else {}

    done = 0
    for key, v in seeds.items():
        artist, title = v["artist"], v["track"]
        ck = f"{artist}|||{title}"
        if ck in cache:
            graph[key] = cache[ck]
            done += 1
            continue
        try:
            params = urllib.parse.urlencode({
                "method": "track.getSimilar", "artist": artist,
                "track": title, "limit": 100,
                "api_key": api_key, "format": "json"
            })
            req = urllib.request.urlopen(
                f"http://ws.audioscrobbler.com/2.0/?{params}", timeout=10)
            data = json.loads(req.read())
            similar = []
            for t in data.get("similartracks",{}).get("track",[]):
                score = float(t.get("match", 0))
                if score < 0.65: break
                similar.append({
                    "artist": t.get("artist",{}).get("name",""),
                    "title":  t.get("name",""),
                    "score":  score,
                    "mbid":   t.get("mbid",""),
                })
            entry = {"artist": artist, "title": title,
                     "plays": v["plays"], "similar": similar}
            graph[key] = entry
            cache[ck]  = entry
        except Exception:
            pass
        done += 1
        if done % 500 == 0:
            log(f"    {done:,}/{len(seeds):,}")
            cache_path.write_text(json.dumps(cache))
        time.sleep(1.1)

    cache_path.write_text(json.dumps(cache))
    graph_path.write_text(json.dumps(graph))
    log(f"    Similarity graph updated → {graph_path}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3c  —  MusicBrainz Phase A (artist-level)
# ══════════════════════════════════════════════════════════════════════════════

def enrich_mb_artist(net_new: dict):
    log("Phase 3c — MusicBrainz artist-level enrichment …")
    import urllib.request, urllib.parse

    cache_path = AGENT_DIR / "mb_artist_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    artists = {}
    for k, v in net_new.items():
        a = v["artist"]
        if a not in artists: artists[a] = []
        artists[a].append(k)

    headers = {"User-Agent": "MusicTasteGraph/1.0 (music-agent)"}
    done = hits = 0

    for artist, keys in artists.items():
        if artist in cache:
            data = cache[artist]
        else:
            try:
                params = urllib.parse.urlencode({"query": f'artist:"{artist}"', "fmt": "json"})
                req = urllib.request.Request(
                    f"https://musicbrainz.org/ws/2/artist?{params}", headers=headers)
                resp = urllib.request.urlopen(req, timeout=10)
                results = json.loads(resp.read()).get("artists", [])
                data = {}
                if results:
                    mbid = results[0]["id"]
                    req2 = urllib.request.Request(
                        f"https://musicbrainz.org/ws/2/artist/{mbid}?inc=genres+tags&fmt=json",
                        headers=headers)
                    resp2 = urllib.request.urlopen(req2, timeout=10)
                    adata = json.loads(resp2.read())
                    data = {
                        "genres": [g["name"] for g in adata.get("genres", [])],
                        "tags":   [t["name"] for t in adata.get("tags", [])
                                   if t.get("count", 0) >= 2],
                    }
                    if data["genres"] or data["tags"]: hits += 1
                cache[artist] = data
                time.sleep(1.1)
            except Exception:
                cache[artist] = {}
                data = {}

        for k in keys:
            net_new[k]["mb_artist_genres"] = data.get("genres", [])
            net_new[k]["mb_artist_tags"]   = data.get("tags", [])

        done += 1
        if done % 200 == 0:
            log(f"    {done:,}/{len(artists):,} artists  hits={hits:,}")
            cache_path.write_text(json.dumps(cache))

    cache_path.write_text(json.dumps(cache))
    log(f"    Done — {hits:,}/{len(artists):,} artists returned data")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3d  —  MusicBrainz Phase B (recording-level)
# ══════════════════════════════════════════════════════════════════════════════

def enrich_mb_recording(net_new: dict):
    tracks_with_mbid = {k: v for k, v in net_new.items() if v.get("track_mbid")}
    log(f"Phase 3d — MusicBrainz recording-level ({len(tracks_with_mbid):,} with MBID) …")
    import urllib.request

    cache_path = AGENT_DIR / "mb_recording_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    headers = {"User-Agent": "MusicTasteGraph/1.0 (music-agent)"}
    done = hits = 0

    for key, v in tracks_with_mbid.items():
        mbid = v["track_mbid"]
        if mbid in cache:
            data = cache[mbid]
        else:
            try:
                req = urllib.request.Request(
                    f"https://musicbrainz.org/ws/2/recording/{mbid}?inc=genres+tags+releases&fmt=json",
                    headers=headers)
                resp = urllib.request.urlopen(req, timeout=10)
                rdata = json.loads(resp.read())
                genres   = [g["name"] for g in rdata.get("genres", [])]
                tags     = [t["name"] for t in rdata.get("tags", [])
                            if t.get("count", 0) >= 2]
                releases = rdata.get("releases", [])
                release  = releases[0].get("title", "") if releases else ""
                data = {"genres": genres, "tags": tags, "release": release}
                if genres or tags: hits += 1
                cache[mbid] = data
                time.sleep(1.1)
            except Exception:
                cache[mbid] = {}
                data = {}

        v["mb_rec_genres"]    = data.get("genres", [])
        v["mb_rec_tags"]      = data.get("tags", [])
        v["mb_release_title"] = data.get("release", "")
        done += 1
        if done % 500 == 0:
            log(f"    {done:,}/{len(tracks_with_mbid):,}  hits={hits:,}")
            cache_path.write_text(json.dumps(cache))

    cache_path.write_text(json.dumps(cache))
    log(f"    Done — {hits:,}/{len(tracks_with_mbid):,} returned recording tags")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3e  —  Discogs
# ══════════════════════════════════════════════════════════════════════════════

def enrich_discogs(net_new: dict, token: str):
    log(f"Phase 3e — Discogs release enrichment ({len(net_new):,} tracks) …")
    import urllib.request, urllib.parse

    cache_path = AGENT_DIR / "discogs_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    headers = {"User-Agent": "MusicTasteGraph/1.0",
               "Authorization": f"Discogs token={token}"}
    done = hits = 0

    for key, v in net_new.items():
        artist = v["artist"]
        albums_val = v.get("albums")
        albums_list = list(albums_val) if albums_val else []
        album  = (v.get("mb_release_title") or
                  (albums_list[0] if albums_list else ""))
        if not album:
            v["discogs_genres"] = []; v["discogs_styles"] = []
            done += 1; continue

        ck = f"{artist}|||{album}"
        if ck in cache:
            d = cache[ck]
        else:
            try:
                params = urllib.parse.urlencode(
                    {"q": f"{artist} {album}", "type": "release", "per_page": 3})
                req = urllib.request.Request(
                    f"https://api.discogs.com/database/search?{params}", headers=headers)
                resp = urllib.request.urlopen(req, timeout=10)
                results = json.loads(resp.read()).get("results", [])
                d = {}
                if results:
                    r = results[0]
                    d = {"genres": r.get("genre", []), "styles": r.get("style", [])}
                    if d["genres"] or d["styles"]: hits += 1
                cache[ck] = d
                time.sleep(2.5)
            except Exception:
                cache[ck] = {}
                d = {}

        v["discogs_genres"] = d.get("genres", [])
        v["discogs_styles"] = d.get("styles", [])
        done += 1
        if done % 200 == 0:
            log(f"    {done:,}/{len(net_new):,}  hits={hits:,}")
            cache_path.write_text(json.dumps(cache))

    cache_path.write_text(json.dumps(cache))
    log(f"    Done — {hits:,}/{len(net_new):,} returned release tags")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3f  —  AcousticBrainz
# ══════════════════════════════════════════════════════════════════════════════

def enrich_acousticbrainz(net_new: dict):
    if not AB_DB.exists():
        log("Phase 3f — AcousticBrainz DB not found, skipping")
        return
    log("Phase 3f — AcousticBrainz lookup …")
    import sqlite3 as sq3

    AB_FIELDS = [
        "mood_happy","mood_sad","mood_aggressive","mood_relaxed",
        "mood_acoustic","mood_party","mood_electronic","danceability",
        "bpm","average_loudness","key_key","key_scale","key_strength",
        "tonal_prob","is_tonal","is_instrumental","voice_prob",
    ]

    con_ab = sq3.connect(AB_DB)
    hits = 0
    for key, v in net_new.items():
        mbid = v.get("track_mbid","")
        if not mbid: continue
        row = con_ab.execute(
            f"SELECT {','.join(AB_FIELDS)} FROM recordings WHERE mbid=?", (mbid,)
        ).fetchone()
        if row:
            v["ab"] = {"source":"direct","confidence":1.0,
                       **{f:val for f,val in zip(AB_FIELDS,row) if val is not None}}
            hits += 1
    con_ab.close()
    log(f"    Done — {hits:,} direct AB hits")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3g  —  Million Song Dataset
# ══════════════════════════════════════════════════════════════════════════════

def enrich_msd(net_new: dict):
    if not MSD_H5.exists():
        log("Phase 3g — MSD file not found, skipping")
        return
    log("Phase 3g — Million Song Dataset matching …")

    import pickle

    MSD_PKL = AGENT_DIR / "msd_index.pkl"

    def norm(s):
        return re.sub(r"[^a-z0-9]","",str(s).lower())

    # Load from pre-built pkl index if available (built once from HDF5, ~1s to reload)
    if MSD_PKL.exists():
        log("    Loading MSD index from cache …")
        with open(MSD_PKL, "rb") as f:
            by_key = pickle.load(f)
        log(f"    Loaded {len(by_key):,} MSD entries from pkl cache")
    else:
        try:
            import h5py
        except ImportError:
            log("    h5py not installed, skipping MSD"); return

        log("    Building MSD index from HDF5 — bulk column reads (one-time, ~3 min) …")
        by_key = {}
        with h5py.File(MSD_H5, "r") as h5:
            meta = h5["metadata"]["songs"]
            anal = h5["analysis"]["songs"]
            mb_s = h5["musicbrainz"]["songs"]
            n    = len(meta)
            # Load entire columns at once — orders of magnitude faster than per-row access
            artists  = meta["artist_name"][:]
            titles   = meta["title"][:]
            tempos   = anal["tempo"][:]
            loudness = anal["loudness"][:]
            keys_arr = anal["key"][:]
            modes    = anal["mode"][:]
            tsigs    = anal["time_signature"][:]
            years    = mb_s["year"][:]
            log(f"    Columns loaded — indexing {n:,} rows …")
            for i in range(n):
                a = artists[i]; t = titles[i]
                if isinstance(a, bytes): a = a.decode("utf-8","ignore")
                if isinstance(t, bytes): t = t.decode("utf-8","ignore")
                k = f"{norm(a)}|||{norm(t)}"
                ld = float(loudness[i]); tp = float(tempos[i])
                if ld == 0 and tp == 0: continue
                by_key[k] = {
                    "msd_loudness": round(ld,3), "msd_tempo": round(tp,3),
                    "msd_key": int(keys_arr[i]), "msd_mode": int(modes[i]),
                    "msd_time_signature": int(tsigs[i]),
                    "msd_year": int(years[i]) if years[i] > 0 else None,
                    "msd_source": "msd_exact", "msd_match_score": 1.0,
                }
                if i % 100_000 == 0 and i > 0:
                    log(f"    {i:,}/{n:,} rows indexed …")

        log(f"    Saving pkl cache → {MSD_PKL}")
        with open(MSD_PKL, "wb") as f:
            pickle.dump(by_key, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"    Saved {len(by_key):,} entries — future runs will load in <2s")

    hits = 0
    for key, v in net_new.items():
        nk = f"{norm(v['artist'])}|||{norm(v['track'])}"
        if nk in by_key:
            v["msd"] = by_key[nk]; hits += 1
    log(f"    Done — {hits:,} MSD matches")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3g.5  —  MSD→AB acoustic imputation via cosine similarity
# ══════════════════════════════════════════════════════════════════════════════

def enrich_msd_imputation(net_new: dict, con):
    """Impute AB mood features for MSD-matched tracks using cosine similarity
    in MSD acoustic space (tempo, loudness, key, mode, time_signature).
    Runs before getSimilar label propagation — acoustically-grounded path first.
    """
    log("Phase 3g.5 — MSD→AB acoustic imputation …")
    try:
        import numpy as np
    except ImportError:
        log("    numpy not available, skipping"); return

    MSD_FEATS = ["msd_tempo", "msd_loudness", "msd_key", "msd_mode", "msd_time_signature"]
    AB_FEATS  = ["mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed",
                 "mood_acoustic", "mood_party", "mood_electronic",
                 "danceability", "tonal_prob", "voice_prob"]

    # Tracks needing imputation: have MSD but no AB
    targets = {k: v for k, v in net_new.items()
               if v.get("msd") and not v.get("ab")}
    if not targets:
        log("    No tracks need MSD imputation"); return

    # Build AB reference pool: universe DB tracks + this batch — must have both MSD + AB
    ab_refs = {}
    for row in con.execute(
            "SELECT track_key, msd, ab FROM tracks WHERE msd IS NOT NULL AND ab IS NOT NULL"):
        try:
            msd = json.loads(row[1]); ab = json.loads(row[2])
            if all(msd.get(f) is not None for f in MSD_FEATS):
                ab_refs[row[0]] = {"msd": msd, "ab": ab}
        except Exception:
            pass
    for k, v in net_new.items():
        if v.get("msd") and v.get("ab"):
            if all(v["msd"].get(f) is not None for f in MSD_FEATS):
                ab_refs[k] = {"msd": v["msd"], "ab": v["ab"]}

    if not ab_refs:
        log("    No AB reference tracks available — skipping"); return

    ref_keys = list(ab_refs.keys())

    # Feature matrices
    ref_msd = np.array([[ab_refs[k]["msd"].get(f, 0.0) for f in MSD_FEATS]
                        for k in ref_keys], dtype=float)
    ref_ab  = np.array([[ab_refs[k]["ab"].get(f, 0.0)  for f in AB_FEATS]
                        for k in ref_keys], dtype=float)

    # Z-score normalise MSD features (scales: tempo ~120 BPM, loudness ~-10 dB,
    # key 0-11, mode 0-1, time_sig 1-7 — must normalise before cosine)
    means = ref_msd.mean(axis=0)
    stds  = ref_msd.std(axis=0); stds[stds == 0] = 1.0
    ref_norm = (ref_msd - means) / stds
    row_norms = np.linalg.norm(ref_norm, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    ref_unit = ref_norm / row_norms  # unit vectors for cosine sim

    # Target matrix — same normalisation (use ref means/stds, not target's own)
    tgt_keys = list(targets.keys())
    tgt_msd  = np.array([[targets[k]["msd"].get(f, 0.0) for f in MSD_FEATS]
                         for k in tgt_keys], dtype=float)
    tgt_norm = (tgt_msd - means) / stds
    tgt_row_norms = np.linalg.norm(tgt_norm, axis=1, keepdims=True)
    tgt_row_norms[tgt_row_norms == 0] = 1.0
    tgt_unit = tgt_norm / tgt_row_norms

    # Cosine similarity: (n_targets × n_refs) — single matrix multiply
    sim_matrix = tgt_unit @ ref_unit.T

    K = min(5, len(ref_keys))
    imputed = 0
    for i, key in enumerate(tgt_keys):
        sims    = sim_matrix[i]
        top_idx = np.argpartition(sims, -K)[-K:]
        top_sim = sims[top_idx]
        pos     = top_sim > 0
        if not pos.any(): continue
        top_idx = top_idx[pos]; top_sim = top_sim[pos]
        weights = top_sim / top_sim.sum()
        ab_vec  = (ref_ab[top_idx] * weights[:, None]).sum(axis=0)
        ab_dict = {f: round(float(ab_vec[j]), 4) for j, f in enumerate(AB_FEATS)}
        ab_dict["ab_source"] = "msd_imputed"
        targets[key]["ab"] = ab_dict
        imputed += 1

    log(f"    Imputed AB features for {imputed:,}/{len(targets):,} tracks")
    log(f"    Reference pool: {len(ab_refs):,} tracks with both MSD + AB data")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3h  —  Label propagation (getSimilar fallback for tracks with no MSD or AB)
# ══════════════════════════════════════════════════════════════════════════════

def enrich_label_propagation(net_new: dict, con):
    log("Phase 3h — Label propagation (getSimilar AB fallback) …")
    graph_path = AGENT_DIR / "similarity_graph.json"
    if not graph_path.exists(): log("    No similarity graph, skipping"); return

    graph = json.loads(graph_path.read_text())

    NUMERIC = ["mood_happy","mood_sad","mood_aggressive","mood_relaxed",
               "mood_acoustic","mood_party","mood_electronic","danceability",
               "tonal_prob","voice_prob"]

    # build AB lookup from universe DB + net_new
    ab_lookup = {}
    for row in con.execute("SELECT track_key, ab FROM tracks WHERE ab IS NOT NULL"):
        ab_lookup[row[0]] = json.loads(row[1])
    for k, v in net_new.items():
        if v.get("ab"): ab_lookup[k] = v["ab"]

    inferred = 0
    for key, v in net_new.items():
        if v.get("ab"): continue  # already has direct AB
        entry = graph.get(key, {})
        similar = entry.get("similar", [])
        if not similar: continue

        sources = []
        for s in similar:
            sk = f"{s['artist']}|||{s['title']}"
            if sk in ab_lookup:
                sources.append((s["score"], ab_lookup[sk]))

        if not sources: continue
        total_w = sum(sc for sc, _ in sources)
        if total_w == 0: continue

        inferred_ab = {"source": "inferred", "confidence": round(total_w/len(sources),3),
                       "n_sources": len(sources)}
        for field in NUMERIC:
            vals = [(sc, ab[field]) for sc, ab in sources if field in ab and ab[field] is not None]
            if vals:
                tw = sum(sc for sc,_ in vals)
                inferred_ab[field] = round(sum(sc*val for sc,val in vals)/tw, 4)

        v["ab"] = inferred_ab
        inferred += 1

    log(f"    Done — {inferred:,} tracks received inferred AB features")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4  —  Write to track_universe.db
# ══════════════════════════════════════════════════════════════════════════════

def write_to_universe(con, net_new: dict, profile: dict, username: str):
    log(f"Phase 4 — Writing {len(net_new):,} net-new tracks to universe …")
    now = datetime.datetime.utcnow().isoformat()

    batch_tracks = []
    for key, v in net_new.items():
        batch_tracks.append((
            key,
            v.get("artist",""), v.get("track",""),
            v.get("track_mbid",""),
            json.dumps(v.get("lastfm_tags",[])),
            json.dumps(v.get("mb_rec_genres",[])),
            json.dumps(v.get("mb_rec_tags",[])),
            v.get("mb_release_title",""),
            json.dumps(v.get("mb_artist_genres",[])),
            json.dumps(v.get("mb_artist_tags",[])),
            json.dumps(v.get("discogs_genres",[])),
            json.dumps(v.get("discogs_styles",[])),
            json.dumps(v.get("ab")) if v.get("ab") else None,
            json.dumps(v.get("msd")) if v.get("msd") else None,
            None, None,
            json.dumps([username]),
            now, now,
        ))

    con.executemany("""
        INSERT OR REPLACE INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, batch_tracks)

    # update contributors for existing tracks this user also plays
    for key in profile:
        if key not in net_new:
            row = con.execute(
                "SELECT contributors FROM tracks WHERE track_key=?", (key,)
            ).fetchone()
            if row:
                contribs = json.loads(row[0] or "[]")
                if username not in contribs:
                    contribs.append(username)
                    con.execute("UPDATE tracks SET contributors=? WHERE track_key=?",
                                (json.dumps(contribs), key))

    # write user profile rows
    batch_profile = []
    for key, v in profile.items():
        batch_profile.append((
            username, key, v["plays"], v.get("ms_played",0),
            json.dumps(list(v.get("albums",set()))),
            json.dumps(list(v.get("uris",set()))),
        ))
    con.executemany("""
        INSERT OR REPLACE INTO user_profiles VALUES (?,?,?,?,?,?)
    """, batch_profile)
    con.commit()
    log(f"    Written. Universe now has {con.execute('SELECT COUNT(*) FROM tracks').fetchone()[0]:,} tracks")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5  —  Re-cluster
# ══════════════════════════════════════════════════════════════════════════════

def recluster(con, min_cluster_size: int = 160):
    log("Phase 5 — Re-clustering universe …")
    from scipy.sparse import hstack, csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    import warnings; warnings.filterwarnings("ignore")

    rows = con.execute("""
        SELECT track_key, lastfm_tags, mb_rec_genres, mb_rec_tags,
               discogs_styles, discogs_genres, mb_artist_genres, mb_artist_tags,
               ab, msd
        FROM tracks
    """).fetchall()

    keys, docs, ac_rows, has_ac_list = [], [], [], []

    def flatten(raw):
        out = []
        if isinstance(raw, str): out.append(raw)
        elif isinstance(raw, list):
            for item in raw: out.extend(flatten(item))
        return out

    def clean(tags):
        return [t for t in [str(x).lower().replace(" ","_") for x in tags]
                if any(c.isalpha() for c in t) and len(t) >= 3]

    AC_FIELDS = ["mood_happy","mood_sad","mood_aggressive","mood_relaxed",
                 "mood_acoustic","mood_party","mood_electronic","danceability",
                 "tonal_prob","voice_prob","msd_loudness","msd_tempo","msd_mode"]

    for row in rows:
        key = row[0]
        lfm  = clean(flatten(json.loads(row[1] or "[]")))
        rec  = clean(flatten(json.loads(row[2] or "[]") + json.loads(row[3] or "[]")))
        alb  = clean(flatten(json.loads(row[4] or "[]") + json.loads(row[5] or "[]")))
        art  = clean(flatten(json.loads(row[6] or "[]") + json.loads(row[7] or "[]")))
        doc  = rec*3 + lfm*2 + alb
        if not doc: doc = art
        docs.append(" ".join(doc) if doc else "unknown")
        keys.append(key)

        ab  = json.loads(row[8]) if row[8] else {}
        msd = json.loads(row[9]) if row[9] else {}
        ac  = []
        has = bool(ab or msd)
        for f in AC_FIELDS:
            val = ab.get(f) if ab.get(f) is not None else msd.get(f"msd_{f.replace('msd_','')}")
            ac.append(float(val) if val is not None else np.nan)
        ac_rows.append(ac)
        has_ac_list.append(has)

    # TF-IDF
    tfidf = TfidfVectorizer(min_df=3, max_df=0.80, max_features=8000,
                            ngram_range=(1,2), sublinear_tf=True)
    T = tfidf.fit_transform(docs)

    # Acoustic
    A = np.array(ac_rows, dtype=float)
    col_m = np.nanmean(A, axis=0)
    col_m = np.where(np.isnan(col_m), 0.0, col_m)
    for j in range(A.shape[1]):
        A[np.isnan(A[:,j]), j] = col_m[j]
    A_scaled = StandardScaler().fit_transform(A)
    has_ac = np.array(has_ac_list)
    A_scaled[~has_ac] = 0.0

    # Stage-1 (acoustic tracks only)
    s1_idx = [i for i, h in enumerate(has_ac_list) if h]
    s2_idx = [i for i, h in enumerate(has_ac_list) if not h]
    T_s1 = T[s1_idx]
    A_s1 = csr_matrix(A_scaled[s1_idx])
    X_s1 = hstack([T_s1, A_s1])

    # Silhouette sweep
    np.random.seed(42)
    N1 = len(s1_idx)
    samp = np.random.choice(N1, min(2000, N1), replace=False)
    X_samp = X_s1[samp].toarray()

    best_k, best_s, best_km = None, -1, None
    for k in range(20, 36):
        km = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=42, batch_size=512)
        km.fit(X_s1)
        lbls = km.labels_
        # check min cluster size
        sizes = np.bincount(lbls)
        if sizes.min() < min_cluster_size:
            log(f"    k={k:2d}  min_size={sizes.min()} < {min_cluster_size}, skip")
            continue
        s = silhouette_score(X_samp, lbls[samp], metric="euclidean") if len(set(lbls[samp]))>1 else -1
        log(f"    k={k:2d}  silhouette={s:.4f}  min_cluster={sizes.min()}")
        if s > best_s:
            best_s, best_k, best_km = s, k, km

    if best_km is None:
        log("    No k met min_cluster_size — relaxing constraint, using best silhouette")
        for k in range(20, 36):
            km = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=42, batch_size=512)
            km.fit(X_s1)
            lbls = km.labels_
            s = silhouette_score(X_samp, lbls[samp], metric="euclidean") if len(set(lbls[samp]))>1 else -1
            if s > best_s:
                best_s, best_k, best_km = s, k, km

    log(f"    → Chosen k={best_k}  silhouette={best_s:.4f}")
    s1_labels = best_km.labels_

    # Stage-2: assign tag-only tracks
    from sklearn.metrics.pairwise import cosine_similarity
    T_s2 = T[s2_idx]
    centroids = np.zeros((best_k, T_s1.shape[1]))
    for cid in range(best_k):
        mask = s1_labels == cid
        if mask.any(): centroids[cid] = T_s1[mask].mean(axis=0)
    s2_labels = cosine_similarity(T_s2, csr_matrix(centroids)).argmax(axis=1)

    # Write cluster assignments
    all_labels = {}
    for i, idx in enumerate(s1_idx): all_labels[keys[idx]] = int(s1_labels[i])
    for i, idx in enumerate(s2_idx): all_labels[keys[idx]] = int(s2_labels[i])

    con.executemany("UPDATE tracks SET cluster_v2=? WHERE track_key=?",
                    [(v, k) for k, v in all_labels.items()])
    con.commit()
    log(f"    Cluster assignments written for {len(all_labels):,} tracks")
    return best_k

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6  —  Cluster affinity scores
# ══════════════════════════════════════════════════════════════════════════════

def compute_affinity_scores(con):
    log("Phase 6 — Computing cluster affinity scores …")
    users = [r[0] for r in con.execute("SELECT DISTINCT username FROM user_profiles")]
    k = con.execute("SELECT MAX(cluster_v2) FROM tracks").fetchone()[0]
    if k is None: return
    k += 1

    for user in users:
        plays = {r[0]: r[1] for r in con.execute(
            "SELECT track_key, plays FROM user_profiles WHERE username=?", (user,))}
        total_plays = sum(plays.values())
        if total_plays == 0: continue

        affinity = defaultdict(float)
        for track_key, p in plays.items():
            row = con.execute(
                "SELECT cluster_v2 FROM tracks WHERE track_key=?", (track_key,)
            ).fetchone()
            if row and row[0] is not None:
                affinity[row[0]] += p / total_plays

        affinity_json = json.dumps({str(k): round(v,4) for k,v in affinity.items()})

        # store in user_profiles as a summary row (cluster_id=-1 as affinity store)
        profile_path = AGENT_DIR / f"{user}_profile.json"
        if profile_path.exists():
            prof = json.loads(profile_path.read_text())
            prof["cluster_affinity"] = dict(affinity)
            profile_path.write_text(json.dumps(prof))

    log(f"    Affinity scores written for {len(users)} users")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7  —  Pocket classifier + {username}_track_index.json
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage-1 AB thresholds ─────────────────────────────────────────────────────
AB_QUIET_THRESH      = 0.65
AB_AGGRESSIVE_THRESH = 0.60
AB_ELECTRONIC_THRESH = 0.65
AB_HAPPY_THRESH      = 0.60
AB_DANCE_THRESH      = 0.55

# ── Tag sets ──────────────────────────────────────────────────────────────────
QUIET_TAGS = {
    "acoustic","folk","singer-songwriter","americana","alt-country","blues",
    "country","bluegrass","indie folk","sad","melancholy","introspective",
    "ballad","quiet","mellow","fingerpicking","unplugged","sparse",
}
AGGRESSIVE_TAGS = {
    "metal","heavy metal","hard rock","punk","hardcore","thrash metal",
    "death metal","black metal","noise rock","garage rock","stoner rock",
    "post-hardcore","screamo","aggressive","intense","heavy",
}
ELECTRONIC_TAGS = {
    "electronic","ambient","synth-pop","synthpop","electro","techno","house",
    "idm","industrial","electronica","downtempo","trip-hop","drum and bass",
    "synth","synthesizer","experimental electronic",
}
HAPPY_TAGS = {
    "pop","pop rock","britpop","indie pop","dance","danceable","upbeat",
    "jangle pop","power pop","soul","funk","r&b","reggae","ska","disco",
    "happy","feel good","sunny","optimistic","bubblegum",
}
SYNTH_TAGS = {
    "synth","synthesizer","synth-pop","synthpop","electro","electronic",
    "industrial","idm","ambient","techno","house","drum machine",
}
# These genres only vote electronic when synth-family tags are also present
ELECTRONIC_COPRESENCE_GUARD = {"new wave","post-punk","coldwave","minimal wave"}

VETO = {
    "quiet":      {"metal","heavy metal","hard rock","punk","hardcore","death metal","thrash"},
    "aggressive": {"acoustic","folk","singer-songwriter","ambient","new age"},
    "electronic": set(),
    "happy":      {"metal","heavy metal","death","dark","sad","melancholy","doom","funeral"},
}

# ── Artist-level overrides (force_pocket, blocked_pockets) ───────────────────
AO = {
    "Tom Waits":           (None,        frozenset({"electronic","happy"})),
    "Mad Season":          ("aggressive",frozenset({"electronic","happy"})),
    "John Frusciante":     (None,        frozenset({"electronic"})),   # album lift below
    "The Smashing Pumpkins":(None,       frozenset({"electronic"})),
    "Smashing Pumpkins":   (None,        frozenset({"electronic"})),
    "Talking Heads":       ("happy",     frozenset({"electronic","aggressive","quiet"})),
    "Alice in Chains":     (None,        frozenset({"happy","electronic"})),
    "Nirvana":             (None,        frozenset({"happy","electronic"})),
    "The Cure":            (None,        frozenset({"aggressive","happy"})),
    "Brian Eno":           (None,        frozenset({"aggressive","happy"})),
    "New Order":           ("electronic",frozenset({"quiet","aggressive"})),
    "The xx":              ("electronic",frozenset({"aggressive","happy"})),
    "Joy Division":        ("electronic",frozenset({"happy","aggressive"})),
    "Radiohead":           (None,        frozenset({"happy"})),
    "Nick Cave & The Bad Seeds":(None,   frozenset({"happy","electronic"})),
    "Nick Cave and the Bad Seeds":(None, frozenset({"happy","electronic"})),
    "Leonard Cohen":       ("quiet",     frozenset({"electronic","happy","aggressive"})),
    "Bob Dylan":           ("quiet",     frozenset({"electronic","happy","aggressive"})),
    "Neil Young":          (None,        frozenset({"electronic","happy"})),
    "Tom Petty":           ("happy",     frozenset({"electronic","quiet","aggressive"})),
    "Tom Petty and the Heartbreakers":("happy",frozenset({"electronic","quiet","aggressive"})),
    "R.E.M.":              (None,        frozenset({"electronic","aggressive"})),
    "Pixies":              ("aggressive",frozenset({"electronic","happy"})),
    "Sonic Youth":         ("aggressive",frozenset({"happy"})),
    "Lou Reed":            (None,        frozenset({"happy","electronic"})),
    "The Velvet Underground":(None,      frozenset({"happy","electronic"})),
    "David Bowie":         (None,        frozenset()),
    "Pink Floyd":          (None,        frozenset({"happy","aggressive"})),
    "Led Zeppelin":        (None,        frozenset({"electronic","happy"})),
    "The Beatles":         ("happy",     frozenset({"electronic","aggressive","quiet"})),
    "The Rolling Stones":  (None,        frozenset({"electronic"})),
    "Fleetwood Mac":       (None,        frozenset({"aggressive","electronic"})),
    "Joni Mitchell":       ("quiet",     frozenset({"electronic","happy","aggressive"})),
    "Paul Simon":          ("happy",     frozenset({"electronic","aggressive"})),
    "Simon & Garfunkel":   ("quiet",     frozenset({"electronic","happy","aggressive"})),
    "Bruce Springsteen":   (None,        frozenset({"electronic"})),
    "Van Morrison":        (None,        frozenset({"electronic","aggressive"})),
    "James Taylor":        ("quiet",     frozenset({"electronic","happy","aggressive"})),
    "Cat Stevens":         ("quiet",     frozenset({"electronic","happy","aggressive"})),
    "Nick Drake":          ("quiet",     frozenset({"electronic","happy","aggressive"})),
    "Elliott Smith":       ("quiet",     frozenset({"electronic","happy","aggressive"})),
    "Townes Van Zandt":    ("quiet",     frozenset({"electronic","happy","aggressive"})),
    "John Martyn":         ("quiet",     frozenset({"electronic","happy","aggressive"})),
    "Bob Marley":          ("happy",     frozenset({"electronic","aggressive","quiet"})),
    "Bob Marley & The Wailers":("happy", frozenset({"electronic","aggressive","quiet"})),
}

ALBUM_QUIET = {
    "nirvana":         {"mtv unplugged in new york"},
    "alice in chains": {"unplugged","jar of flies","sap"},
}
ALBUM_ELECTRONIC = {
    "john frusciante":      {"pbx funicular intaglio zone","enclosure","letur lefr",
                             "a sphere in the heart of silence","dc ep"},
    "the smashing pumpkins":{"adore"},
    "smashing pumpkins":    {"adore"},
}
TRACK_OVERRIDES = {
    "Nirvana|||Dumb": "quiet",
}

# ── Top-15 genre groups (tag → group label) ──────────────────────────────────
TAG_TO_GENRE = {}
for _tags, _label in [
    (["classic rock","rock","album rock","arena rock","heartland rock"], "Classic Rock"),
    (["grunge","seattle sound","post-grunge"],                           "Grunge"),
    (["alt-country","americana","country rock","outlaw country","roots rock","southern rock"], "Alt-Country"),
    (["folk","folk rock","indie folk","contemporary folk","singer-songwriter","acoustic"], "Folk"),
    (["experimental","avant-garde","noise","art rock","krautrock","progressive rock","prog rock"], "Experimental"),
    (["post-punk","new wave","coldwave","darkwave","minimal wave"],      "Post-Punk"),
    (["blues rock","blues","chicago blues","delta blues","electric blues"], "Blues Rock"),
    (["hard rock","heavy metal","metal","doom metal","stoner rock","glam metal"], "Hard Rock"),
    (["indie pop","indie rock","alternative","lo-fi","chamber pop"],     "Indie Pop"),
    (["garage rock","punk","punk rock","power pop","proto-punk"],        "Garage Rock"),
    (["glam rock","glam","t. rex","bowie","art pop"],                   "Glam Rock"),
    (["new wave","synth-pop","synthpop","post-punk","electronic rock"],  "New Wave"),
    (["psychedelic rock","psychedelic","acid rock","space rock"],        "Psychedelic Rock"),
    (["electronic","ambient","idm","techno","house","synth","trip-hop","downtempo","electronica"], "Electronic"),
    (["britpop","uk indie","british rock","shoegaze","madchester"],      "Britpop"),
]:
    for _t in _tags:
        TAG_TO_GENRE[_t] = _label


def _classify_pocket(artist: str, track: str, albums: list,
                     ab: dict, msd: dict, all_tags: list) -> str:
    """Three-stage pocket classifier. Returns one of:
    quiet | aggressive | electronic | happy | blob
    """
    tag_set = {t.lower().strip() for t in all_tags if isinstance(t, str)}

    # ── Stage 3a: track-level hard override ──────────────────────────────────
    tk = f"{artist}|||{track}"
    if tk in TRACK_OVERRIDES:
        return TRACK_OVERRIDES[tk]

    # ── Stage 3b: album-level override ───────────────────────────────────────
    artist_lo = artist.lower()
    for alb in (albums or []):
        alb_lo = alb.lower()
        if artist_lo in ALBUM_QUIET and alb_lo in ALBUM_QUIET[artist_lo]:
            return "quiet"
        if artist_lo in ALBUM_ELECTRONIC and alb_lo in ALBUM_ELECTRONIC[artist_lo]:
            return "electronic"

    # ── Stage 1: AB acoustic thresholds ──────────────────────────────────────
    if ab:
        happy  = ab.get("mood_happy", 0) or 0
        sad    = ab.get("mood_sad", 0) or 0
        agg    = ab.get("mood_aggressive", 0) or 0
        relax  = ab.get("mood_relaxed", 0) or 0
        acou   = ab.get("mood_acoustic", 0) or 0
        elect  = ab.get("mood_electronic", 0) or 0
        dance  = ab.get("danceability", 0) or 0

        votes = {}
        if acou >= AB_QUIET_THRESH:
            votes["quiet"] = acou
        if agg >= AB_AGGRESSIVE_THRESH and acou < 0.50:
            votes["aggressive"] = agg
        if elect >= AB_ELECTRONIC_THRESH:
            votes["electronic"] = elect
        if happy >= AB_HAPPY_THRESH and dance >= AB_DANCE_THRESH:
            votes["happy"] = happy

        if votes:
            # Remove pockets vetoed by co-present tags
            for pocket, veto_tags in VETO.items():
                if pocket in votes and tag_set & veto_tags:
                    del votes[pocket]

            # Artist-level block
            ao = AO.get(artist, AO.get(f"The {artist}"))
            if ao:
                _, blocked = ao
                for b in blocked:
                    votes.pop(b, None)

            if votes:
                return max(votes, key=votes.get)

    # ── Stage 2: tag-based voting ─────────────────────────────────────────────
    scores = {"quiet": 0, "aggressive": 0, "electronic": 0, "happy": 0}

    has_synth = bool(tag_set & SYNTH_TAGS)

    for t in tag_set:
        if t in QUIET_TAGS:      scores["quiet"]      += 1
        if t in AGGRESSIVE_TAGS: scores["aggressive"] += 1
        if t in HAPPY_TAGS:      scores["happy"]      += 1
        if t in ELECTRONIC_TAGS:
            if t in ELECTRONIC_COPRESENCE_GUARD:
                if has_synth: scores["electronic"] += 1
            else:
                scores["electronic"] += 1

    # Apply veto rules
    for pocket, veto_tags in VETO.items():
        if tag_set & veto_tags:
            scores[pocket] = 0

    # Artist-level force / block
    ao = AO.get(artist, AO.get(f"The {artist}"))
    if ao:
        force, blocked = ao
        for b in blocked:
            scores[b] = 0
        if force and scores.get(force, 0) == 0:
            scores[force] = 0.5   # soft force only if nothing else wins

    best_score = max(scores.values())
    if best_score > 0:
        winner = max(scores, key=scores.get)
        # artist force wins ties
        if ao and ao[0] and scores.get(ao[0], 0) == best_score:
            return ao[0]
        return winner

    # Artist force with no tag evidence
    if ao and ao[0]:
        return ao[0]

    return "blob"


def _map_genre(all_tags: list) -> str:
    """Map tag list to top-15 genre label. Returns 'Other' if no match."""
    for t in all_tags:
        tl = t.lower().strip() if isinstance(t, str) else ""
        if tl in TAG_TO_GENRE:
            return TAG_TO_GENRE[tl]
    return "Other"


def _extract_decade(ab: dict, msd: dict) -> str:
    """Extract decade string (e.g. '1990s') from MSD year. Returns '' if unknown."""
    year = None
    if msd:
        y = msd.get("msd_year")
        if y and int(y) > 1900:
            year = int(y)
    if year:
        return f"{(year // 10) * 10}s"
    return ""


def _flatten_tags(raw) -> list:
    """Flatten lastfm_tags (list of [tag, count] pairs) and other tag lists."""
    out = []
    if not raw:
        return out
    if isinstance(raw, str):
        try: raw = json.loads(raw)
        except Exception: return [raw]
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                out.append(str(item[0]))
            elif isinstance(item, str):
                out.append(item)
    return out


def build_track_index(con, username: str, min_plays: int = 3):
    """Phase 7 — Run pocket classifier on all user tracks (≥min_plays) and
    write {username}_track_index.json to ~/.music-agent/ and project dir."""
    log(f"Phase 7 — Building {username}_track_index.json …")

    rows = con.execute("""
        SELECT t.track_key, t.artist, t.title, up.plays,
               t.lastfm_tags, t.mb_rec_genres, t.mb_rec_tags,
               t.mb_artist_genres, t.mb_artist_tags,
               t.discogs_styles,
               t.ab, t.msd, up.albums
        FROM tracks t
        JOIN user_profiles up ON t.track_key = up.track_key
        WHERE up.username = ? AND up.plays >= ?
        ORDER BY up.plays DESC
    """, (username, min_plays)).fetchall()

    index = {}
    pocket_counts = defaultdict(int)

    for row in rows:
        (track_key, artist, title, plays,
         lfm_raw, mb_rec_genres_raw, mb_rec_tags_raw,
         mb_art_genres_raw, mb_art_tags_raw,
         disc_styles_raw,
         ab_raw, msd_raw, albums_raw) = row

        # Parse AB / MSD
        ab  = json.loads(ab_raw)  if ab_raw  else {}
        msd = json.loads(msd_raw) if msd_raw else {}

        # Build ordered tag list (priority: mb_rec > lastfm > discogs_styles > mb_artist)
        lfm_tags    = _flatten_tags(json.loads(lfm_raw)            if lfm_raw else [])
        rec_tags    = _flatten_tags(json.loads(mb_rec_genres_raw)  if mb_rec_genres_raw else []
                                  + json.loads(mb_rec_tags_raw)    if mb_rec_tags_raw else [])
        art_tags    = _flatten_tags(json.loads(mb_art_genres_raw)  if mb_art_genres_raw else []
                                  + json.loads(mb_art_tags_raw)    if mb_art_tags_raw else [])
        disc_styles = _flatten_tags(json.loads(disc_styles_raw)    if disc_styles_raw else [])

        all_tags = rec_tags + lfm_tags + disc_styles + art_tags

        albums = json.loads(albums_raw) if albums_raw else []
        if isinstance(albums, str):
            try: albums = json.loads(albums)
            except Exception: albums = [albums]

        pocket = _classify_pocket(artist, title, albums, ab, msd, all_tags)
        genre  = _map_genre(all_tags)
        decade = _extract_decade(ab, msd)

        index[track_key] = {
            "artist": artist,
            "track":  title,
            "plays":  plays,
            "pocket": pocket,
            "genre":  genre,
            "decade": decade,
        }
        pocket_counts[pocket] += 1

    # Write to ~/.music-agent/
    out_path = AGENT_DIR / f"{username}_track_index.json"
    out_path.write_text(json.dumps(index))

    # Also copy to project dir for Streamlit Cloud deployment
    project_path = PROJECT_DIR / f"{username}_track_index.json"
    try:
        import shutil
        shutil.copy(out_path, project_path)
        log(f"    Copied to project dir: {project_path.name}")
    except Exception as e:
        log(f"    Warning: could not copy to project dir — {e}")

    log(f"    {len(index):,} tracks indexed for {username}")
    log(f"    Pocket distribution: " +
        " | ".join(f"{p}={n}" for p, n in sorted(pocket_counts.items())))

    return index


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def load_secrets():
    api_key = discogs_token = ""
    try:
        text = SECRETS.read_text()
        m = re.search(r'LASTFM_API_KEY\s*=\s*["\'](.+?)["\']', text)
        if m: api_key = m.group(1)
        m = re.search(r'DISCOGS_TOKEN\s*=\s*["\'](.+?)["\']', text)
        if m: discogs_token = m.group(1)
    except Exception as e:
        log(f"Warning: could not read secrets — {e}")
    return api_key, discogs_token

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username",          help="Contributor username")
    parser.add_argument("--data-dir",          help="Path to Spotify Extended Streaming History folder")
    parser.add_argument("--migrate-marlonrando", action="store_true")
    parser.add_argument("--min-plays",         type=int, default=3)
    parser.add_argument("--min-cluster-size",  type=int, default=160)
    parser.add_argument("--skip-enrichment",   action="store_true")
    parser.add_argument("--skip-cluster",      action="store_true")
    parser.add_argument("--block-artists",     nargs="+", default=[],
                        metavar="ARTIST",
                        help="Artists to exclude from the corpus (case-insensitive)")
    parser.add_argument("--start-from",        default="3a",
                        choices=["1","2","3a","3b","3c","3d","3e","3f","3g","3g5","3h","4","5","6","7"],
                        help="Resume pipeline from this phase (skip earlier phases)")
    parser.add_argument("--skip-index",        action="store_true",
                        help="Skip Phase 7 track index build")
    parser.add_argument("--index-only",        action="store_true",
                        help="Only run Phase 7 (rebuild track index, skip all enrichment)")
    parser.add_argument("--run-msd-imputation",action="store_true",
                        help="Run Phase 3g.5 MSD acoustic imputation against full DB, "
                             "then rebuild track index (use for backfill after pkl is ready)")
    args = parser.parse_args()

    t0 = time.time()
    api_key, discogs_token = load_secrets()

    log("═"*60)
    log("  Music Taste Graph — Contributor Ingest Pipeline")
    log("═"*60)

    # Phase 0 — init DB
    con = init_db()
    existing_count = con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]

    if args.migrate_marlonrando or existing_count == 0:
        if MARLONRANDO_META.exists():
            migrate_marlonrando(con)
        else:
            log("marlonrando_track_metadata.json not found — starting fresh DB")

    if args.username and args.data_dir:
        data_dir = Path(os.path.expanduser(args.data_dir))

        # Phase 1
        block_set = {a.lower() for a in args.block_artists}
        profile = parse_spotify_export(data_dir, args.username, args.min_plays,
                                       block_artists=block_set)

        # Phase 2
        net_new = find_net_new(con, profile)

        # phase ordering for --start-from
        PHASE_ORDER = ["3a","3b","3c","3d","3e","3f","3g","3g5","3h","4","5","6"]
        def should_run(phase):
            try:
                return PHASE_ORDER.index(phase) >= PHASE_ORDER.index(args.start_from)
            except ValueError:
                return True

        if net_new and not args.skip_enrichment:
            # resolve track MBIDs from Last.fm getSimilar where available
            graph_path = AGENT_DIR / "similarity_graph.json"
            if graph_path.exists():
                graph = json.loads(graph_path.read_text())
                for entry in graph.values():
                    for s in entry.get("similar", []):
                        sk = f"{s['artist']}|||{s['title']}"
                        if sk in net_new and s.get("mbid"):
                            net_new[sk]["track_mbid"] = s["mbid"]

            # Phase 3a–3h (skippable via --start-from)
            if should_run("3a"): enrich_lastfm_tags(net_new, api_key)
            if should_run("3b"): enrich_similarity_graph(profile, net_new, api_key)
            if should_run("3c"): enrich_mb_artist(net_new)
            if should_run("3d"): enrich_mb_recording(net_new)
            if should_run("3e"): enrich_discogs(net_new, discogs_token)
            if should_run("3f"): enrich_acousticbrainz(net_new)
            if should_run("3g"):  enrich_msd(net_new)
            if should_run("3g5"): enrich_msd_imputation(net_new, con)
            if should_run("3h"):  enrich_label_propagation(net_new, con)

        # Phase 4
        if should_run("4"):
            write_to_universe(con, net_new, profile, args.username)

    # ── MSD imputation backfill (standalone mode) ──────────────────────────
    if args.run_msd_imputation and args.username:
        log(f"MSD Imputation backfill for {args.username} …")
        # Pull all DB tracks that have MSD but no AB into a temporary dict
        backfill = {}
        for row in con.execute(
                "SELECT track_key, artist, title, msd, ab FROM tracks "
                "WHERE msd IS NOT NULL AND ab IS NULL"):
            backfill[row[0]] = {
                "artist": row[1], "track": row[2],
                "msd": json.loads(row[3]), "ab": None
            }
        if backfill:
            enrich_msd_imputation(backfill, con)
            # Write imputed AB back to DB
            updated = [(json.dumps(v["ab"]), k)
                       for k, v in backfill.items() if v.get("ab")]
            con.executemany("UPDATE tracks SET ab=? WHERE track_key=?", updated)
            con.commit()
            log(f"    Wrote {len(updated):,} imputed AB records to DB")

    if not args.skip_cluster and not args.index_only:
        # Phase 5
        best_k = recluster(con, args.min_cluster_size)
        # Phase 6
        compute_affinity_scores(con)

    # ── Phase 7: pocket classifier + track index ───────────────────────────
    if args.username and not args.skip_index:
        build_track_index(con, args.username, args.min_plays)

    elapsed = (time.time() - t0) / 60
    total_tracks = con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    total_users  = con.execute("SELECT COUNT(DISTINCT username) FROM user_profiles").fetchone()[0]

    log("═"*60)
    log(f"  COMPLETE  ({elapsed:.1f} min)")
    log(f"  track_universe.db : {total_tracks:,} tracks  {total_users} users")
    log(f"  DB path           : {DB_PATH}")
    log("═"*60)
    con.close()

if __name__ == "__main__":
    main()
