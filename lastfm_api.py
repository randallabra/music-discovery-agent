"""
Last.fm API helpers for the Music Discovery Agent.

Uses track.getTopTags (with artist.getTopTags as fallback) to fetch
crowd-sourced genre/mood tags at the individual song level.
No OAuth required — read-only calls with API key only.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"
_RATE_LIMIT_DELAY = 0.22   # ~4.5 req/s, safely under Last.fm's 5 req/s limit
_MAX_WORKERS     = 4       # concurrent requests — stays safely under rate limit

# ─────────────────────────────────────────────────────────────
#  Persistent tag cache
#  Saves fetched tags to disk so repeat runs skip the API entirely.
# ─────────────────────────────────────────────────────────────

_CACHE_FILE    = Path.home() / ".music-agent" / "lastfm_tag_cache.json"
_CACHE_VERSION = 2   # bump to invalidate old cache formats


def _load_tag_cache() -> dict:
    try:
        raw = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and raw.get("_v") == _CACHE_VERSION:
            return raw
    except Exception:
        pass
    return {"_v": _CACHE_VERSION}


def _save_tag_cache(cache: dict) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


def _track_cache_key(artist: str, track: str) -> str:
    return f"t:{artist.lower().strip()}|||{track.lower().strip()}"


def _artist_cache_key(artist: str) -> str:
    return f"a:{artist.lower().strip()}"


# ─────────────────────────────────────────────────────────────
#  Rate limiter — serialises the START of each request across
#  threads so we never exceed ~4.5 req/s globally.
# ─────────────────────────────────────────────────────────────

_rate_lock      = threading.Lock()
_last_call_time: list[float] = [0.0]


def _rate_limited_fetch(params: dict) -> list[tuple[str, int]]:
    with _rate_lock:
        now  = time.time()
        wait = _RATE_LIMIT_DELAY - (now - _last_call_time[0])
        if wait > 0:
            time.sleep(wait)
        _last_call_time[0] = time.time()
    return _fetch_tags(params)


# ─────────────────────────────────────────────────────────────
#  Lane → Last.fm tag sets
#  Includes both compound forms ("alternative rock") AND the plain
#  single-word forms ("alternative", "rock") that Last.fm users
#  actually apply most often.
# ─────────────────────────────────────────────────────────────

LANE_TAGS: dict[str, set[str]] = {
    # ── Empirically grounded from 500-track Last.fm sample ──────────────────
    # Broad tags (rock, alternative, indie, alternative rock) intentionally
    # excluded — they appear across every rock lane and carry no discriminating
    # signal. Each lane uses only tags that are specific to it.

    "Melancholy Balladry": {
        "melancholic", "melancholy", "sad", "mellow", "piano", "ballad",
        "slowcore", "lullaby", "tearjerker", "heartbreak", "grief", "dirge",
        "soft", "gentle", "quiet",
    },
    "Introspective Songcraft": {
        "singer-songwriter", "folk", "folk rock", "americana", "alt-country",
        "country", "heartland rock", "indie folk", "folk pop", "psychedelic folk",
        "country rock", "freak folk", "chamber folk", "storytelling",
        "confessional", "acoustic",
    },
    "Hook-Perfected Songs": {
        "power pop", "bubblegum", "new wave", "synth-pop", "pop rock",
        "motown", "girl group", "catchy", "hook", "art pop",
    },
    "Atmospheric / Texture-First": {
        "ambient", "drone", "krautrock", "soundscape", "avant-garde",
        "neoclassical", "modern classical", "microsound", "ambient pop",
        "electronic", "electronica", "trip-hop",
    },
    "Propulsive Guitar Rock": {
        "garage rock", "art punk", "post-punk revival", "jangle pop",
        "post-punk", "college rock", "noise rock", "crank wave",
        "gothic rock", "new wave",
    },
    "Hybrid Rock (Melody-Led)": {
        "britpop", "baroque pop", "art rock", "chamber pop",
        "canadian", "indie pop", "chamber music",
    },
    "Groove-First Rock (Selective)": {
        "funk rock", "funk", "california", "southern rock", "rock and roll",
        "british invasion", "progressive rock", "boogie", "blues rock",
        "blues", "jam band", "swamp rock", "60s", "70s",
    },
    "Energy-First Rock (Melody-Bounded)": {
        "hard rock", "grunge", "seattle", "stoner rock", "stoner",
        "alternative metal", "desert rock", "post-grunge", "supergroup",
        "seattle sound", "robot rock", "classic rock",
    },
    "Rhythm-Dominant Rock (Cathartic)": {
        "hardcore", "post-hardcore", "math rock", "screamo", "metalcore",
        "noise rock", "heavy", "aggressive",
    },
    "Texture-First Rock (Atmospheric)": {
        "shoegaze", "dream pop", "noise pop", "post-rock", "ethereal",
        "wall of sound", "space rock", "reverb", "lo-fi",
    },
    "Cathartic / Adrenaline Rock": {
        "thrash metal", "death metal", "black metal", "grindcore",
        "sludge metal", "doom metal", "heavy metal", "metal",
    },

    # ── Jazz + Hip-Hop: seed tags retained (sparse in this user's sample) ───
    "Melody-Anchored Hip-Hop": {
        "alternative hip hop", "conscious hip hop", "jazz rap",
        "hip hop soul", "neo soul", "hip hop", "hip-hop",
    },
    "Groove-First Hip-Hop (Selective)": {
        "g-funk", "bounce", "gangsta rap", "boom bap", "dirty south",
        "hip hop", "hip-hop", "rap",
    },
    "Textural / Atmospheric Hip-Hop": {
        "cloud rap", "chillhop", "instrumental hip hop", "lo-fi hip hop",
        "hip hop", "hip-hop", "trap",
    },
    "Melody-Led Jazz (Theme-Centric)": {
        "bebop", "cool jazz", "hard bop", "vocal jazz", "standards",
        "big band", "jazz", "swing",
    },
    "Harmonic-First Jazz": {
        "post-bop", "modal jazz", "fusion", "contemporary jazz", "jazz",
    },
    "Spiritual / Expansive Jazz": {
        "free jazz", "avant-garde jazz", "spiritual jazz", "jazz",
    },
    "Rhythm-Forward Jazz (Selective)": {
        "latin jazz", "afrobeat", "bossa nova", "samba", "jazz funk", "jazz",
    },
}


# ─────────────────────────────────────────────────────────────
#  API calls
# ─────────────────────────────────────────────────────────────

def _fetch_tags(params: dict) -> list[tuple[str, int]]:
    """Shared fetch logic — returns [(tag_lower, count), ...] or []."""
    params["format"] = "json"
    url = LASTFM_BASE + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        tags_raw = data.get("toptags", {}).get("tag", [])
        if isinstance(tags_raw, dict):   # single tag edge case
            tags_raw = [tags_raw]
        return [(t["name"].lower().strip(), int(t["count"])) for t in tags_raw if int(t["count"]) > 0]
    except Exception:
        return []


def get_track_tags(artist: str, track: str, api_key: str) -> list[tuple[str, int]]:
    return _fetch_tags({
        "method":      "track.getTopTags",
        "artist":      artist,
        "track":       track,
        "api_key":     api_key,
        "autocorrect": "1",
    })


def get_artist_tags(artist: str, api_key: str) -> list[tuple[str, int]]:
    return _fetch_tags({
        "method":      "artist.getTopTags",
        "artist":      artist,
        "api_key":     api_key,
        "autocorrect": "1",
    })


# ─────────────────────────────────────────────────────────────
#  Lane fit scoring
# ─────────────────────────────────────────────────────────────

def lane_fit_score(tags: list[tuple[str, int]], lane: str) -> float:
    """
    Score a tag list against a lane's tag set.
    Returns 0.0–1.0 (weighted overlap over top 15 tags).
    """
    target = LANE_TAGS.get(lane, set())
    if not target or not tags:
        return 0.0
    top = tags[:15]
    total_weight = sum(count for _, count in top)
    if total_weight == 0:
        return 0.0
    matched_weight = sum(count for tag, count in top if tag in target)
    return matched_weight / total_weight


def lane_fit_label(score: float) -> str:
    if score >= 0.15:
        return "High"
    if score >= 0.04:
        return "Medium"
    return ""


# ─────────────────────────────────────────────────────────────
#  Batch fetch for anchor pool
# ─────────────────────────────────────────────────────────────

def fetch_pool_lane_fits(
    tracks: list[dict],
    lane: str,
    api_key: str,
    progress_callback=None,
) -> list[dict]:
    """
    Fetch Last.fm tags for each anchor track and score against the lane.
    Falls back to artist.getTopTags when track tags are sparse (< 3 tags).
    Returns enriched track list with 'lane_fit' and 'lane_score' keys,
    sorted High → Medium → blank, sub-sorted by plays descending.

    Performance:
    - Persistent disk cache (~/.music-agent/lastfm_tag_cache.json) —
      cached tracks skip the API entirely; repeat runs are near-instant.
    - Up to 4 parallel requests with a shared rate limiter so total
      throughput stays safely under Last.fm's 5 req/s limit.
    """
    disk_cache:       dict = _load_tag_cache()
    artist_mem_cache: dict = {}          # in-memory artist tag cache this session
    cache_lock    = threading.Lock()
    counter: list[int] = [0]
    total = len(tracks)

    def _fetch_one(t: dict) -> dict:
        tk = _track_cache_key(t["artist"], t["track"])

        # ── 1. Disk cache hit — no API call needed ────────────────────────
        with cache_lock:
            cached = disk_cache.get(tk)
        if cached is not None:
            tags = cached
        else:
            # ── 2. Fetch track-level tags ─────────────────────────────────
            tags = _rate_limited_fetch({
                "method":      "track.getTopTags",
                "artist":      t["artist"],
                "track":       t["track"],
                "api_key":     api_key,
                "autocorrect": "1",
            })

            # ── 3. Fallback to artist tags if sparse ──────────────────────
            if len(tags) < 3:
                ak = _artist_cache_key(t["artist"])
                with cache_lock:
                    artist_tags = artist_mem_cache.get(ak) or disk_cache.get(ak)
                if artist_tags is None:
                    artist_tags = _rate_limited_fetch({
                        "method":      "artist.getTopTags",
                        "artist":      t["artist"],
                        "api_key":     api_key,
                        "autocorrect": "1",
                    })
                    with cache_lock:
                        artist_mem_cache[ak] = artist_tags
                        disk_cache[ak]       = artist_tags
                if len(artist_tags) > len(tags):
                    tags = artist_tags

            # ── 4. Persist to disk cache ──────────────────────────────────
            with cache_lock:
                disk_cache[tk] = tags

        score = lane_fit_score(tags, lane)

        # Thread-safe progress update
        with cache_lock:
            counter[0] += 1
            n = counter[0]
        if progress_callback:
            progress_callback(n, total)

        return {**t, "lane_fit": lane_fit_label(score), "lane_score": score}

    # ── Parallel execution ────────────────────────────────────────────────────
    enriched: list[dict] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = [executor.submit(_fetch_one, t) for t in tracks]
        for future in as_completed(futures):
            try:
                enriched.append(future.result())
            except Exception:
                pass  # skip a failed track rather than crashing the whole pool

    _save_tag_cache(disk_cache)

    order = {"High": 0, "Medium": 1, "": 2}
    enriched.sort(key=lambda r: (order[r["lane_fit"]], -r["plays"]))
    return enriched
