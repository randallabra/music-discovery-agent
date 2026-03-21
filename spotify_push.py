"""
Spotify Web API helpers — playlist creation from recommendations.

Uses Authorization Code Flow (via Spotipy) to authenticate a user and
create a playlist from the agent's output.  No credentials are ever
hard-coded or cached between sessions — MemoryCacheHandler ensures
every connection requires fresh user authentication.
"""
from __future__ import annotations

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import MemoryCacheHandler

SCOPE = "playlist-modify-public playlist-modify-private"


# ─────────────────────────────────────────────────────────────
#  OAuth helpers
# ─────────────────────────────────────────────────────────────

def make_oauth(client_id: str, client_secret: str, redirect_uri: str) -> SpotifyOAuth:
    """Build a SpotifyOAuth object with no on-disk token cache."""
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPE,
        cache_handler=MemoryCacheHandler(),
        show_dialog=True,   # always prompt — never silently reuse a prior session
    )


def get_auth_url(oauth: SpotifyOAuth, state: str = "") -> str:
    """Return the Spotify authorization URL. Encodes `state` for CSRF / session recovery."""
    return oauth.get_authorize_url(state=state)


def exchange_code(oauth: SpotifyOAuth, code: str) -> dict:
    """
    Exchange an authorization code for a token dict.
    Returns a dict containing at minimum 'access_token'.
    """
    return oauth.get_access_token(code, check_cache=False)


# ─────────────────────────────────────────────────────────────
#  Spotify client
# ─────────────────────────────────────────────────────────────

def make_client(access_token: str) -> spotipy.Spotify:
    return spotipy.Spotify(auth=access_token)


def get_current_user(sp: spotipy.Spotify) -> dict:
    return sp.current_user()


# ─────────────────────────────────────────────────────────────
#  Track search
# ─────────────────────────────────────────────────────────────

def search_track(sp: spotipy.Spotify, artist: str, track: str) -> str | None:
    """
    Search Spotify for an artist+track combination.
    Returns a Spotify URI string on success, or None if not found.
    Two-pass: strict field query first, loose fallback second.
    """
    # Pass 1 — field-qualified search
    results = sp.search(q=f"artist:{artist} track:{track}", type="track", limit=1)
    items = results.get("tracks", {}).get("items", [])
    if items:
        return items[0]["uri"]

    # Pass 2 — plain text fallback (handles punctuation / article mismatches)
    results2 = sp.search(q=f"{artist} {track}", type="track", limit=1)
    items2 = results2.get("tracks", {}).get("items", [])
    if items2:
        return items2[0]["uri"]

    return None


# ─────────────────────────────────────────────────────────────
#  Playlist creation
# ─────────────────────────────────────────────────────────────

def push_playlist(
    sp: spotipy.Spotify,
    user_id: str,
    name: str,
    description: str,
    recs: list,
    public: bool = True,
    progress_callback=None,
) -> dict:
    """
    Create a Spotify playlist and add recommendation tracks.

    recs: list of Recommendation objects (must have .artist and .track attributes).
    progress_callback: optional callable(current_index, total) for UI progress updates.

    Returns:
        {
            "playlist_url":  str,
            "playlist_name": str,
            "found":         list[str],   # "Artist — Track" for matched tracks
            "not_found":     list[str],   # "Artist — Track" for unmatched tracks
        }
    """
    playlist = sp.user_playlist_create(
        user=user_id,
        name=name,
        public=public,
        description=description,
    )
    playlist_url = playlist["external_urls"]["spotify"]

    found_uris: list[str] = []
    found: list[str] = []
    not_found: list[str] = []

    total = len(recs)
    for i, r in enumerate(recs):
        uri = search_track(sp, r.artist, r.track)
        if uri:
            found_uris.append(uri)
            found.append(f"{r.artist} — {r.track}")
        else:
            not_found.append(f"{r.artist} — {r.track}")
        if progress_callback:
            progress_callback(i + 1, total)

    # Spotify limit: 100 tracks per add call
    for i in range(0, len(found_uris), 100):
        sp.playlist_add_items(playlist["id"], found_uris[i : i + 100])

    return {
        "playlist_url": playlist_url,
        "playlist_name": name,
        "found": found,
        "not_found": not_found,
    }
