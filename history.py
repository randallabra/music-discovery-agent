from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Artists whose names suggest utility/sleep/ambient listening — purged automatically
UTILITY_KEYWORDS = [
    "white noise", "sleep", "rain sound", "brown noise", "pink noise",
    "binaural", "focus music", "study music", "meditation", "lofi beats",
    "lo-fi beats", "ambient noise", "baby sleep", "sleep aid",
    "unknown", "untitled",
]


@dataclass
class ArtistStats:
    name: str
    total_plays: int
    unique_tracks: int
    top_tracks: list[tuple[str, int]]  # [(track_name, plays), ...] descending


@dataclass
class AnchorPool:
    tracks: list[dict]           # [{"artist": ..., "track": ..., "plays": ...}]
    purged_artists: list[str]    # artists that failed purge
    eligible_count: int          # artists that passed purge
    total_scrobbles: int
    total_artists: int
    known_tracks: frozenset      # ALL (artist_lower, track_lower) from full history — used as post-filter


class AnchorPoolTooSmallError(Exception):
    pass


def _row_to_artist_track(row: dict, fmt: str) -> tuple[str, str]:
    """Extract (artist, track) from a row, normalized to strip whitespace."""
    if fmt == "lastfm":
        return (row.get("artist") or "").strip(), (row.get("track") or "").strip()
    if fmt == "spotify_basic":
        return (row.get("artistName") or "").strip(), (row.get("trackName") or "").strip()
    if fmt == "spotify_extended":
        # Spotify extended: filter out rows with very low ms_played (skips, < 30 seconds)
        ms = int(row.get("ms_played") or 0)
        if ms < 30_000:
            return "", ""  # treat as skipped
        artist = (row.get("master_metadata_album_artist_name") or "").strip()
        track = (row.get("master_metadata_track_name") or "").strip()
        return artist, track
    return "", ""


# ---------- Parsing ----------

def _resolve_spotify_subformat(fieldnames: list[str]) -> str:
    """
    Spotify exports come in two layouts. Determine which one silently
    so the user never has to know about the distinction.
    """
    h = {c.lower().strip() for c in fieldnames}
    if "master_metadata_track_name" in h:
        return "spotify_extended"
    return "spotify_basic"


def parse_history_csv(csv_path: Path, source: str) -> tuple[dict[str, ArtistStats], int, str]:
    """
    Parse a Last.fm or Spotify CSV export.

    source: 'lastfm' or 'spotify' (user-declared, not inferred from column names)

    Returns (artist_stats_dict, total_scrobbles, resolved_format_label).
    """
    plays: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = 0
    skipped = 0

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file appears to be empty: {csv_path}")

        if source == "lastfm":
            fmt = "lastfm"
        else:
            fmt = _resolve_spotify_subformat(list(reader.fieldnames))

        for row in reader:
            artist, track = _row_to_artist_track(row, fmt)
            if not artist or not track:
                skipped += 1
                continue
            plays[artist][track] += 1
            total += 1

    if skipped and skipped > total * 0.1:
        print(f"  Note: skipped {skipped:,} rows (missing artist/track or short Spotify plays)")

    stats: dict[str, ArtistStats] = {}
    for artist, track_map in plays.items():
        top = sorted(track_map.items(), key=lambda x: x[1], reverse=True)
        stats[artist] = ArtistStats(
            name=artist,
            total_plays=sum(track_map.values()),
            unique_tracks=len(track_map),
            top_tracks=top,
        )

    return stats, total, fmt


# ---------- Purge ----------

def _is_utility(artist_name: str) -> bool:
    lower = artist_name.lower()
    return any(kw in lower for kw in UTILITY_KEYWORDS)


def purge_artists(
    stats: dict[str, ArtistStats],
    max_artist_plays: int,
    max_unique_tracks: int,
    blacklist: set[str],  # normalized (lower) names
) -> tuple[dict[str, ArtistStats], list[str]]:
    """
    Returns (eligible, purged_names).
    Purge criteria (any one is sufficient):
      - total_plays >= max_artist_plays   (you know this artist cold)
      - unique_tracks >= max_unique_tracks (broad utility listening, not a taste signal)
      - utility keywords in artist name
      - artist in explicit blacklist
    """
    eligible: dict[str, ArtistStats] = {}
    purged: list[str] = []

    for name, s in stats.items():
        if (
            s.total_plays >= max_artist_plays
            or s.unique_tracks >= max_unique_tracks
            or _is_utility(name)
            or name.strip().lower() in blacklist
        ):
            purged.append(name)
        else:
            eligible[name] = s

    return eligible, sorted(purged)


# ---------- Anchor Pool ----------

def build_anchor_pool(
    eligible: dict[str, ArtistStats],
    top_tracks_per_artist: int,
    anchor_pool_size: int,
    collision_memory_set: set[tuple[str, str]],
    min_track_plays: int = 8,
    oversize_factor: int = 2,
    freshness_penalties: dict[str, float] | None = None,
) -> list[dict]:
    """
    Select anchor tracks from most-played eligible artists.
    Builds at oversize_factor * anchor_pool_size to allow lane-fit
    enrichment to sort and truncate to the final target size.
    Only includes tracks played at least min_track_plays times.

    freshness_penalties: {artist_lower: weight_factor} from ProjectState.
    Artists that appeared in recent anchor pools have their effective play
    count reduced, making room for lower-played but fresher artists.

    Returns list of {"artist": ..., "track": ..., "plays": ...}.
    """
    penalties = freshness_penalties or {}
    build_limit = anchor_pool_size * oversize_factor

    def _sort_key(s: ArtistStats) -> float:
        factor = penalties.get(s.name.strip().lower(), 1.0)
        return s.total_plays * factor

    sorted_artists = sorted(eligible.values(), key=_sort_key, reverse=True)

    anchors: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for s in sorted_artists:
        if len(anchors) >= build_limit:
            break
        added = 0
        for track, plays in s.top_tracks:
            if added >= top_tracks_per_artist or len(anchors) >= build_limit:
                break
            if plays < min_track_plays:
                continue
            key = (s.name.strip().lower(), track.strip().lower())
            if key not in collision_memory_set and key not in seen:
                anchors.append({"artist": s.name, "track": track, "plays": plays})
                seen.add(key)
                added += 1

    return anchors


# ---------- Main entry point ----------

def process_history(
    csv_path: Path,
    source: str,             # 'lastfm' or 'spotify' — user-declared
    max_artist_plays: int,
    max_unique_tracks: int,
    top_tracks_per_artist: int,
    anchor_pool_size: int,
    blacklist: set[str],
    collision_memory: list[dict],
    min_track_plays: int = 8,
    freshness_penalties: dict[str, float] | None = None,
    verbose: bool = False,
) -> AnchorPool:
    print(f"  Parsing {csv_path.name}...")
    stats, total_scrobbles, fmt = parse_history_csv(csv_path, source)
    fmt_label = {"lastfm": "Last.fm", "spotify_basic": "Spotify", "spotify_extended": "Spotify Extended"}.get(fmt, fmt)
    print(f"  Source: {fmt_label}  |  {total_scrobbles:,} plays  |  {len(stats):,} unique artists")

    eligible, purged = purge_artists(stats, max_artist_plays, max_unique_tracks, blacklist)
    print(f"  Purged {len(purged):,} saturated/utility/blacklisted artists")
    print(f"  {len(eligible):,} artists eligible")

    collision_set = {
        (r["artist"].strip().lower(), r["track"].strip().lower())
        for r in collision_memory
    }

    # Build full known_tracks set from ALL history (before any purge) —
    # used as post-generation safety filter so Claude never recommends
    # a song the user has already heard, even if it's not in the anchor pool.
    known_tracks: frozenset = frozenset(
        (artist.strip().lower(), track.strip().lower())
        for artist, s in stats.items()
        for track, _ in s.top_tracks
    )

    anchors = build_anchor_pool(
        eligible, top_tracks_per_artist, anchor_pool_size, collision_set,
        min_track_plays, freshness_penalties=freshness_penalties,
    )

    if len(anchors) < 5:
        raise AnchorPoolTooSmallError(
            f"Only {len(anchors)} anchor tracks survived purge + collision filtering. "
            "Try loosening the purge thresholds."
        )

    return AnchorPool(
        tracks=anchors,
        purged_artists=purged,
        eligible_count=len(eligible),
        total_scrobbles=total_scrobbles,
        total_artists=len(stats),
        known_tracks=known_tracks,
    )
