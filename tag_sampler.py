#!/usr/bin/env python3
"""
tag_sampler.py  —  One-time Last.fm tag sampling script
                   for the Music Discovery Agent.

Reads a listening history CSV, selects up to --sample tracks (default 500),
fetches Last.fm tags for each, then writes three files to ./tag_sample_out/:

  tag_frequencies.csv    — every unique tag seen, sorted by track-occurrence count
  lane_clusters.txt      — human-readable report: lane coverage + top tags per lane
  lane_tags_draft.py     — ready-to-paste LANE_TAGS dict for lastfm_api.py

Mutual-exclusivity pass: each tag is assigned to the single lane where it appears
most frequently. Tags that appear roughly equally across 3+ lanes are flagged as
"shared" (retained in all relevant lanes but noted in the report).

Usage
-----
  python3 tag_sampler.py \\
      --csv   /path/to/history.csv \\
      --source lastfm              \\   # or: spotify
      --api-key YOUR_LASTFM_API_KEY  \\
      --sample 500

The script is read-only — it never modifies any agent file.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


# ─── agent imports (optional — fallback parser included) ───────────────────────
_AGENT_DIR = Path(__file__).parent
sys.path.insert(0, str(_AGENT_DIR))
try:
    from history import parse_history_csv, UTILITY_KEYWORDS
    _HAVE_HISTORY = True
except ImportError:
    _HAVE_HISTORY = False
    UTILITY_KEYWORDS = [
        "white noise", "sleep", "rain sound", "brown noise", "pink noise",
        "binaural", "focus music", "study music", "meditation", "lofi beats",
        "lo-fi beats", "ambient noise", "baby sleep", "sleep aid",
        "unknown", "untitled",
    ]

OUT_DIR = Path("tag_sample_out")
LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"
RATE_LIMIT = 0.22   # ~4.5 req/s, under Last.fm's 5 req/s cap


# ─────────────────────────────────────────────────────────────────────────────
#  Lane seed tags — minimal, high-confidence anchors per lane.
#  These label tracks so empirical co-occurring tags can be discovered.
# ─────────────────────────────────────────────────────────────────────────────

LANE_SEEDS: dict[str, set[str]] = {
    "Melancholy Balladry": {
        "ballad", "sad", "melancholic", "melancholy", "grief", "slowcore",
        "lullaby", "tearjerker", "dirge",
    },
    "Introspective Songcraft": {
        "singer-songwriter", "storytelling", "confessional", "lyrical",
        "narrative", "folk", "americana",
    },
    "Hook-Perfected Songs": {
        "power pop", "bubblegum", "hook", "catchy", "motown",
        "girl group", "boy band",
    },
    "Atmospheric / Texture-First": {
        "ambient", "drone", "krautrock", "soundscape",
        "neoclassical", "modern classical", "microsound",
    },
    "Propulsive Guitar Rock": {
        "garage rock", "post-punk", "jangle pop", "college rock",
        "paisley underground",
    },
    "Hybrid Rock (Melody-Led)": {
        "britpop", "new wave", "pop rock", "alternative pop",
        "chamber pop",
    },
    "Groove-First Rock (Selective)": {
        "blues rock", "funk rock", "southern rock", "boogie",
        "swamp rock", "jam band",
    },
    "Energy-First Rock (Melody-Bounded)": {
        "grunge", "post-grunge", "arena rock", "hard rock",
        "pop punk", "emo", "alternative rock",
    },
    "Rhythm-Dominant Rock (Cathartic)": {
        "hardcore", "post-hardcore", "math rock", "screamo", "metalcore",
    },
    "Texture-First Rock (Atmospheric)": {
        "shoegaze", "post-rock", "dream pop", "wall of sound",
        "noise pop",
    },
    "Cathartic / Adrenaline Rock": {
        "thrash metal", "death metal", "black metal", "grindcore",
        "sludge metal", "doom metal",
    },
    "Melody-Anchored Hip-Hop": {
        "alternative hip hop", "conscious hip hop", "jazz rap",
        "hip hop soul", "neo soul",
    },
    "Groove-First Hip-Hop (Selective)": {
        "g-funk", "bounce", "gangsta rap", "boom bap", "dirty south",
    },
    "Textural / Atmospheric Hip-Hop": {
        "cloud rap", "chillhop", "instrumental hip hop", "lo-fi hip hop",
    },
    "Melody-Led Jazz (Theme-Centric)": {
        "bebop", "cool jazz", "hard bop", "vocal jazz", "standards", "big band",
    },
    "Harmonic-First Jazz": {
        "post-bop", "modal jazz", "fusion", "contemporary jazz",
    },
    "Spiritual / Expansive Jazz": {
        "free jazz", "avant-garde jazz", "spiritual jazz",
    },
    "Rhythm-Forward Jazz (Selective)": {
        "latin jazz", "afrobeat", "bossa nova", "samba", "jazz funk",
    },
}

# Tags too generic/personal to carry any lane signal — always excluded
NOISE_TAGS: set[str] = {
    "seen live", "favourite", "favorites", "favourites", "favorite",
    "love", "awesome", "great", "amazing", "beautiful", "cool", "best",
    "classic", "good", "my favorite", "all", "music", "song", "album",
    "track", "artist", "band", "bands", "under 2000 listeners",
    "under 5000 listeners", "spotify", "youtube", "itunes", "lastfm",
    "recommended", "check out", "discovery", "playlist",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback CSV parser (used if history.py is not importable)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_parse(csv_path: Path, source: str) -> list[tuple[str, str, int]]:
    """
    Returns [(artist, track, play_count), ...] sorted by plays descending.
    Minimal parser — just enough to get artist+track pairs.
    """
    plays: dict[tuple[str, str], int] = defaultdict(int)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.lower().strip() for c in (reader.fieldnames or [])]

        for row in reader:
            row_lower = {k.lower().strip(): v for k, v in row.items()}
            if source == "lastfm":
                artist = (row_lower.get("artist") or "").strip()
                track  = (row_lower.get("track")  or "").strip()
            elif "master_metadata_track_name" in fieldnames:
                artist = (row_lower.get("master_metadata_album_artist_name") or "").strip()
                track  = (row_lower.get("master_metadata_track_name")        or "").strip()
                ms     = int(row_lower.get("ms_played") or 0)
                if ms < 30_000:
                    continue
            else:
                artist = (row_lower.get("artistname") or row_lower.get("artist name") or "").strip()
                track  = (row_lower.get("trackname")  or row_lower.get("track name")  or "").strip()

            if artist and track:
                plays[(artist, track)] += 1

    return sorted(
        [(a, t, c) for (a, t), c in plays.items()],
        key=lambda x: x[2], reverse=True,
    )


def _is_utility(name: str) -> bool:
    lo = name.lower()
    return any(kw in lo for kw in UTILITY_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
#  Last.fm API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_tags(params: dict) -> list[tuple[str, int]]:
    params["format"] = "json"
    url = LASTFM_BASE + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = data.get("toptags", {}).get("tag", [])
        if isinstance(raw, dict):
            raw = [raw]
        return [
            (t["name"].lower().strip(), int(t["count"]))
            for t in raw
            if int(t["count"]) > 0
        ]
    except Exception:
        return []


def get_track_tags(artist: str, track: str, api_key: str) -> list[tuple[str, int]]:
    return _fetch_tags({
        "method": "track.getTopTags", "artist": artist,
        "track": track, "api_key": api_key, "autocorrect": "1",
    })


def get_artist_tags(artist: str, api_key: str) -> list[tuple[str, int]]:
    return _fetch_tags({
        "method": "artist.getTopTags", "artist": artist,
        "api_key": api_key, "autocorrect": "1",
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Lane assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_lane(tags: list[tuple[str, int]]) -> str | None:
    """
    Return the lane whose seed tags best match the track's tag set.
    Returns None if no seed matches found.
    """
    tag_names = {t for t, _ in tags}
    scores = {
        lane: len(seeds & tag_names)
        for lane, seeds in LANE_SEEDS.items()
    }
    best_lane = max(scores, key=scores.get)
    return best_lane if scores[best_lane] > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis: mutual-exclusivity pass
# ─────────────────────────────────────────────────────────────────────────────

def build_exclusive_lane_tags(
    lane_tag_counts: dict[str, dict[str, int]],
    min_occurrences: int = 2,
    shared_threshold: float = 0.85,  # if top lane has < this fraction of total → "shared"
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """
    For each candidate tag, assign it to the lane where it appears most often.
    Tags that appear roughly equally across 3+ lanes are flagged as "shared".

    Returns:
        exclusive_tags: lane → set of tags assigned exclusively to that lane
        shared_tags:    tag → [list of lanes it spans] (for reporting)
    """
    # Collect all tags across all lanes, count total occurrences
    all_tags: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for lane, tag_counts in lane_tag_counts.items():
        for tag, count in tag_counts.items():
            if tag in NOISE_TAGS:
                continue
            if count >= min_occurrences:
                all_tags[tag][lane] += count

    exclusive_tags: dict[str, set[str]] = {lane: set() for lane in LANE_SEEDS}
    shared_tags: dict[str, list[str]] = {}

    for tag, lane_counts in all_tags.items():
        total = sum(lane_counts.values())
        best_lane = max(lane_counts, key=lane_counts.get)
        best_frac = lane_counts[best_lane] / total

        if best_frac >= shared_threshold or len(lane_counts) == 1:
            # Clearly belongs to one lane
            exclusive_tags[best_lane].add(tag)
        else:
            # Appears across multiple lanes with no clear winner
            top_lanes = sorted(lane_counts, key=lane_counts.get, reverse=True)
            shared_tags[tag] = top_lanes
            # Still add to all qualifying lanes (user can review)
            for lane in top_lanes:
                exclusive_tags[lane].add(tag)

    return exclusive_tags, shared_tags


# ─────────────────────────────────────────────────────────────────────────────
#  Output writers
# ─────────────────────────────────────────────────────────────────────────────

def write_tag_frequencies(
    global_tag_counts: dict[str, int],
    out_path: Path,
) -> None:
    rows = sorted(global_tag_counts.items(), key=lambda x: x[1], reverse=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Tag", "Track Occurrences"])
        for tag, count in rows:
            w.writerow([tag, count])
    print(f"  ✓ {out_path.name}  ({len(rows)} unique tags)")


def write_lane_clusters(
    lane_tag_counts:  dict[str, dict[str, int]],
    lane_track_count: dict[str, int],
    exclusive_tags:   dict[str, set[str]],
    shared_tags:      dict[str, list[str]],
    total_tracks:     int,
    out_path:         Path,
) -> None:
    lines = [
        "═" * 70,
        "  LAST.FM TAG CLUSTER REPORT",
        f"  Tracks sampled: {total_tracks}",
        "═" * 70,
        "",
        "SHARED TAGS (appear roughly equally across 3+ lanes — review carefully)",
        "─" * 70,
    ]
    if shared_tags:
        for tag, lanes in sorted(shared_tags.items()):
            lines.append(f"  {tag!r:40s}  →  {', '.join(lanes)}")
    else:
        lines.append("  None — all tags assigned exclusively.")
    lines += ["", ""]

    for lane in LANE_SEEDS:
        track_n = lane_track_count.get(lane, 0)
        lines += [
            "─" * 70,
            f"  {lane}",
            f"  Tracks labeled to this lane: {track_n}",
            "─" * 70,
        ]
        tag_counts = lane_tag_counts.get(lane, {})
        assigned   = exclusive_tags.get(lane, set())
        top = sorted(
            [(t, c) for t, c in tag_counts.items() if t in assigned],
            key=lambda x: x[1], reverse=True,
        )[:30]
        if top:
            for tag, count in top:
                marker = " [shared]" if tag in shared_tags else ""
                lines.append(f"    {count:>4}×  {tag}{marker}")
        else:
            lines.append("    (no tags above threshold — lane underrepresented in sample)")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ {out_path.name}")


def write_lane_tags_draft(
    exclusive_tags: dict[str, set[str]],
    shared_tags:    dict[str, list[str]],
    out_path:       Path,
) -> None:
    """Write a ready-to-paste Python file defining LANE_TAGS."""
    lines = [
        '"""',
        "lane_tags_draft.py — Generated by tag_sampler.py",
        "Review the tags below, then copy the LANE_TAGS dict into lastfm_api.py.",
        "",
        "Tags marked [shared] appear in multiple lanes — decide whether to keep,",
        "move, or remove them before pasting.",
        '"""',
        "",
        "LANE_TAGS: dict[str, set[str]] = {",
    ]

    for lane, tags in exclusive_tags.items():
        sorted_tags = sorted(tags)
        shared_in_lane = [t for t in sorted_tags if t in shared_tags]
        lines.append(f'    "{lane}": {{')
        for tag in sorted_tags:
            note = "  # [shared]" if tag in shared_tags else ""
            lines.append(f'        "{tag}",{note}')
        lines.append("    },")
        lines.append("")

    lines += ["}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample Last.fm tags for a listening history CSV and cluster by lane."
    )
    parser.add_argument("--csv",     required=True, help="Path to listening history CSV")
    parser.add_argument("--source",  required=True, choices=["lastfm", "spotify"],
                        help="CSV source format")
    parser.add_argument("--api-key", required=True, help="Last.fm API key")
    parser.add_argument("--sample",  type=int, default=500,
                        help="Max tracks to sample (default: 500)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Error: file not found — {csv_path}")

    OUT_DIR.mkdir(exist_ok=True)

    # ── 1. Parse CSV ──────────────────────────────────────────────────────────
    print("\n[1/4] Parsing listening history...")
    if _HAVE_HISTORY:
        stats, total_scrobbles, fmt = parse_history_csv(csv_path, args.source)
        # Flatten to (artist, track, plays) list, utility-purged, sorted by plays
        all_tracks: list[tuple[str, str, int]] = []
        for artist, s in stats.items():
            if _is_utility(artist):
                continue
            for track, plays in s.top_tracks:
                all_tracks.append((artist, track, plays))
        all_tracks.sort(key=lambda x: x[2], reverse=True)
        print(f"  {total_scrobbles:,} scrobbles  |  {len(stats):,} artists  |  {len(all_tracks):,} unique tracks")
    else:
        all_tracks = [
            (a, t, c) for a, t, c in _fallback_parse(csv_path, args.source)
            if not _is_utility(a)
        ]
        print(f"  {len(all_tracks):,} unique tracks (fallback parser)")

    sample = all_tracks[:args.sample]
    print(f"  Sampling top {len(sample)} tracks by play count")

    # ── 2. Fetch tags ─────────────────────────────────────────────────────────
    print(f"\n[2/4] Fetching Last.fm tags ({len(sample)} tracks)...")
    print("  (track.getTopTags → artist.getTopTags fallback when < 3 track tags)")

    artist_cache: dict[str, list[tuple[str, int]]] = {}
    results: list[dict] = []

    for i, (artist, track, plays) in enumerate(sample):
        tags = get_track_tags(artist, track, args.api_key)
        time.sleep(RATE_LIMIT)

        if len(tags) < 3:
            akey = artist.lower().strip()
            if akey not in artist_cache:
                artist_cache[akey] = get_artist_tags(artist, args.api_key)
                time.sleep(RATE_LIMIT)
            tags = artist_cache[akey]

        results.append({"artist": artist, "track": track, "plays": plays, "tags": tags})

        # Progress indicator
        if (i + 1) % 50 == 0 or (i + 1) == len(sample):
            pct = (i + 1) / len(sample) * 100
            print(f"  {i+1}/{len(sample)}  ({pct:.0f}%)")

    # ── 3. Analyse ────────────────────────────────────────────────────────────
    print("\n[3/4] Analysing tags...")

    global_tag_counts:  dict[str, int]             = defaultdict(int)
    lane_tag_counts:    dict[str, dict[str, int]]  = {l: defaultdict(int) for l in LANE_SEEDS}
    lane_track_count:   dict[str, int]             = defaultdict(int)
    unlabeled = 0

    for r in results:
        tags = [(t, c) for t, c in r["tags"] if t not in NOISE_TAGS]
        for tag, _ in tags:
            global_tag_counts[tag] += 1

        lane = assign_lane(tags)
        if lane:
            lane_track_count[lane] += 1
            for tag, count in tags:
                if tag not in NOISE_TAGS:
                    lane_tag_counts[lane][tag] += 1
        else:
            unlabeled += 1

    print(f"  {len(results) - unlabeled} tracks labeled to a lane")
    print(f"  {unlabeled} tracks unlabeled (no seed-tag match)")
    print(f"  {len(global_tag_counts)} unique tags observed")

    # Lane coverage report
    print("\n  Lane coverage:")
    for lane, count in sorted(lane_track_count.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * min(count, 40)
        print(f"    {count:>3}  {lane[:45]:<45}  {bar}")

    exclusive_tags, shared_tags = build_exclusive_lane_tags(lane_tag_counts)
    print(f"\n  Shared (ambiguous) tags: {len(shared_tags)}")

    # ── 4. Write output ───────────────────────────────────────────────────────
    print(f"\n[4/4] Writing output to ./{OUT_DIR}/")

    write_tag_frequencies(
        global_tag_counts,
        OUT_DIR / "tag_frequencies.csv",
    )
    write_lane_clusters(
        lane_tag_counts,
        lane_track_count,
        exclusive_tags,
        shared_tags,
        total_tracks=len(results),
        out_path=OUT_DIR / "lane_clusters.txt",
    )
    write_lane_tags_draft(
        exclusive_tags,
        shared_tags,
        OUT_DIR / "lane_tags_draft.py",
    )

    print("\n✓ Done. Next steps:")
    print("  1. Review tag_sample_out/lane_clusters.txt — check lane coverage and shared tags")
    print("  2. Review tag_sample_out/lane_tags_draft.py — edit as needed")
    print("  3. Copy the LANE_TAGS dict into music-agent/lastfm_api.py")
    print("  4. Re-run the Streamlit app — anchor pool lane fit will use the enriched tags\n")


if __name__ == "__main__":
    main()
