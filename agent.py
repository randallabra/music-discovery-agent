#!/usr/bin/env python3
"""
Music Discovery Agent — Behavior-Grounded Playlist Curation  (DCSv2)

Run interactively:
  python agent.py

Run with pre-filled defaults (still prompts for confirmation):
  python agent.py --history recenttracks.csv --project myproject --lane "Melancholy Balladry"

Environment:
  ANTHROPIC_API_KEY  — required for Claude API calls
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
from pathlib import Path

from config import LANES, RunConfig
from history import AnchorPool, AnchorPoolTooSmallError, process_history
from recommender import RecommendationResult, get_recommendations
from state import ProjectState, load_state, save_state


# ═══════════════════════════════════════════════════════════════
#  Terminal helpers
# ═══════════════════════════════════════════════════════════════

WIDTH = 62

def _banner() -> None:
    print()
    print("╔" + "═" * WIDTH + "╗")
    print("║" + "  Music Discovery Agent  ·  DCSv2 Engine".center(WIDTH) + "║")
    print("╚" + "═" * WIDTH + "╝")
    print()

def _section(title: str) -> None:
    print()
    print(f"  {'─' * (WIDTH - 4)}")
    print(f"  {title.upper()}")

def _rule() -> None:
    print("  " + "─" * (WIDTH - 2))

def _ask(prompt: str, default: str = "") -> str:
    """Prompt the user. Returns stripped input or default if blank."""
    hint = f"  [{default}]" if default else ""
    try:
        val = input(f"\n  {prompt}{hint}\n  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nAborted.")
        sys.exit(0)
    return val if val else default

def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print(f"  Please enter a whole number.")

def _confirm(prompt: str = "Continue?", default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    try:
        val = input(f"\n  {prompt} {hint} > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n\nAborted.")
        sys.exit(0)
    if val == "":
        return default
    return val in ("y", "yes")


# ═══════════════════════════════════════════════════════════════
#  Interactive wizard steps
# ═══════════════════════════════════════════════════════════════

def _step_history(default_path: str = "") -> tuple[Path, str]:
    """
    Returns (csv_path, source) where source is 'lastfm' or 'spotify'.
    """
    _section("Step 1 — Listening History")
    print()
    print("  You'll need a CSV export of your listening history. Two options:")
    print()
    print("  A) Spotify")
    print("       Request your data at:")
    print("         https://www.spotify.com/us/account/privacy/")
    print("       Note: export takes up to 5 days for the past year,")
    print("       or up to 30 days for your full lifetime history.")
    print()
    print("  B) Last.fm  (recommended — available in under a minute)")
    print("       If you have the Last.fm scrobbler installed, export here:")
    print("         https://lastfm.ghan.nl/export/")
    print()

    # Source selection
    while True:
        raw = _ask("Which source is your CSV from?  [1] Last.fm  [2] Spotify", "1")
        if raw in ("1", "lastfm", "last.fm", "last fm"):
            source = "lastfm"
            break
        if raw in ("2", "spotify"):
            source = "spotify"
            break
        print("  Please enter 1 for Last.fm or 2 for Spotify.")

    # File path
    while True:
        raw = _ask("Path to your CSV file", default_path)
        if not raw:
            print("  A file path is required.")
            continue
        p = Path(raw).expanduser()
        if not p.exists():
            print(f"  File not found: {p}")
            continue
        return p, source


def _step_project(default: str = "") -> str:
    _section("Step 2 — Listener Profile")
    print("  The profile name stores your blacklist and session history.")
    print("  Use one profile per listener (e.g. 'alice', 'bob').")
    while True:
        name = _ask("Profile name", default)
        if name:
            return name.replace(" ", "_")
        print("  A profile name is required.")


def _step_parameters(
    state: ProjectState,
    defaults: dict,
) -> dict:
    _section("Step 3 — Discovery Parameters")
    print("  These control how aggressively familiar artists are purged.")
    print("  Press Enter to keep the value in [brackets].")
    print()
    print(f"  Current profile: {defaults['project']}")
    print(f"    Previous runs   : {state.run_count}")
    print(f"    Collision memory: {len(state.collision_memory)} tracks already recommended")
    print(f"    Blacklisted     : {len(state.blacklist)} artists")

    max_plays = _ask_int(
        "Max plays per artist before purging as 'saturated'",
        defaults["max_artist_plays"],
    )
    max_tracks = _ask_int(
        "Max unique tracks per artist before purging",
        defaults["max_unique_tracks"],
    )
    batch_size = _ask_int(
        "Songs to recommend this run",
        defaults["batch_size"],
    )
    anchor_size = _ask_int(
        "Anchor pool size (taste signals sent to Claude)",
        defaults["anchor_pool_size"],
    )
    return {
        "max_artist_plays": max_plays,
        "max_unique_tracks": max_tracks,
        "batch_size": batch_size,
        "anchor_pool_size": anchor_size,
    }


def _step_anchor_pool(pool: AnchorPool) -> bool:
    """
    Display the computed anchor pool and ask the user to confirm it.
    Returns True if confirmed, False if they want to re-run with different thresholds.
    """
    _section("Anchor Pool Preview")
    print(f"  {len(pool.tracks)} taste signals from {pool.eligible_count:,} eligible artists.")
    print(f"  These drive the adjacency scoring — they are NOT candidates for recommendation.")
    print()
    for i, t in enumerate(pool.tracks):
        artist = t["artist"][:38]
        track = t["track"][:38]
        print(f"  {i+1:2}.  {artist}  —  {track}  ({t['plays']} plays)")

    print()
    print("  Does this anchor pool represent your taste correctly?")
    print("  If it's full of artists you barely know, tighten the thresholds.")
    return _confirm("Proceed with this anchor pool?", default=True)


def _step_lane(default_lane: str = "") -> str:
    _section("Step 4 — Musical Lane")
    print("  The lane sets the filtering logic for recommendations.")
    print("  It controls what kinds of songs the engine targets — not which artists.")
    print()

    for i, (name, desc) in enumerate(LANES):
        idx = str(i + 1).rjust(2)
        wrapped = textwrap.fill(desc, width=50)
        first_line, *rest = wrapped.split("\n")
        print(f"  {idx}  {name}")
        print(f"      {first_line}")
        for line in rest:
            print(f"      {line}")

    custom_idx = len(LANES) + 1
    print(f"\n  {str(custom_idx).rjust(2)}  Custom — type your own lane name")

    # Find default index
    default_hint = ""
    for i, (name, _) in enumerate(LANES):
        if name.lower() == default_lane.lower():
            default_hint = str(i + 1)
            break
    if not default_hint and default_lane:
        default_hint = default_lane  # passthrough for custom

    while True:
        raw = _ask("Lane number or custom name", default_hint)
        if not raw:
            print("  A lane is required.")
            continue
        # Numeric selection
        try:
            n = int(raw)
            if 1 <= n <= len(LANES):
                return LANES[n - 1][0]
            if n == custom_idx:
                custom = _ask("Enter your custom lane name", "")
                if custom:
                    return custom
            else:
                print(f"  Enter a number between 1 and {custom_idx}.")
        except ValueError:
            # Treat as a literal lane name
            return raw


def _step_vibe(default: str = "") -> str:
    _section("Step 5 — Vibe Focus  (optional)")
    print("  Narrow the lane further with a descriptive modifier.")
    print('  Examples: "fingerpicked guitar"  |  "female vocals"  |  "pre-1980"')
    raw = _ask("Vibe focus", default or "skip")
    return "" if raw.lower() in ("skip", "") else raw


def _step_blacklist(state: ProjectState, defaults: list[str]) -> list[str]:
    _section("Step 6 — Blacklist")
    if state.blacklist:
        print(f"  Currently blacklisted ({len(state.blacklist)}):")
        for a in sorted(state.blacklist)[:15]:
            print(f"    - {a}")
        if len(state.blacklist) > 15:
            print(f"    ... and {len(state.blacklist) - 15} more")
    else:
        print("  No artists are currently blacklisted.")
    print()
    print("  Add artists to permanently exclude from all future runs.")
    print("  Separate multiple artists with a comma.")

    raw = _ask("Artists to add (or Enter to skip)", "")
    if not raw:
        return []
    return [a.strip() for a in raw.split(",") if a.strip()]


def _step_output(default: str) -> Path:
    _section("Step 7 — Output File")
    print("  This file will be importable directly into Soundiiz.")
    raw = _ask("Output CSV filename", default)
    p = Path(raw)
    if not p.suffix:
        p = p.with_suffix(".csv")
    return p


def _show_summary(
    history_csv: Path,
    project: str,
    lane: str,
    vibe: str,
    params: dict,
    output_path: Path,
    new_blacklist: list[str],
) -> None:
    print()
    _rule()
    print(f"  Profile   : {project}")
    print(f"  History   : {history_csv.name}")
    print(f"  Lane      : {lane}")
    print(f"  Vibe      : {vibe or '—'}")
    print(f"  Purge     : ≥ {params['max_artist_plays']} plays  OR  ≥ {params['max_unique_tracks']} unique tracks")
    print(f"  Batch     : {params['batch_size']} songs")
    print(f"  Anchor    : {params['anchor_pool_size']} tracks")
    print(f"  Output    : {output_path}")
    if new_blacklist:
        print(f"  Adding to blacklist: {', '.join(new_blacklist)}")
    _rule()


# ═══════════════════════════════════════════════════════════════
#  Output writers
# ═══════════════════════════════════════════════════════════════

def write_soundiiz_csv(recommendations: list, output_path: Path) -> None:
    """No header row — Soundiiz reads row 1 as the first artist/track entry."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in recommendations:
            writer.writerow([r.artist, r.track])


def write_detail_csv(result: RecommendationResult, output_path: Path) -> None:
    """Full-detail CSV alongside the Soundiiz file, with DCS scores and rationales."""
    detail_path = output_path.with_stem(output_path.stem + "_detail")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Artist", "Track", "DCS_Score",
            "CLS (Co-Listening Strength): How strongly the song connects to your anchors across many listeners",
            "CMS (Credible Mention Strength): Quality, depth, and independence of critical references",
            "MES (Mechanics Evidence Score): Only applied when critics describe what happens in the song (tension, release, arrangement, motion — never inferred)",
            "Rationale",
        ])
        for r in result.recommendations:
            writer.writerow([
                r.artist, r.track,
                r.dcs_score or "",
                r.cls_score or "", r.cms_score or "", r.mes_score or "",
                r.rationale,
            ])
    print(f"  Detail   : {detail_path}")


def print_recommendations(result: RecommendationResult) -> None:
    print()
    print(f"  {'#':<3} {'DCS':<5} {'Artist':<36} Track")
    print(f"  {'─'*3} {'─'*5} {'─'*36} {'─'*32}")
    for i, r in enumerate(result.recommendations):
        dcs = f"{r.dcs_score:.2f}" if r.dcs_score is not None else "  — "
        artist = r.artist[:35]
        track = r.track[:42]
        print(f"  {i+1:<3} {dcs:<5} {artist:<36} {track}")


# ═══════════════════════════════════════════════════════════════
#  Core run logic (shared between interactive and CLI modes)
# ═══════════════════════════════════════════════════════════════

def execute_run(config: RunConfig, state: ProjectState) -> int:
    """
    Runs the full pipeline given a finalized config and loaded state.
    Returns exit code.
    """
    # Persist blacklist additions before doing anything else
    if config.blacklist_additions:
        state.add_to_blacklist(list(config.blacklist_additions))
        save_state(state, config.state_file)

    # Process history → anchor pool
    pool = process_history(
        csv_path=config.history_csv,
        source=config.source,
        max_artist_plays=config.max_artist_plays,
        max_unique_tracks=config.max_unique_tracks,
        top_tracks_per_artist=config.top_tracks_per_artist,
        anchor_pool_size=config.anchor_pool_size,
        blacklist=state.blacklist,
        collision_memory=state.collision_memory,
        verbose=config.verbose,
    )

    if config.dry_run:
        _section("Anchor Pool")
        for i, t in enumerate(pool.tracks):
            print(f"  {i+1:2}.  {t['artist']}  —  {t['track']}  ({t['plays']} plays)")
        print("\n  Dry run — skipping Claude call.")
        return 0

    # Call Claude
    _section("Generating Recommendations")
    result = get_recommendations(config, pool, state)

    if not result.recommendations:
        print("  No recommendations returned. Discovery space may be exhausted.")
        if config.verbose:
            print("\n  Raw Claude response:")
            print(result.raw_response[:800])
        return 1

    # Display
    _section(f"Recommendations  ({len(result.recommendations)} tracks  ·  {result.tokens_used:,} tokens)")
    print_recommendations(result)

    # Write output
    _section("Output")
    write_soundiiz_csv(result.recommendations, config.output_path)
    write_detail_csv(result, config.output_path)
    print(f"  Soundiiz : {config.output_path}")

    # Persist state
    state.add_recommendations([
        {"artist": r.artist, "track": r.track}
        for r in result.recommendations
    ])
    state.run_count += 1
    state.last_lane = config.lane
    save_state(state, config.state_file)
    print(f"\n  State saved — collision memory now: {len(state.collision_memory)} tracks total")
    print()
    return 0


# ═══════════════════════════════════════════════════════════════
#  Interactive session
# ═══════════════════════════════════════════════════════════════

def run_interactive(defaults: argparse.Namespace) -> int:
    """
    Walk the user through each parameter, show an anchor pool preview,
    then confirm before calling Claude.
    """
    _banner()

    # ── Step 1: History CSV + source ─────────────────────────────
    history_csv, source = _step_history(getattr(defaults, "history", "") or "")

    # ── Step 2: Profile ──────────────────────────────────────────
    project = _step_project(getattr(defaults, "project", "") or "")

    # ── Load state early so we can show run history ──────────────
    state_dir = Path(getattr(defaults, "state_dir", None) or Path.home() / ".music-agent")
    state_file = state_dir / f"{project}.json"
    state = load_state(state_file)

    # ── Step 3: Parameters ───────────────────────────────────────
    param_defaults = {
        "project": project,
        "max_artist_plays": getattr(defaults, "max_artist_plays", 200),
        "max_unique_tracks": getattr(defaults, "max_unique_tracks", 10),
        "batch_size": getattr(defaults, "batch_size", 20),
        "anchor_pool_size": getattr(defaults, "anchor_pool_size", 30),
    }
    params = _step_parameters(state, param_defaults)

    # ── Compute anchor pool so user can review it ─────────────────
    _section("Processing History")
    while True:
        try:
            pool = process_history(
                csv_path=history_csv,
                source=source,
                max_artist_plays=params["max_artist_plays"],
                max_unique_tracks=params["max_unique_tracks"],
                top_tracks_per_artist=getattr(defaults, "top_tracks_per_artist", 3),
                anchor_pool_size=params["anchor_pool_size"],
                blacklist=state.blacklist,
                collision_memory=state.collision_memory,
            )
        except AnchorPoolTooSmallError as e:
            print(f"\n  Warning: {e}")
            if not _confirm("Adjust thresholds and retry?", default=True):
                return 2
            params = _step_parameters(state, params)
            continue

        confirmed = _step_anchor_pool(pool)
        if confirmed:
            break
        # User wants to adjust thresholds
        params = _step_parameters(state, params)

    # ── Step 4: Lane ─────────────────────────────────────────────
    default_lane = (
        getattr(defaults, "lane", None)
        or state.last_lane
        or ""
    )
    lane = _step_lane(default_lane)

    # ── Step 5: Vibe focus ────────────────────────────────────────
    vibe = _step_vibe(getattr(defaults, "vibe_focus", "") or "")

    # ── Step 6: Blacklist ─────────────────────────────────────────
    new_blacklist = _step_blacklist(state, [])

    # ── Step 7: Output path ───────────────────────────────────────
    lane_slug = lane.split("(")[0].strip().lower().replace(" ", "_").replace("/", "_")
    default_output = f"{project}_{lane_slug}_run{state.run_count + 1}.csv"
    output_path = _step_output(getattr(defaults, "output", None) or default_output)

    # ── Summary + confirm ─────────────────────────────────────────
    _show_summary(history_csv, project, lane, vibe, params, output_path, new_blacklist)

    if not _confirm("Run with these settings?", default=True):
        print("\n  Cancelled.")
        return 0

    # ── Build config and run ──────────────────────────────────────
    config = RunConfig(
        history_csv=history_csv,
        source=source,
        lane=lane,
        project=project,
        output_path=output_path,
        state_dir=state_dir,
        vibe_focus=vibe,
        max_artist_plays=params["max_artist_plays"],
        max_unique_tracks=params["max_unique_tracks"],
        anchor_pool_size=params["anchor_pool_size"],
        top_tracks_per_artist=getattr(defaults, "top_tracks_per_artist", 3),
        model=getattr(defaults, "model", "claude-opus-4-6"),
        batch_size=params["batch_size"],
        blacklist_additions=tuple(new_blacklist),
        dry_run=getattr(defaults, "dry_run", False),
        verbose=getattr(defaults, "verbose", False),
    )
    # State already loaded; re-use it (blacklist additions applied inside execute_run)
    state_for_run = load_state(config.state_file)
    return execute_run(config, state_for_run)


# ═══════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Behavior-grounded music discovery agent (DCSv2). "
                    "Runs interactively by default; all flags are optional pre-fills.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--history", metavar="CSV",
                   help="Path to Last.fm or Spotify export CSV")
    p.add_argument("--source", choices=["lastfm", "spotify"], default="lastfm",
                   help="CSV source format (used with --no-interactive)")
    p.add_argument("--lane",
                   help='Musical lane, e.g. "Melancholy Balladry"')
    p.add_argument("--project",
                   help="Profile name — isolates blacklist & collision memory")
    p.add_argument("--output", metavar="CSV",
                   help="Output CSV path")
    p.add_argument("--state-dir", metavar="DIR",
                   help="Directory for state files (default: ~/.music-agent/)")
    p.add_argument("--vibe-focus", default="",
                   help='Optional sub-focus within lane, e.g. "guitar-forward"')
    p.add_argument("--max-artist-plays", type=int, default=200)
    p.add_argument("--max-unique-tracks", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--anchor-pool-size", type=int, default=30)
    p.add_argument("--top-tracks-per-artist", type=int, default=3)
    p.add_argument("--model", default="claude-opus-4-6")
    p.add_argument("--blacklist-add", action="append", metavar="ARTIST",
                   help="Add artist to blacklist (repeatable)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show anchor pool only; skip Claude call")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-interactive", action="store_true",
                   help="Skip wizard and run directly (requires --history, --lane, --project)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.no_interactive:
            # Direct mode: all three required flags must be present
            missing = [f for f in ("history", "lane", "project") if not getattr(args, f, None)]
            if missing:
                parser.error(f"--no-interactive requires: {', '.join('--' + m for m in missing)}")
            config = RunConfig.from_args(args)
            state = load_state(config.state_file)
            exit_code = execute_run(config, state)
        else:
            # Interactive wizard (all args act as pre-filled defaults)
            exit_code = run_interactive(args)

    except AnchorPoolTooSmallError as e:
        print(f"\n  Error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n\n  Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n  Error: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
