from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------- Lane catalogue ----------
# Each entry: (display name, one-line character description)

LANES: list[tuple[str, str]] = [
    ("Melancholy Balladry",               "grief, restraint, emotional weight — songs that sit with you"),
    ("Introspective Songcraft",           "observation, interiority, lyric-forward, narrative precision"),
    ("Hook-Perfected Songs",              "melodic payoff, vocal hooks, compositional economy"),
    ("Atmospheric / Texture-First",       "soundscapes, mood, sonic environment over melody"),
    ("Propulsive Guitar Rock",            "forward motion, guitar-led, rhythmic momentum"),
    ("Hybrid Rock (Melody-Led)",          "rock architecture with melodic priority"),
    ("Groove-First Rock (Selective)",     "rhythm-led rock, restraint in the pocket"),
    ("Energy-First Rock (Melody-Bounded)","intensity with a melodic ceiling — power without abandon"),
    ("Rhythm-Dominant Rock (Cathartic)",  "catharsis through rhythm, physical release"),
    ("Texture-First Rock (Atmospheric)",  "atmospheric rock, texture and space over riffs"),
    ("Cathartic / Adrenaline Rock",       "pure kinetic release, no brakes"),
    ("Melody-Anchored Hip-Hop",           "sample-led, melodic, mood-forward — hooks matter"),
    ("Groove-First Hip-Hop (Selective)",  "rhythm-led hip-hop, compositional selectivity"),
    ("Textural / Atmospheric Hip-Hop",    "ambient, layered, subterranean"),
    ("Melody-Led Jazz (Theme-Centric)",   "memorable themes, melody over abstraction"),
    ("Harmonic-First Jazz",               "structure-led, chord-driven, harmonic complexity"),
    ("Spiritual / Expansive Jazz",        "open, meditative, modal — space and breath"),
    ("Rhythm-Forward Jazz (Selective)",   "groove-led jazz, pocket and pulse"),
]


# ---------- RunConfig ----------

@dataclass
class RunConfig:
    # Required
    history_csv: Path
    lane: str
    project: str

    # Source: 'lastfm' or 'spotify' — user-declared, not inferred
    source: str = "lastfm"

    # Optional I/O
    output_path: Optional[Path] = None
    state_dir: Path = field(default_factory=lambda: Path.home() / ".music-agent")

    # Lane modifier
    vibe_focus: str = ""
    decade: str = "Any"

    # Purge thresholds
    max_artist_plays: int = 200
    max_unique_tracks: int = 10

    # Anchor pool
    anchor_pool_size: int = 50
    top_tracks_per_artist: int = 2
    min_track_plays: int = 8

    # Claude
    model: str = "claude-sonnet-4-6"
    batch_size: int = 20

    # CLI-only additions (applied this run, then persisted to blacklist)
    blacklist_additions: tuple = field(default_factory=tuple)

    # Flags
    dry_run: bool = False
    verbose: bool = False

    def __post_init__(self):
        self.history_csv = Path(self.history_csv)
        self.state_dir = Path(self.state_dir)
        if self.output_path is None:
            lane_slug = self.lane.split("(")[0].strip().lower().replace(" ", "_").replace("/", "_")
            self.output_path = Path(f"{self.project}_{lane_slug}_recommendations.csv")
        else:
            self.output_path = Path(self.output_path)

    @property
    def state_file(self) -> Path:
        return self.state_dir / f"{self.project}.json"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunConfig":
        return cls(
            history_csv=args.history,
            source=getattr(args, "source", "lastfm"),
            lane=args.lane,
            project=args.project,
            output_path=args.output,
            state_dir=Path(args.state_dir) if args.state_dir else Path.home() / ".music-agent",
            vibe_focus=args.vibe_focus or "",
            max_artist_plays=args.max_artist_plays,
            max_unique_tracks=args.max_unique_tracks,
            anchor_pool_size=args.anchor_pool_size,
            top_tracks_per_artist=args.top_tracks_per_artist,
            model=args.model,
            batch_size=args.batch_size,
            blacklist_additions=tuple(args.blacklist_add or []),
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
