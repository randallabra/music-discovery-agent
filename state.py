from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_FRESHNESS_LOOKBACK = 5      # how many past runs to consider
_FRESHNESS_PENALTY   = 0.18  # weight reduction per run an artist appeared in
_FRESHNESS_FLOOR     = 0.10  # minimum weight factor (artist never fully excluded)


@dataclass
class ProjectState:
    blacklist: set[str] = field(default_factory=set)           # normalized (lower) artist names
    collision_memory: list[dict] = field(default_factory=list)  # [{"artist": ..., "track": ...}]
    run_count: int = 0
    last_lane: Optional[str] = None
    anchor_pool_history: list = field(default_factory=list)
    ignore_collision_memory: bool = False
    # list of lists: each entry is the set of (lower) artist names used in one anchor pool,
    # most recent last. Capped at _FRESHNESS_LOOKBACK entries.

    # --- Normalization ---

    @staticmethod
    def _norm(s: str) -> str:
        return s.strip().lower()

    # --- Blacklist ---

    def is_blacklisted(self, artist: str) -> bool:
        return self._norm(artist) in self.blacklist

    def add_to_blacklist(self, artists: list[str]) -> None:
        for a in artists:
            self.blacklist.add(self._norm(a))

    # --- Collision memory ---

    def in_collision_memory(self, artist: str, track: str) -> bool:
        if self.ignore_collision_memory:
            return False
        key = (self._norm(artist), self._norm(track))
        return key in self._collision_set()

    def _collision_set(self) -> set[tuple[str, str]]:
        return {(self._norm(r["artist"]), self._norm(r["track"])) for r in self.collision_memory}

    def add_recommendations(self, recs: list[dict]) -> None:
        existing = self._collision_set()
        for r in recs:
            key = (self._norm(r["artist"]), self._norm(r["track"]))
            if key not in existing:
                self.collision_memory.append({"artist": r["artist"], "track": r["track"]})
                existing.add(key)

    # --- Anchor pool freshness ---

    def record_anchor_pool(self, artist_names: list[str]) -> None:
        """
        Call after each run with the list of artist names that appeared
        in the anchor pool. Keeps only the last _FRESHNESS_LOOKBACK entries.
        """
        normalized = [a.strip().lower() for a in artist_names]
        self.anchor_pool_history.append(normalized)
        self.anchor_pool_history = self.anchor_pool_history[-_FRESHNESS_LOOKBACK:]

    def freshness_penalties(self) -> dict[str, float]:
        """
        Returns a dict of {artist_lower: weight_factor} for artists that have
        appeared in recent anchor pools. Factor ranges from 1.0 (never seen)
        down to _FRESHNESS_FLOOR (appeared in every recent run).
        Artists not in the dict implicitly have factor 1.0.
        """
        penalties: dict[str, float] = {}
        lookback = self.anchor_pool_history[-_FRESHNESS_LOOKBACK:]
        for past_artists in lookback:
            for artist in past_artists:
                penalties[artist] = penalties.get(artist, 0) + 1
        return {
            artist: max(_FRESHNESS_FLOOR, 1.0 - _FRESHNESS_PENALTY * count)
            for artist, count in penalties.items()
        }

    # --- Serialization ---

    def to_dict(self) -> dict:
        return {
            "blacklist": sorted(self.blacklist),
            "collision_memory": self.collision_memory,
            "run_count": self.run_count,
            "last_lane": self.last_lane,
            "anchor_pool_history": self.anchor_pool_history,
            "ignore_collision_memory": self.ignore_collision_memory,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectState":
        return cls(
            blacklist=set(d.get("blacklist", [])),
            collision_memory=d.get("collision_memory", []),
            run_count=d.get("run_count", 0),
            last_lane=d.get("last_lane"),
            anchor_pool_history=d.get("anchor_pool_history", []),
            ignore_collision_memory=d.get("ignore_collision_memory", False),
        )


def load_state(state_file: Path) -> ProjectState:
    if not state_file.exists():
        return ProjectState()
    with open(state_file, encoding="utf-8") as f:
        return ProjectState.from_dict(json.load(f))


def save_state(state: ProjectState, state_file: Path) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write
    fd, tmp_path = tempfile.mkstemp(dir=state_file.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)
        os.replace(tmp_path, state_file)
    except Exception:
        os.unlink(tmp_path)
        raise
