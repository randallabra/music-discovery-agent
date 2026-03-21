from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Optional

import anthropic

from config import RunConfig
from history import AnchorPool
from state import ProjectState

# ---------- Data structures ----------

@dataclass
class Recommendation:
    artist: str
    track: str
    dcs_score: Optional[float]
    cls_score: Optional[str]
    cms_score: Optional[str]
    mes_score: Optional[str]
    rationale: str


@dataclass
class RecommendationResult:
    recommendations: list[Recommendation]
    raw_response: str
    tokens_used: int
    model: str


# ---------- System prompt ----------

SYSTEM_PROMPT = """\
You are a behavior-grounded music discovery engine. Your task is to generate song-level \
recommendations using DCSv2 scoring. You are given an anchor pool (the user's demonstrated \
taste after purging familiar/saturated artists) and must discover genuinely new music.

## Core Philosophy
- Discovery, not reinforcement. Familiarity is a disqualifier, not a signal.
- Anchors define adjacency — they are reference points, never candidates for recommendation.
- Every recommendation must be explainable by evidence, not vibes.
- If discovery space is genuinely exhausted, say so. Do not pad with weak candidates.

## DCSv2 Scoring Model (apply to every candidate)
Component weights:
  CLS — Co-Listening Strength:      45%
  CMS — Credible Mention Strength:  35%
  MES — Mechanics Evidence Score:   20%

### CLS (Co-Listening Strength — 45%)
Song-level adjacency to the anchor pool via real listener co-occurrence.
Use Last.fm, RateYourMusic, and similar behavioral data from your training.
- Score 0.9+: song co-occurs across multiple anchors in many listener libraries
- Score 0.6–0.9: song co-occurs with several anchors
- Score <0.6: weak signal — only include if CMS is exceptional
- Penalize hub/overplayed tracks (songs that appear everywhere regardless of taste)
- Missing CLS = "no evidence", which is not an automatic failure but heavily penalizes composite

### CMS (Credible Mention Strength — 35%)
Songs must appear in the Credible Mention Pool — explicitly cited by whitelisted sources:

  Tier 1 (CSE-Track, base weight 1.00):
    - Pitchfork: track reviews ("Reviews: Tracks")
    - NPR Music: Song of the Day / song-first features

  Tier 2 (CSE-Lists, base weight 0.80–0.75):
    - Rolling Stone: canonical song lists (strong recency handicap — post-2010 lists weighted lower)
    - New York Times: song lists, critics' picks, album reviews that explicitly name tracks

  Tier 3 (CAE → CSE-Track-equivalent if mechanics language present, base weight 0.55→0.75):
    - Stereogum: album reviews with specific track discussion

  Tier 4 (CAE, base weight 0.50):
    - AllMusic: track mentions within reviews, "highlights" callouts

  Tier 5 (LCP — corroboration only, base weight 0.30):
    - RateYourMusic / Sonemic: list frequency, above-album-median persistence
    - NOT used as primary signal; only confirms CMS from higher tiers

  Tier 6 (eligibility gate only, base weight 0.15, heavy recency handicap):
    - Spotify charts, Apple Music charts, YouTube Music charts
    - Chart appearance alone does NOT make a song eligible; only used to break near-ties

Explicit exclusions — NEVER count these as evidence:
  - Label-owned publications / promo channels
  - Press releases or sponsored content
  - Playlist-brand SEO blogs
  - Artist-run promotional content

### MES (Mechanics Evidence Score — 20%)
Only apply when critics describe what physically happens in the song:
  tension, release, arrangement, chord motion, rhythmic structure, timbral texture,
  dynamic arc, melodic contour

NOT inferred from genre tags, instrumentation credits, or artist bio.
MES = 0.0 unless you can cite specific descriptive language from criticism.

## Candidate Generation Process
Stage 1 — Build Credible Mention Pool (CMP):
  Gather all songs explicitly cited by Tier 1–5 sources that are adjacent to the lane.
  Songs must be EXPLICITLY mentioned — not inferred from metadata.

Stage 2 — Taste Graph Expansion (TGE):
  Starting from the anchor pool, expand via song-to-song co-listening:
  - Prioritize songs that connect to MULTIPLE anchor tracks
  - Explore listener behavior "when people who love these specific songs listen further"
  - Avoid artist-level collapse (do not substitute artist similarity for song adjacency)
  - Weight anchor tracks by lane relevance: anchor tracks that align with the MUSICAL_LANE
    are stronger expansion seeds than those that do not. A folk/acoustic lane should expand
    from the user's folk/acoustic anchors first, even if their rock anchors have more plays.
    Use the lane as a lens on the pool, not a filter — all anchors remain visible, but
    lane-aligned anchors carry more weight in TGE traversal.

Stage 3 — Score + rank by DCSv2.

## VIBE_FOCUS Interpretation (when present in run configuration)
VIBE_FOCUS is a comma-separated list of qualitative descriptors (e.g. "minor key, acoustic, fingerpicked, breezy, atmospheric").

VIBE_FOCUS is a soft scoring signal only. It must NEVER reduce batch size or eliminate candidates.

Rules:
- Treat each descriptor as an independent OR qualifier — a track matches if it fits ANY one descriptor.
- Never require a track to satisfy multiple descriptors simultaneously.
- A track that matches one or more descriptors gets a modest MES boost (up to +0.08) when the
  match is supported by specific critic language or production description.
- A track with ZERO vibe matches is NOT disqualified. If its DCS score is competitive, include it.
- A high-DCS track with no vibe match outranks a low-DCS track with a perfect vibe match.
- The goal is to tilt the ranking slightly toward the vibe — not to filter the candidate pool.
- Never cite vibe focus as a reason to produce fewer songs than the requested batch size.

## Anti-Collapse Constraints (non-negotiable)
- No single era (decade) may exceed 35% of output
- No single micro-genre or adjacent scene may dominate
- Max 2 songs per artist in any single output batch
- Never recommend artists from collision memory or blacklist
- Never recommend the anchor tracks themselves
- No live versions, remasters, or obvious duplicates unless the live version is the canonical reference

## Output Format
Respond with EXACTLY a CSV block — no prose before, no prose after.
Use this header and format:
```csv
Artist,Track,DCS_Score,CLS,CMS,MES,Rationale
"Artist Name","Track Title",0.82,0.85,0.78,0.80,"Evidence-grounded rationale. No commas inside rationale field."
```

Rules:
- DCS_Score: composite float 0.00–1.00
- CLS/CMS/MES: individual component floats 0.00–1.00
- Rationale: 1–2 sentences tied to THIS SPECIFIC SONG — not the artist generally.
  Never write "shares DNA with [anchor artist]" or "fans of X will enjoy."
  Instead: describe what happens in this specific track (production, melody, structure,
  co-listening pattern, or specific critic language). The rationale must be song-level.
- Quote all string fields with double quotes
- Rank rows by DCS_Score descending
- If you cannot reach the requested batch size with credible candidates, produce fewer rows
  and add a final row with Artist="NOTE", Track="DISCOVERY_EXHAUSTED", DCS_Score=0.00
"""


# ---------- Prompt builders ----------

def build_user_message(
    anchor_pool: AnchorPool,
    state: ProjectState,
    config: RunConfig,
) -> str:
    # Anchor pool lines
    anchor_lines = "\n".join(
        f"{i+1:2}. {t['artist']} — {t['track']} ({t['plays']} plays)"
        for i, t in enumerate(anchor_pool.tracks)
    )

    # Collision memory — cap at 150 most recent for token budget
    recent_collision = state.collision_memory[-150:]
    if recent_collision:
        collision_lines = "\n".join(
            f"  - {r['artist']} — {r['track']}"
            for r in recent_collision
        )
    else:
        collision_lines = "  (none — this is the first run)"

    # Blacklist
    if state.blacklist:
        blacklist_lines = "\n".join(f"  - {a}" for a in sorted(state.blacklist))
    else:
        blacklist_lines = "  (none)"

    # Purged (saturated) artists — user has heard these extensively;
    # do NOT recommend any track by these artists
    purged = anchor_pool.purged_artists[:120]  # cap for token budget
    if purged:
        purged_lines = "\n".join(f"  - {a}" for a in purged)
    else:
        purged_lines = "  (none)"

    # Known tracks — top played songs from full history.
    # These are songs the user has ALREADY heard, regardless of artist.
    # Never recommend any of these specific tracks under any circumstances.
    known = anchor_pool.known_tracks_by_plays or []
    if known:
        known_lines = "\n".join(
            f"  - {artist} — {track} ({plays} plays)"
            for artist, track, plays in known[:200]  # cap at 200 for token budget
        )
    else:
        known_lines = "  (none)"

    if config.vibe_focus:
        vibe_line = (
            f"\nVIBE_FOCUS: {config.vibe_focus}"
            "\n  → Treat each descriptor as an independent OR qualifier, not a required AND."
            "\n  → A candidate qualifies if it meaningfully matches ANY of these descriptors."
            "\n  → Do not require all descriptors to be present in a single track."
            "\n  → Use matching descriptors to boost MES scoring where critic language supports them."
        )
    else:
        vibe_line = ""

    return f"""## Run Configuration
MUSICAL_LANE: {config.lane}{vibe_line}
BATCH_SIZE: {config.batch_size}
MAX_ARTIST_PLAYS (purge threshold used): {config.max_artist_plays}
MAX_UNIQUE_TRACKS (purge threshold used): {config.max_unique_tracks}

## Anchor Pool — {len(anchor_pool.tracks)} tracks
These are the highest-played tracks from non-saturated artists after all purges.
They define taste adjacency. Do NOT recommend them.

{anchor_lines}

## Collision Memory — previously recommended (never re-recommend)
{collision_lines}

## Artist Blacklist — absolute exclusions (never eligible)
{blacklist_lines}

## Saturated Artists — user knows these extensively; do NOT recommend any track by them
These artists were purged from the anchor pool because the user has heard them too many times.
Their individual tracks are NOT listed but are part of the user's listening history.
Never recommend any song by any artist on this list.
{purged_lines}

## Known Tracks — already in the user's listening history (never recommend)
These are specific songs the user has already heard, sorted by play count descending.
TWO absolute rules apply:
  1. Do NOT recommend any track listed here, regardless of artist name formatting.
  2. Do NOT recommend any song whose TITLE appears in this list, even by a different
     artist. If the user knows "Song to the Siren" via This Mortal Coil, do not
     recommend the Tim Buckley original or any other recording of it — the song is
     familiar regardless of who performed it. Covers are not discoveries.
     EXCEPTION: two songs that share a title but have different songwriters (e.g.
     "Elephant" by Tame Impala and "Elephant" by Jason Isbell) are genuinely
     different songs and may both be considered.
{known_lines}

---
Execute the full DCSv2 pipeline (CMP → TGE → score → anti-collapse) for musical lane: \
**{config.lane}**

Generate exactly {config.batch_size} recommendations (or fewer if discovery space is exhausted).
Output the CSV block only.
"""


# ---------- Claude API call ----------

def call_claude(
    config: RunConfig,
    anchor_pool: AnchorPool,
    state: ProjectState,
    client: Optional[anthropic.Anthropic] = None,
) -> tuple[str, int]:
    if client is None:
        client = anthropic.Anthropic()

    user_message = build_user_message(anchor_pool, state, config)

    response = client.messages.create(
        model=config.model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens


# ---------- Response parsing ----------

def _extract_csv_block(raw: str) -> str:
    """Pull CSV content out of a possible markdown code fence."""
    # Try fenced block first
    match = re.search(r"```(?:csv)?\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: find first line that looks like a header
    lines = raw.strip().splitlines()
    for i, line in enumerate(lines):
        if re.match(r'^"?Artist"?\s*,', line, re.IGNORECASE):
            return "\n".join(lines[i:])
    return raw  # last resort: pass everything


def _safe_float(s: str) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(s.strip())))
    except (ValueError, AttributeError):
        return None


def parse_response(raw_text: str) -> list[Recommendation]:
    csv_block = _extract_csv_block(raw_text)
    reader = csv.DictReader(io.StringIO(csv_block))

    recs: list[Recommendation] = []
    for row in reader:
        artist = (row.get("Artist") or "").strip().strip('"')
        track = (row.get("Track") or "").strip().strip('"')

        if not artist or not track:
            continue
        if artist.upper() == "NOTE":
            continue  # discovery-exhausted marker

        recs.append(Recommendation(
            artist=artist,
            track=track,
            dcs_score=_safe_float(row.get("DCS_Score", "")),
            cls_score=row.get("CLS", "").strip(),
            cms_score=row.get("CMS", "").strip(),
            mes_score=row.get("MES", "").strip(),
            rationale=(row.get("Rationale") or "").strip().strip('"'),
        ))

    return recs


# ---------- Normalization helpers ----------

# Strip ALL parenthetical/bracketed content — covers (Remastered), [Live],
# (Live at Madison Square Garden, 2003), (2019 Mix), etc.
_PARENS_RE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]", re.IGNORECASE)

# Strip standalone 4-digit years anywhere in the string (e.g. "Song Name 2019")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Strip "remastered" / "remaster" wherever it appears — not a different song
_REMASTER_RE = re.compile(r"\bre-?master(ed)?\b", re.IGNORECASE)

# Strip leading articles that vary between CSV and Claude output
_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def _normalise(s: str) -> str:
    """
    Aggressively normalise a string for fuzzy known-track matching.
    Case-insensitive. Strips parentheticals, years, 'remastered',
    leading articles, and punctuation so that:
      'Lonely Boy (Remastered 2022)' == 'Lonely Boy'
      'Live at MSG, 2019'            == 'Live at MSG'  (year stripped)
      'The Black Keys'               == 'Black Keys'
    """
    s = s.strip().lower()
    s = _PARENS_RE.sub("", s)       # strip ALL (...) and [...] blocks
    s = _REMASTER_RE.sub("", s)     # strip standalone "remastered" outside brackets
    s = _YEAR_RE.sub("", s)         # strip standalone 4-digit years
    s = _ARTICLE_RE.sub("", s)      # strip leading "The ", "A ", "An "
    s = re.sub(r"[^\w\s]", "", s)   # strip remaining punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _make_known_set(known_tracks: frozenset) -> set[tuple[str, str]]:
    """Build a normalised lookup set from the raw known_tracks frozenset."""
    return {(_normalise(a), _normalise(t)) for a, t in known_tracks}


# ---------- Post-filter against state ----------

def filter_against_state(
    recs: list[Recommendation],
    state: ProjectState,
    known_tracks: frozenset = frozenset(),
    known_titles: frozenset = frozenset(),
) -> list[Recommendation]:
    """
    Safety net: remove anything Claude missed.
    Checks against:
      1. Collision memory (previously recommended tracks)
      2. Blacklist (explicitly excluded artists)
      3. known_tracks (artist + title) — fuzzy normalised match
      4. known_titles (title only, any artist) — blocks covers:
         if "Song to the Siren" is known via This Mortal Coil,
         the Tim Buckley original is also blocked.
    """
    known_normalised = _make_known_set(known_tracks)
    known_titles_normalised = {_normalise(t) for t in known_titles}
    filtered = []
    for r in recs:
        if state.is_blacklisted(r.artist):
            continue
        if state.in_collision_memory(r.artist, r.track):
            continue
        norm_key = (_normalise(r.artist), _normalise(r.track))
        if norm_key in known_normalised:
            continue
        if _normalise(r.track) in known_titles_normalised:
            continue   # title known under a different artist — cover blocked
        filtered.append(r)
    return filtered


# ---------- Main entry point ----------

def get_recommendations(
    config: RunConfig,
    anchor_pool: AnchorPool,
    state: ProjectState,
    client: Optional[anthropic.Anthropic] = None,
) -> RecommendationResult:
    print(f"Calling Claude ({config.model}) for lane: {config.lane}...")
    raw_text, tokens = call_claude(config, anchor_pool, state, client)

    recs = parse_response(raw_text)
    recs = filter_against_state(
        recs, state,
        known_tracks=anchor_pool.known_tracks,
        known_titles=anchor_pool.known_titles or frozenset(),
    )

    print(f"  {len(recs)} recommendations parsed | {tokens:,} tokens used")
    return RecommendationResult(
        recommendations=recs,
        raw_response=raw_text,
        tokens_used=tokens,
        model=config.model,
    )
