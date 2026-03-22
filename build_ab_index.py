"""
build_ab_index.py — One-time AcousticBrainz local index builder.

Downloads each of the 30 AcousticBrainz high-level dump files (tar.zst),
streams through them, extracts data for our 9,361 target MBIDs, and stores
the acoustic features in a local SQLite database.

Index lives at: ~/.music-agent/acousticbrainz.db
State tracker:  ~/.music-agent/ab_build_state.json  (which files are done)

Usage:
    python3 build_ab_index.py                        # run all 30 files
    python3 build_ab_index.py --files 0 5            # only files 0-5
    python3 build_ab_index.py --resume               # skip already-done files
    python3 build_ab_index.py --report               # show DB stats, no download
    python3 build_ab_index.py --no-delete            # keep .tar.zst after processing

Estimated time:    ~50–90 min total (depends on download speed)
Peak disk usage:   ~1–2 GB per file (streaming; deleted after each)
SQLite DB size:    ~5–15 MB for ~9k tracks

AcousticBrainz was retired in 2022; the dump covers recordings through mid-2022.
Roughly 60–70% of our MBIDs should be found in the dump.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sqlite3
import ssl
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import zstandard as zstd


# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path.home() / ".music-agent"
DB_PATH     = BASE_DIR / "acousticbrainz.db"
STATE_PATH  = BASE_DIR / "ab_build_state.json"
TARGETS_PATH = BASE_DIR / "target_mbids.json"

DUMP_BASE = (
    "https://data.metabrainz.org/pub/musicbrainz/acousticbrainz/dumps/"
    "acousticbrainz-highlevel-json-20220623/"
)
NUM_FILES = 30   # files 0..29

MBID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})-\d+\.json$",
    re.I,
)


# ── SQLite schema ─────────────────────────────────────────────────────────────

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS recordings (
    mbid                TEXT PRIMARY KEY,
    -- Mood probabilities (0.0 – 1.0)
    mood_happy          REAL,
    mood_sad            REAL,
    mood_aggressive     REAL,
    mood_relaxed        REAL,
    mood_acoustic       REAL,
    mood_party          REAL,
    mood_electronic     REAL,
    -- Other high-level
    danceability        REAL,   -- probability of "danceable"
    is_instrumental     INTEGER,  -- 1 = instrumental, 0 = voice
    voice_prob          REAL,   -- probability of "voice"
    is_tonal            INTEGER,  -- 1 = tonal, 0 = atonal
    tonal_prob          REAL,
    -- Rhythm / tonal
    bpm                 REAL,
    key_key             TEXT,
    key_scale           TEXT,   -- "major" or "minor"
    key_strength        REAL,
    -- Loudness
    average_loudness    REAL,
    -- Duration (seconds)
    duration            REAL,
    -- Dump file source
    dump_file_num       INTEGER
);

CREATE TABLE IF NOT EXISTS processed_files (
    file_num  INTEGER PRIMARY KEY,
    ts        TEXT,
    hits      INTEGER
);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(CREATE_SQL)
    conn.commit()
    return conn


# ── Feature extraction ────────────────────────────────────────────────────────

def _prob(data: dict, category: str, value: str) -> float | None:
    """Extract highlevel probability: highlevel[category]['all'][value]"""
    try:
        return float(data["highlevel"][category]["all"][value])
    except (KeyError, TypeError, ValueError):
        return None


def extract_features(data: dict, file_num: int) -> dict:
    hl  = data.get("highlevel", {})
    ll  = data.get("lowlevel", {})
    rhy = data.get("rhythm", {})
    ton = data.get("tonal", {})
    meta = data.get("metadata", {})

    # Instrumental vs voice
    vi = hl.get("voice_instrumental", {})
    vi_all  = vi.get("all", {})
    vi_val  = vi.get("value", "")
    voice_prob = vi_all.get("voice", None)
    is_instrumental = 1 if vi_val == "instrumental" else (0 if vi_val == "voice" else None)

    # Tonal vs atonal
    ta = hl.get("tonal_atonal", {})
    ta_val = ta.get("value", "")
    tonal_prob = ta.get("all", {}).get("tonal", None)
    is_tonal = 1 if ta_val == "tonal" else (0 if ta_val == "atonal" else None)

    # Duration
    duration = None
    try:
        duration = float(meta.get("audio_properties", {}).get("length", None))
    except (TypeError, ValueError):
        pass

    return {
        "mood_happy":       _prob(data, "mood_happy",       "happy"),
        "mood_sad":         _prob(data, "mood_sad",         "sad"),
        "mood_aggressive":  _prob(data, "mood_aggressive",  "aggressive"),
        "mood_relaxed":     _prob(data, "mood_relaxed",     "relaxed"),
        "mood_acoustic":    _prob(data, "mood_acoustic",    "acoustic"),
        "mood_party":       _prob(data, "mood_party",       "party"),
        "mood_electronic":  _prob(data, "mood_electronic",  "electronic"),
        "danceability":     _prob(data, "danceability",     "danceable"),
        "is_instrumental":  is_instrumental,
        "voice_prob":       voice_prob,
        "is_tonal":         is_tonal,
        "tonal_prob":       tonal_prob,
        "bpm":              rhy.get("bpm"),
        "key_key":          ton.get("key_key"),
        "key_scale":        ton.get("key_scale"),
        "key_strength":     ton.get("key_strength"),
        "average_loudness": ll.get("average_loudness"),
        "duration":         duration,
        "dump_file_num":    file_num,
    }


INSERT_SQL = """
INSERT OR REPLACE INTO recordings (
    mbid, mood_happy, mood_sad, mood_aggressive, mood_relaxed,
    mood_acoustic, mood_party, mood_electronic, danceability,
    is_instrumental, voice_prob, is_tonal, tonal_prob,
    bpm, key_key, key_scale, key_strength, average_loudness,
    duration, dump_file_num
) VALUES (
    :mbid, :mood_happy, :mood_sad, :mood_aggressive, :mood_relaxed,
    :mood_acoustic, :mood_party, :mood_electronic, :danceability,
    :is_instrumental, :voice_prob, :is_tonal, :tonal_prob,
    :bpm, :key_key, :key_scale, :key_strength, :average_loudness,
    :duration, :dump_file_num
)
"""


# ── Progress bar ──────────────────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 40) -> str:
    pct  = done / total if total else 0
    fill = int(pct * width)
    return f"[{'█' * fill}{'░' * (width - fill)}] {pct:5.1%}"


# ── Streaming download + parse ────────────────────────────────────────────────

_CHUNK = 1 << 20   # 1 MB read chunks for zstd


def process_dump_file(
    file_num: int,
    targets: dict,             # mbid → {artist, track, plays}
    conn: sqlite3.Connection,
    no_delete: bool = False,
) -> int:
    """Download, stream-decompress, extract target tracks. Returns hit count."""
    fname = f"acousticbrainz-highlevel-json-20220623-{file_num}.tar.zst"
    url   = DUMP_BASE + fname
    dl_path = Path("/tmp") / fname

    # ── 1. Download ────────────────────────────────────────────────────────
    print(f"\n[File {file_num:2d}/29] Downloading {fname}…", flush=True)
    t0 = time.time()
    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "curl/7.88.1", "Accept": "*/*"},
    )
    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dl_path, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = time.time() - t0
                speed   = downloaded / elapsed / 1e6  # MB/s
                if total_size:
                    pct = downloaded / total_size * 100
                    print(
                        f"\r  {pct:5.1f}%  {downloaded/1e6:.0f}/{total_size/1e6:.0f} MB"
                        f"  {speed:.1f} MB/s",
                        end="", flush=True,
                    )
    dl_secs = time.time() - t0
    print(f"\n  Download: {downloaded/1e6:.0f} MB in {dl_secs:.0f}s", flush=True)

    # ── 2. Stream-decompress + iterate tar ────────────────────────────────
    hits      = 0
    examined  = 0
    batch: list[dict] = []
    # Track MBIDs we've already collected this file (prefer first/offset-0 submission)
    seen_this_file: set[str] = set()

    dctx = zstd.ZstdDecompressor()
    print(f"  Scanning for {len(targets):,} target MBIDs…", end="", flush=True)
    t1 = time.time()

    with open(dl_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    examined += 1

                    m = MBID_RE.search(member.name)
                    if not m:
                        continue
                    mbid = m.group(1).lower()

                    if mbid not in targets:
                        continue
                    if mbid in seen_this_file:
                        continue  # keep first (offset-0) submission only

                    # Extract and parse this member
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    try:
                        data = json.loads(f.read().decode("utf-8"))
                    except Exception:
                        continue

                    feats = extract_features(data, file_num)
                    feats["mbid"] = mbid
                    batch.append(feats)
                    seen_this_file.add(mbid)
                    hits += 1

                    if examined % 200_000 == 0:
                        elapsed = time.time() - t1
                        print(
                            f"\r  {examined:,} files scanned, {hits} hits, {elapsed:.0f}s elapsed   ",
                            end="", flush=True,
                        )

    # Commit batch
    if batch:
        conn.executemany(INSERT_SQL, batch)
        conn.commit()

    elapsed = time.time() - t1
    print(f"\r  Scanned {examined:,} files → {hits} hits in {elapsed:.0f}s            ", flush=True)

    # ── 3. Record completion ───────────────────────────────────────────────
    conn.execute(
        "INSERT OR REPLACE INTO processed_files (file_num, ts, hits) VALUES (?,?,?)",
        (file_num, time.strftime("%Y-%m-%dT%H:%M:%S"), hits),
    )
    conn.commit()

    # ── 4. Delete archive ─────────────────────────────────────────────────
    if not no_delete:
        dl_path.unlink(missing_ok=True)
        print(f"  Archive deleted.", flush=True)
    else:
        print(f"  Archive kept at {dl_path}", flush=True)

    return hits


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(conn: sqlite3.Connection, targets: dict) -> None:
    total_hits, = conn.execute("SELECT COUNT(*) FROM recordings").fetchone()
    done_files,  = conn.execute("SELECT COUNT(*) FROM processed_files").fetchone()
    print(f"\n{'═'*60}")
    print(f"AcousticBrainz Index — {DB_PATH}")
    print(f"{'═'*60}")
    print(f"  Dump files processed : {done_files}/30")
    print(f"  Recordings indexed   : {total_hits:,}")
    print(f"  Target MBIDs         : {len(targets):,}")
    print(f"  Coverage             : {total_hits/len(targets)*100:.1f}%")
    print()

    if total_hits == 0:
        return

    # Mood stats
    for col in ["mood_happy","mood_sad","mood_aggressive","mood_relaxed","danceability"]:
        avg, = conn.execute(f"SELECT AVG({col}) FROM recordings WHERE {col} IS NOT NULL").fetchone()
        if avg is not None:
            print(f"  avg {col:20s}: {avg:.3f}")

    # Key distribution
    print()
    rows = conn.execute(
        "SELECT key_scale, COUNT(*) n FROM recordings "
        "WHERE key_scale IS NOT NULL GROUP BY key_scale ORDER BY n DESC"
    ).fetchall()
    for scale, n in rows:
        print(f"  {scale:10s}: {n:,} tracks")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build AcousticBrainz SQLite index.")
    ap.add_argument("--files", nargs=2, type=int, metavar=("START","END"),
                    help="Process only file numbers START..END (inclusive)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip files already marked complete in the DB")
    ap.add_argument("--report", action="store_true",
                    help="Print DB stats and exit (no downloading)")
    ap.add_argument("--no-delete", action="store_true",
                    help="Keep downloaded .tar.zst files after processing")
    ap.add_argument("--targets", default=str(TARGETS_PATH),
                    help=f"Path to target_mbids.json (default: {TARGETS_PATH})")
    ap.add_argument("--db", default=str(DB_PATH),
                    help=f"Path to output SQLite DB (default: {DB_PATH})")
    args = ap.parse_args()

    db_path  = Path(args.db)
    tgt_path = Path(args.targets)

    # ── Load targets ────────────────────────────────────────────────────
    if not tgt_path.exists():
        print(f"ERROR: {tgt_path} not found. Run the MBID extraction first.", file=sys.stderr)
        sys.exit(1)
    targets: dict = json.loads(tgt_path.read_text())
    target_set = set(targets.keys())
    print(f"Loaded {len(target_set):,} target MBIDs from {tgt_path}")

    # ── Init DB ─────────────────────────────────────────────────────────
    conn = init_db(db_path)

    if args.report:
        print_report(conn, targets)
        return

    # ── Determine which files to process ────────────────────────────────
    if args.files:
        file_range = list(range(args.files[0], args.files[1] + 1))
    else:
        file_range = list(range(NUM_FILES))

    if args.resume:
        done = {r[0] for r in conn.execute("SELECT file_num FROM processed_files")}
        file_range = [n for n in file_range if n not in done]
        print(f"Resuming: {len(done)} files already done, {len(file_range)} remaining.")

    if not file_range:
        print("Nothing to do.")
        print_report(conn, targets)
        return

    # ── Process files ────────────────────────────────────────────────────
    total_hits = 0
    overall_t0 = time.time()

    for i, file_num in enumerate(file_range):
        print(f"\n{'─'*60}")
        print(f"File {file_num} ({i+1}/{len(file_range)}) — "
              f"elapsed {(time.time()-overall_t0)/60:.1f} min")
        hits = process_dump_file(
            file_num,
            target_set,
            conn,
            no_delete=args.no_delete,
        )
        total_hits += hits

    print(f"\n{'═'*60}")
    print(f"All done in {(time.time()-overall_t0)/60:.1f} min. Total hits: {total_hits:,}")
    print_report(conn, targets)


if __name__ == "__main__":
    main()
