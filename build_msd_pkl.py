#!/usr/bin/env python3
"""
build_msd_pkl.py
================
One-time build of msd_index.pkl from msd_summary_file.h5.
After this runs (~20 min), ingest_contributor.py Phase 3g
loads the pkl in <2 min instead of re-scanning the HDF5.

Usage:
    python3 build_msd_pkl.py
"""
import h5py, pickle, re, time
from pathlib import Path

H5_PATH  = Path.home() / ".music-agent/msd_summary_file.h5"
PKL_PATH = Path.home() / ".music-agent/msd_index.pkl"

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower()) if s else ""

def main():
    if not H5_PATH.exists():
        print(f"ERROR: HDF5 not found at {H5_PATH}")
        return

    print(f"[{time.strftime('%T')}] Opening HDF5 ({H5_PATH.stat().st_size/1e6:.0f} MB)…", flush=True)
    t0 = time.time()

    by_key         = {}
    by_artist_mbid = {}

    with h5py.File(H5_PATH, "r") as f:
        meta   = f["metadata"]["songs"]
        anal   = f["analysis"]["songs"]
        musics = f["musicbrainz"]["songs"]
        n = len(meta)
        print(f"[{time.strftime('%T')}] Loading {n:,} rows — bulk column reads…", flush=True)

        # Load entire columns at once — orders of magnitude faster than per-row access
        artists  = meta["artist_name"][:]
        titles   = meta["title"][:]
        mbids    = meta["artist_mbid"][:]
        tempos   = anal["tempo"][:]
        loudness = anal["loudness"][:]
        keys     = anal["key"][:]
        modes    = anal["mode"][:]
        tsigs    = anal["time_signature"][:]
        years    = musics["year"][:]

        print(f"[{time.strftime('%T')}] Columns loaded — building index…", flush=True)

        for i in range(n):
            try:
                artist = artists[i]
                title  = titles[i]
                if isinstance(artist, bytes): artist = artist.decode("utf-8", errors="ignore")
                if isinstance(title,  bytes): title  = title.decode("utf-8",  errors="ignore")
                nk = f"{norm(artist)}|||{norm(title)}"

                year = int(years[i]) if years[i] > 0 else None

                entry = {
                    "msd_tempo":          round(float(tempos[i]), 3),
                    "msd_loudness":       round(float(loudness[i]), 3),
                    "msd_key":            int(keys[i]),
                    "msd_mode":           int(modes[i]),
                    "msd_time_signature": int(tsigs[i]),
                    "msd_year":           year,
                    "msd_source":         "msd_exact",
                    "msd_match_score":    1.0,
                }
                by_key[nk] = entry

                mbid = mbids[i]
                if isinstance(mbid, bytes): mbid = mbid.decode("utf-8", errors="ignore")
                if mbid:
                    by_artist_mbid.setdefault(mbid, []).append(nk)

            except Exception:
                pass

            if i % 100_000 == 0 and i > 0:
                print(f"[{time.strftime('%T')}]   {i:,}/{n:,}  ({100*i/n:.0f}%)", flush=True)

    print(f"[{time.strftime('%T')}] Parsed {len(by_key):,} tracks. Saving pkl…", flush=True)
    with open(PKL_PATH, "wb") as fout:
        pickle.dump({"by_key": by_key, "by_artist_mbid": by_artist_mbid}, fout, protocol=4)

    elapsed = (time.time() - t0) / 60
    size_mb = PKL_PATH.stat().st_size / 1e6
    print(f"[{time.strftime('%T')}] Done — {size_mb:.1f} MB pkl written in {elapsed:.1f} min", flush=True)
    print(f"[{time.strftime('%T')}] Path: {PKL_PATH}", flush=True)

if __name__ == "__main__":
    main()
