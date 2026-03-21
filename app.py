"""
Music Discovery Agent — Streamlit Interface
Run locally: streamlit run app.py
API key:     set ANTHROPIC_API_KEY env var, or add to .streamlit/secrets.toml
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

import streamlit as st

from config import LANES, RunConfig
from history import AnchorPool, AnchorPoolTooSmallError, process_history
from lastfm_api import fetch_pool_lane_fits
from recommender import RecommendationResult, get_recommendations
from state import ProjectState, load_state, save_state

# ─────────────────────────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Music Discovery Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  Session state initialisation
# ─────────────────────────────────────────────────────────────

STEP_LABELS = [
    "Overview",
    "History",
    "Profile",
    "Parameters",
    "Blacklist",
    "Lane",
    "Anchor Pool",
    "Run",
    "Export",
]

def _init():
    defaults = {
        "step":             1,   # 1 = Overview
        "source":           None,      # 'lastfm' | 'spotify'
        "tmp_csv_path":     None,      # Path — temp file for uploaded CSV
        "csv_filename":     "",        # display name
        "project":          "",
        "state_obj":        None,      # ProjectState (loaded from disk)
        "params": {
            "max_artist_plays":  200,
            "max_unique_tracks": 10,
            "min_track_plays":   8,
            "batch_size":        20,
            "anchor_pool_size":  50,
        },
        "anchor_pool":      None,      # AnchorPool object
        "lane_fit_pool":    None,      # enriched tracks with lane_fit scores
        "lane":             "",
        "vibe_focus":       "",
        "new_blacklist":    [],        # artists to add this session
        "output_filename":  "",
        "result":           None,      # RecommendationResult
        # Spotify export state — reset each session; never pre-filled
        "spotify_token":    None,      # token dict from Spotipy exchange
        "spotify_user":     None,      # dict from sp.current_user()
        "spotify_push_result": None,   # result dict from push_playlist()
        "spotify_auth_error":  "",     # human-readable error string
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─────────────────────────────────────────────────────────────
#  Spotify OAuth callback — runs on every script execution
#  Must come before any step rendering so that the ?code= param
#  is captured immediately when Spotify redirects back.
# ─────────────────────────────────────────────────────────────

def _spotify_secrets() -> tuple[str, str, str]:
    """Return (client_id, client_secret, redirect_uri) from secrets, or ('','','')."""
    try:
        cid  = st.secrets.get("SPOTIFY_CLIENT_ID",     "") or ""
        sec  = st.secrets.get("SPOTIFY_CLIENT_SECRET", "") or ""
        # Local:  http://127.0.0.1:8501/   (Spotify accepts loopback IP for dev)
        # Cloud:  https://music-discovery-agent.streamlit.app/
        ruri = st.secrets.get("SPOTIFY_REDIRECT_URI",  "") or ""
        return cid, sec, ruri
    except Exception:
        return "", "", ""


def _handle_spotify_callback():
    """
    Detect Spotify's authorization redirect (?code=...&state=profile_name),
    exchange the code for a token, and restore the user's session state.
    The Spotify OAuth `state` parameter carries the profile name so we can
    reload the saved results file after the mandatory page refresh.
    """
    code    = st.query_params.get("code")
    profile = st.query_params.get("state", "")   # we encode profile name here

    if not code:
        return
    if st.session_state.get("spotify_token"):
        # Already exchanged — just clear the URL params
        st.query_params.clear()
        return

    client_id, client_secret, redirect_uri = _spotify_secrets()
    if not client_id or not client_secret:
        st.query_params.clear()
        return

    try:
        from spotify_push import make_oauth, exchange_code as _exc, make_client, get_current_user
        oauth      = make_oauth(client_id, client_secret, redirect_uri)
        token_info = _exc(oauth, code)
        sp         = make_client(token_info["access_token"])
        user       = get_current_user(sp)
        st.session_state.spotify_token = token_info
        st.session_state.spotify_user  = user
    except Exception as e:
        st.session_state.spotify_auth_error = str(e)
        st.query_params.clear()
        return

    # Restore session from disk so the Export step has something to work with
    if profile:
        st.session_state.project = profile
        sf = _state_dir() / f"{profile}.json"
        if sf.exists():
            st.session_state.state_obj = load_state(sf)
        last = _load_last_result(profile)
        if last:
            st.session_state.result = last
            st.session_state.lane   = last._lane_hint  # stored at save time

    st.query_params.clear()
    st.session_state.step = 9   # jump straight to Export


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _go(step: int):
    st.session_state.step = step
    st.rerun()

def _state_dir() -> Path:
    return Path.home() / ".music-agent"

def _state_file(project: str) -> Path:
    return _state_dir() / f"{project}.json"

def _save_upload(uploaded_file) -> Path:
    """Write Streamlit UploadedFile to a named temp file; return its path."""
    suffix = Path(uploaded_file.name).suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return Path(tmp.name)

def _lane_slug(lane: str) -> str:
    return lane.split("(")[0].strip().lower().replace(" ", "_").replace("/", "_")

def _default_output() -> str:
    p = st.session_state.get("project", "user")
    l = _lane_slug(st.session_state.get("lane", "recommendations"))
    run = 1
    state = st.session_state.get("state_obj")
    if state:
        run = state.run_count + 1
    return f"{p}_{l}_run{run}.csv"

def _save_last_result(profile: str, result, lane: str) -> None:
    """
    Persist the recommendation result to disk so it survives the Spotify OAuth
    page redirect (which resets Streamlit session state).
    """
    import json
    path = _state_dir() / f"{profile}_last_result.json"
    _state_dir().mkdir(parents=True, exist_ok=True)
    data = {
        "profile": profile,
        "lane": lane,
        "recommendations": [
            {
                "artist":    r.artist,
                "track":     r.track,
                "dcs_score": r.dcs_score,
                "cls_score": r.cls_score,
                "cms_score": r.cms_score,
                "mes_score": r.mes_score,
                "rationale": r.rationale,
            }
            for r in result.recommendations
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_last_result(profile: str):
    """
    Load the last saved result from disk. Returns a RecommendationResult-like
    object (with a ._lane_hint attribute carrying the lane name), or None.
    """
    import json
    from recommender import Recommendation, RecommendationResult
    path = _state_dir() / f"{profile}_last_result.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        recs = [
            Recommendation(
                artist    = r["artist"],
                track     = r["track"],
                dcs_score = r.get("dcs_score"),
                cls_score = r.get("cls_score"),
                cms_score = r.get("cms_score"),
                mes_score = r.get("mes_score"),
                rationale = r.get("rationale", ""),
            )
            for r in data.get("recommendations", [])
        ]
        result = RecommendationResult(
            recommendations=recs,
            raw_response="(restored from disk)",
            tokens_used=0,
            model="(restored)",
        )
        result._lane_hint = data.get("lane", "")
        return result
    except Exception:
        return None


def _recs_to_soundiiz_csv(recs: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    for r in recs:
        w.writerow([r.artist, r.track])
    return buf.getvalue()

def _recs_to_detail_csv(recs: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "Artist", "Track", "DCS_Score",
        "CLS (Co-Listening Strength): How strongly the song connects to your anchors across many listeners",
        "CMS (Credible Mention Strength): Quality, depth, and independence of critical references",
        "MES (Mechanics Evidence Score): Only applied when critics describe what happens in the song (tension, release, arrangement, motion — never inferred)",
        "Rationale",
    ])
    for r in recs:
        w.writerow([r.artist, r.track,
                    r.dcs_score or "", r.cls_score or "",
                    r.cms_score or "", r.mes_score or "",
                    r.rationale])
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
#  Progress bar
# ─────────────────────────────────────────────────────────────

def _progress_bar():
    step = st.session_state.step
    total = len(STEP_LABELS)
    cols = st.columns(total)
    for i, (col, label) in enumerate(zip(cols, STEP_LABELS), start=1):
        if i < step:
            col.markdown(f"<div style='text-align:center;color:#888;font-size:12px'>✓ {label}</div>",
                         unsafe_allow_html=True)
        elif i == step:
            col.markdown(f"<div style='text-align:center;color:#1DB954;font-weight:700;font-size:12px'>{label}</div>",
                         unsafe_allow_html=True)
        else:
            col.markdown(f"<div style='text-align:center;color:#ccc;font-size:12px'>{label}</div>",
                         unsafe_allow_html=True)
    pct = (step - 1) / (total - 1) if total > 1 else 0
    st.progress(pct)
    st.markdown("---")


# ─────────────────────────────────────────────────────────────
#  Step 1 — Overview
# ─────────────────────────────────────────────────────────────

_HERO_IMAGE = Path(__file__).parent / "assets" / "overview_hero.png"

def step_overview():
    import base64

    # ── Encode image as base64 so it works inside the iframe ──
    img_b64 = ""
    if _HERO_IMAGE.exists():
        img_b64 = base64.b64encode(_HERO_IMAGE.read_bytes()).decode()

    # ── Right-side content as HTML (no Streamlit widgets — button handled below) ──
    right_html = """
    <style>
      body { margin: 0; padding: 0; font-family: sans-serif; background: transparent; }
      .rp { padding: 2.2rem 2.4rem 1rem 2.4rem; color: #111; }
      .rp h1 { font-size: 1.55rem; font-weight: 700; margin: 0 0 0.15rem 0; color: #C0392B; }
      .rp .sub { font-size: 0.82rem; color: #555; margin-bottom: 1rem; }
      .rp hr { border: none; border-top: 1px solid #ccc; margin: 0.8rem 0 1rem 0; }
      .rp h4 { font-size: 0.92rem; font-weight: 700; color: #C0392B;
                margin: 1.3rem 0 0.3rem 0; text-transform: uppercase; letter-spacing: .04em; }
      .rp p, .rp li { font-size: 0.88rem; line-height: 1.6; color: #111; margin: 0.2rem 0; }
      .rp ul { padding-left: 1.2rem; margin: 0.3rem 0; }
      .rp b { color: #000; font-weight: 700; }
      .rp i { color: #333; }
      .rp .score { display:flex; gap:0.5rem; align-items:flex-start; margin:0.4rem 0; }
      .rp .badge { background:#C0392B; color:#fff; font-size:0.72rem; font-weight:700;
                   padding:0.15rem 0.45rem; border-radius:4px; white-space:nowrap;
                   margin-top:0.1rem; flex-shrink:0; }
    </style>
    <div class="rp">
      <h1>Music Discovery Agent</h1>
      <p class="sub">A behavior-grounded, song-level discovery engine</p>
      <hr>
      <p>This AI Agent turns an LLM into a <b>song-level music discovery engine</b>.
      It takes a full listening history (Last.fm or Spotify CSV) and recommends
      new songs that are statistically and critically adjacent to your taste —
      without falling back on genre tags, musical scenes, or similar artists.</p>
      <p>The goal is <b>discovery</b> of new artists and songs, not <i>reinforcement</i>
      of existing listening behaviors.</p>

      <h4>Mechanics</h4>
      <p>Adjust <b>parameters</b> to filter your history into an <b>anchor pool</b> that
      inspires the new playlist. Select a <b>musical lane</b> (e.g., Melancholy Balladry,
      Propulsive Guitar Rock, Atmospheric / Texture-First) to drive the feeling, emotion,
      and tone of the output. Add optional <b>vibe descriptors</b> (riffs, hooks, soaring
      chorus, flow, etc.) to further tailor results.</p>

      <h4>Decision Science</h4>
      <p>Recommendations are scored using a weighted blend of three signals:</p>
      <div class="score"><span class="badge">TGE 45%</span>
        <p><b>Taste Graph Expansion</b> — Song-to-song co-listening neighborhoods
        drawn from Last.fm's track.getSimilar API</p></div>
      <div class="score"><span class="badge">CMS 35%</span>
        <p><b>Credible Mention Strength</b> — Explicit critic references from
        Pitchfork, NPR, Rolling Stone, Stereogum, AllMusic, NYT</p></div>
      <div class="score"><span class="badge">MES 20%</span>
        <p><b>Mechanics Evidence Score</b> — Applied only when critics describe
        what actually happens in the song: tension, release, arrangement, motion</p></div>

      <h4>Fine-Point Features</h4>
      <ul>
        <li><b>Blacklist</b> excludes artists of no interest</li>
        <li><b>Anti-collapse rule</b> prevents any single artist, era, or subgenre
            from dominating the output</li>
        <li><b>Collision memory</b> ensures no track is ever recommended twice</li>
        <li><b>Freshness rotation</b> varies the anchor pool across runs for genuine divergence</li>
        <li>Fully <b>self-hosted</b> — nothing stored in the cloud</li>
      </ul>

      <h4>Outputs</h4>
      <ul>
        <li><b>Soundiiz CSV</b> — import directly to Spotify, Apple Music, Tidal,
            or any major streamer via soundiiz.com</li>
        <li><b>Rationale CSV</b> — explains each recommendation using production style,
            melodic structure, co-listening data, or the critic's own words</li>
      </ul>
    </div>
    """

    # ── Full-page split layout injected as CSS + HTML ──────────
    # Left panel is position:sticky so the image stays fixed while the
    # right panel content scrolls naturally with the Streamlit page.
    st.markdown(f"""
    <style>
      /* Remove default block padding for overview only */
      section.main .block-container {{
          padding-top: 1rem !important;
          padding-bottom: 0 !important;
      }}
      .ov-wrap {{
          display: flex;
          width: 100%;
          min-height: 88vh;
          gap: 0;
      }}
      /* Left panel — sticky so image stays fixed while right scrolls */
      .ov-left {{
          flex: 0 0 57%;
          max-width: 57%;
          position: sticky;
          top: 70px;          /* clears the Streamlit top bar */
          height: calc(100vh - 80px);
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
          padding: 1rem;
          box-sizing: border-box;
      }}
      .ov-left img {{
          max-height: calc(100vh - 120px);
          max-width: 100%;
          object-fit: contain;
          display: block;
      }}
      /* Thin divider */
      .ov-div {{
          flex: 0 0 1px;
          background: #333;
          align-self: stretch;
      }}
      /* Right panel — scrolls naturally with the page */
      .ov-right {{
          flex: 1;
          min-width: 0;
          overflow-y: visible;
      }}
    </style>
    <div class="ov-wrap">
      <div class="ov-left">
        {'<img src="data:image/png;base64,' + img_b64 + '" alt="Music Discovery Agent">' if img_b64 else '<p style="color:#666">Image not found</p>'}
      </div>
      <div class="ov-div"></div>
      <div class="ov-right">
        {right_html}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Get started button — rendered by Streamlit below the HTML ──
    st.markdown("<div style='margin-left:58%;margin-top:1rem;'>", unsafe_allow_html=True)
    if st.button("Get started  →", type="primary", use_container_width=False, key="overview_start"):
        _go(2)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Step 2 — Listening History
# ─────────────────────────────────────────────────────────────

def step_history():
    st.title("Music Discovery Agent")
    st.subheader("Step 2 — Listening History")
    st.caption("Upload your full listening history CSV. Choose the service you use.")

    left, mid, right = st.columns([10, 0.08, 10], gap="small")

    with left:
        st.markdown("#### Last.fm")
        st.markdown(
            "If you've previously set up a Last.fm scrobbler to track your streaming activity, "
            "export your complete lifetime history at "
            "[lastfm.ghan.nl/export](https://lastfm.ghan.nl/export/)\n\n"
            "⏱ Usually ready in under a minute"
        )
        lastfm_file = st.file_uploader(
            "Upload Last.fm CSV",
            type=["csv"],
            key="lastfm_upload",
            help="Export from lastfm.ghan.nl/export",
        )
        if lastfm_file:
            st.success(f"✓  {lastfm_file.name}  ({lastfm_file.size:,} bytes)")

    with mid:
        st.markdown(
            "<div style='border-left:1px solid #333;height:360px;'></div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### Spotify")
        st.markdown(
            "You can always request your data directly from Spotify at "
            "[spotify.com/us/account/privacy](https://www.spotify.com/us/account/privacy/)\n\n"
            "⏱ Could take up to **5 days** to export data from the past year, "
            "and up to **30 days** to export your listening data across your full lifetime history"
        )
        spotify_file = st.file_uploader(
            "Upload Spotify CSV",
            type=["csv"],
            key="spotify_upload",
            help="Export from spotify.com/us/account/privacy",
        )
        if spotify_file:
            st.success(f"✓  {spotify_file.name}  ({spotify_file.size:,} bytes)")

    # Resolve which side has a file
    uploaded = None
    source   = None
    if lastfm_file and spotify_file:
        st.warning("Please upload from one service at a time.")
    elif lastfm_file:
        uploaded, source = lastfm_file, "lastfm"
    elif spotify_file:
        uploaded, source = spotify_file, "spotify"

    st.markdown("")
    if st.button("Next →", type="primary", disabled=(uploaded is None)):
        if st.session_state.tmp_csv_path:
            try:
                os.unlink(st.session_state.tmp_csv_path)
            except OSError:
                pass
        st.session_state.tmp_csv_path = _save_upload(uploaded)
        st.session_state.csv_filename  = uploaded.name
        st.session_state.source        = source
        _go(3)


# ─────────────────────────────────────────────────────────────
#  Step 3 — Listener Profile
# ─────────────────────────────────────────────────────────────

def step_profile():
    st.subheader("Step 3 — Listener Profile")
    st.caption("The profile name stores your blacklist and session history. Use one profile per listener.")

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        project = st.text_input(
            "Profile name",
            value=st.session_state.project or "",
            placeholder="e.g.  alice  or  marlonrando",
            help="Letters, numbers, and underscores only. Spaces are converted to underscores.",
        )
        project = project.strip().replace(" ", "_")

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back"):
                _go(2)
        with col_next:
            if st.button("Next →", type="primary", disabled=not project):
                st.session_state.project = project
                st.session_state.state_obj = load_state(_state_file(project))
                _go(4)

    with right:
        if project:
            sf = _state_file(project)
            if sf.exists():
                s = load_state(sf)
                st.info(
                    f"**Existing profile found**\n\n"
                    f"- Previous runs: **{s.run_count}**\n"
                    f"- Collision memory: **{len(s.collision_memory)}** tracks\n"
                    f"- Blacklisted artists: **{len(s.blacklist)}**\n"
                    f"- Last lane: **{s.last_lane or '—'}**"
                )
            else:
                st.info("New profile — will be created on first run.")


# ─────────────────────────────────────────────────────────────
#  Step 3 — Discovery Parameters
# ─────────────────────────────────────────────────────────────

def step_parameters():
    st.subheader("Step 4 — Discovery Parameters")
    st.info(
        "**Goal:** Recommend new songs by under-explored or new-to-user artists, "
        "driven by song-level taste adjacency — not 'similar artists.'\n\n"
        "These parameters control how aggressively familiar artists are purged before "
        "building your taste signal pool, and how many recommendations to generate."
    )

    p = st.session_state.params

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Artist purge thresholds**")
        max_plays = st.number_input(
            "Plays per artist before purging as 'saturated'",
            min_value=10, max_value=10000, step=10,
            value=p["max_artist_plays"],
            help="Artists at or above this total are purged as 'saturated.' At 200, an artist you've played 200+ times is considered fully known, and the agent will not recommend new songs by them.",
        )
        max_tracks = st.number_input(
            "Unique tracks per artist before purging",
            min_value=1, max_value=200, step=1,
            value=p["max_unique_tracks"],
            help="If you've heard this many distinct tracks by an artist, they're purged regardless of total play count — breadth of familiarity is itself a signal of saturation.",
        )
        min_track_plays = st.number_input(
            "Min plays per track to qualify for anchor pool",
            min_value=1, max_value=100, step=1,
            value=p["min_track_plays"],
            help="A track must have been played at least this many times to be considered a taste signal. Tracks below this threshold are too weak to reliably indicate preference.",
        )

    with right:
        st.markdown("**Recommendation settings**")
        batch_size = st.number_input(
            "Songs to recommend this run",
            min_value=5, max_value=50, step=5,
            value=p["batch_size"],
            help="A smaller batch size will render more quickly / easily for your Claude.",
        )
        anchor_size = st.number_input(
            "Anchor pool size",
            min_value=10, max_value=100, step=5,
            value=p["anchor_pool_size"],
            help="The size of your anchor pool will determine how much Claude recommendations adhere to your artist and song history, or branch out to new artists and songs.",
        )

    st.markdown("")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back"):
            _go(3)
    with col_next:
        if st.button("Next →", type="primary"):
            st.session_state.params = {
                "max_artist_plays":  int(max_plays),
                "max_unique_tracks": int(max_tracks),
                "min_track_plays":   int(min_track_plays),
                "batch_size":        int(batch_size),
                "anchor_pool_size":  int(anchor_size),
            }
            st.session_state.anchor_pool = None   # force recompute
            _go(5)


# ─────────────────────────────────────────────────────────────
#  Step 4 — Blacklist
# ─────────────────────────────────────────────────────────────

def step_blacklist():
    st.subheader("Step 5 — Blacklist")
    st.caption(
        "Artists listed here will be excluded from your anchor pool "
        "and will never appear in recommendations. "
        "Enter one artist per line — or comma-separated."
    )

    state: ProjectState = st.session_state.state_obj or ProjectState()

    left, right = st.columns(2, gap="large")

    with left:
        raw = st.text_area(
            "Artists to exclude",
            height=240,
            placeholder="The Beatles, Bob Dylan, Fleetwood Mac",
            label_visibility="collapsed",
        )

    with right:
        existing = sorted(state.blacklist)
        if existing:
            st.markdown(f"**Currently blacklisted — {len(existing)} artists**")
            st.dataframe({"Artist": existing}, use_container_width=True, height=220, hide_index=True)
        else:
            st.markdown("**Currently blacklisted**")
            st.caption("No artists blacklisted yet for this profile.")

    st.markdown("")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back"):
            _go(4)
    with col_next:
        if st.button("Next →  (select lane)", type="primary"):
            additions = []
            if raw.strip():
                for line in raw.replace(",", "\n").splitlines():
                    a = line.strip()
                    if a:
                        additions.append(a)
            if additions:
                st.session_state.new_blacklist = additions
                # Apply to in-memory state now so anchor pool is built with these exclusions
                state.add_to_blacklist(additions)
                st.session_state.state_obj = state
            st.session_state.anchor_pool = None  # force recompute with updated blacklist
            _go(6)


# ─────────────────────────────────────────────────────────────
#  Step 5 — Anchor Pool Preview
# ─────────────────────────────────────────────────────────────

def step_anchor_pool():
    st.subheader("Step 7 — Anchor Pool Preview")
    lane = st.session_state.lane or "—"
    st.caption(
        f"Lane: **{lane}** — these tracks are your strongest taste signals after purging saturated artists. "
        "They drive adjacency scoring for this lane — they are **not** candidates for recommendation."
    )
    st.caption(
        "Utility purge — removes artists whose name contains keywords like "
        '"white noise", "sleep", "binaural", "unknown", "untitled", etc.'
    )

    state: ProjectState = st.session_state.state_obj or ProjectState()
    p = st.session_state.params

    if st.session_state.anchor_pool is None:
        with st.spinner("Parsing your listening history…"):
            try:
                pool = process_history(
                    csv_path=Path(st.session_state.tmp_csv_path),
                    source=st.session_state.source,
                    max_artist_plays=p["max_artist_plays"],
                    max_unique_tracks=p["max_unique_tracks"],
                    top_tracks_per_artist=2,
                    anchor_pool_size=p["anchor_pool_size"],
                    blacklist=state.blacklist,
                    collision_memory=state.collision_memory,
                    min_track_plays=p["min_track_plays"],
                    freshness_penalties=state.freshness_penalties(),
                )
                st.session_state.anchor_pool = pool
            except AnchorPoolTooSmallError as e:
                st.error(f"**Anchor pool too small.** {e}")
                st.info("Go back and loosen the purge thresholds.")
                if st.button("← Adjust parameters"):
                    _go(4)
                return
            except Exception as e:
                st.error(f"**Error parsing CSV:** {e}")
                if st.button("← Back"):
                    _go(6)
                return

    pool: AnchorPool = st.session_state.anchor_pool

    # Stats row
    c1, c2, c3 = st.columns(3)
    c1.metric("Total plays", f"{pool.total_scrobbles:,}")
    c2.metric("Artists purged", f"{pool.total_artists - pool.eligible_count:,}")
    c3.metric("Anchor tracks", len(pool.tracks))

    st.markdown("")

    import pandas as pd

    # ── Last.fm lane fit enrichment ───────────────────────────
    lastfm_key = os.environ.get("LASTFM_API_KEY") or ""
    if not lastfm_key:
        try:
            lastfm_key = st.secrets.get("LASTFM_API_KEY", "")
        except Exception:
            lastfm_key = ""

    # Re-fetch if pool changed or lane changed (lane_fit_pool keyed by lane)
    cached_fit = st.session_state.lane_fit_pool
    cached_lane = (cached_fit[0].get("_lane") if cached_fit else None)
    if lastfm_key and (cached_fit is None or cached_lane != lane):
        st.markdown("**Checking genre tags via Last.fm…**")
        prog = st.progress(0)
        status = st.empty()

        def _on_progress(i, total):
            prog.progress(i / total)
            status.caption(f"Fetching tags: {i} / {total} tracks")

        enriched = fetch_pool_lane_fits(pool.tracks, lane, lastfm_key, _on_progress)
        # After lane-fit enrichment the list is already sorted High → Medium → blank,
        # sub-sorted by plays. Now truncate to the configured anchor_pool_size so that
        # lane-relevant tracks win slots over high-play off-lane tracks.
        target_size = st.session_state.get("anchor_pool_size", 50)
        enriched = enriched[:target_size]
        # Update the anchor pool tracks to the truncated lane-sorted list
        pool.tracks = [{k: v for k, v in e.items() if not k.startswith("_")} for e in enriched]
        st.session_state.anchor_pool = pool
        # Tag each entry with the lane so we can detect stale cache
        for e in enriched:
            e["_lane"] = lane
        st.session_state.lane_fit_pool = enriched
        prog.empty()
        status.empty()
    elif lastfm_key and cached_fit:
        enriched = cached_fit
    else:
        # No Last.fm key — fall back to plays-sorted, no lane fit column
        enriched = sorted(pool.tracks, key=lambda r: -r["plays"])[:st.session_state.get("anchor_pool_size", 50)]

    df = pd.DataFrame(enriched)
    if lastfm_key and "lane_fit" in df.columns:
        df = df[["artist", "track", "plays", "lane_fit"]]
        df.columns = ["Artist", "Track", "Plays", "Lane Fit"]
        caption = (
            "**Lane Fit** sourced from Last.fm crowd-sourced track tags — "
            "High / Medium = strong genre overlap with selected lane. "
            "Blank = insufficient tag data (Claude still uses the track as a taste signal)."
        )
    else:
        df = df[["artist", "track", "plays"]]
        df.columns = ["Artist", "Track", "Plays"]
        caption = "Sorted by play count. Add a LASTFM_API_KEY to secrets.toml to enable lane fit scoring."

    df = df.drop(columns=["_lane"], errors="ignore")
    df.index = range(1, len(df) + 1)
    st.dataframe(df, use_container_width=True, height=400)
    st.caption(caption)

    st.markdown("")
    st.markdown(
        "**Does this pool look right?** "
        "If you see artists you barely know, go back and tighten the thresholds. "
        "If too many familiar artists are still present, lower the play cap."
    )

    col_back, col_adj, col_next = st.columns([1, 1, 2])
    with col_back:
        if st.button("← Back"):
            _go(6)
    with col_adj:
        if st.button("Adjust thresholds"):
            st.session_state.anchor_pool = None
            _go(4)
    with col_next:
        if st.button("Looks good  →  Review & Run", type="primary"):
            _go(8)


# ─────────────────────────────────────────────────────────────
#  Step 5 — Lane + Vibe Focus
# ─────────────────────────────────────────────────────────────

def step_lane():
    st.subheader("Step 6 — Musical Lane")
    st.caption(
        "The lane sets the filtering logic — what kinds of songs the engine targets. "
        "It's not about which artists, it's about how songs behave. "
        "Your anchor pool will be previewed through this lens."
    )

    state: ProjectState = st.session_state.state_obj or ProjectState()
    lane_names = [l[0] for l in LANES]
    lane_descs = {l[0]: l[1] for l in LANES}

    # Default to last used lane if available
    default_lane = st.session_state.lane or state.last_lane or lane_names[0]
    default_idx = lane_names.index(default_lane) if default_lane in lane_names else 0

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        selected_lane = st.radio(
            "Select a lane",
            options=lane_names,
            index=default_idx,
            label_visibility="collapsed",
        )

    with right:
        st.markdown("#### About this lane")
        st.info(lane_descs.get(selected_lane, ""))

        st.markdown("#### Vibe focus  *(optional)*")
        vibe = st.text_input(
            "Narrow the lane further",
            value=st.session_state.vibe_focus or "",
            placeholder='e.g. "fingerpicked guitar"  or  "female vocals"  or  "pre-1980"',
            label_visibility="collapsed",
        )

    st.markdown("")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back"):
            _go(5)
    with col_next:
        if st.button("Next →  (build anchor pool)", type="primary"):
            st.session_state.lane = selected_lane
            st.session_state.vibe_focus = vibe
            st.session_state.anchor_pool = None    # force recompute if lane changed
            st.session_state.lane_fit_pool = None  # force re-fetch lane fits
            _go(7)


# ─────────────────────────────────────────────────────────────
#  Step 7 — Summary + Run
# ─────────────────────────────────────────────────────────────

def step_run():
    st.subheader("Step 8 — Review & Generate")

    p = st.session_state.params
    state: ProjectState = st.session_state.state_obj or ProjectState()
    new_bl = st.session_state.new_blacklist
    pool: AnchorPool = st.session_state.anchor_pool

    # ── Settings summary ──────────────────────────────────────
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Run configuration**")
        st.markdown(f"""
| | |
|---|---|
| Profile | `{st.session_state.project}` |
| History | `{st.session_state.csv_filename}` |
| Source | {st.session_state.source.capitalize()} |
| Lane | {st.session_state.lane} |
| Vibe focus | {st.session_state.vibe_focus or '—'} |
| Purge | ≥ {p['max_artist_plays']} plays or ≥ {p['max_unique_tracks']} unique tracks |
| Batch size | {p['batch_size']} songs |
| Anchor pool | {len(pool.tracks)} tracks |
""")

    with right:
        st.markdown("**Profile status**")
        st.markdown(f"""
| | |
|---|---|
| Previous runs | {state.run_count} |
| Collision memory | {len(state.collision_memory)} tracks |
| Blacklisted | {len(state.blacklist)} artists |
| Adding to blacklist | {len(new_bl)} artists |
""")
        if new_bl:
            st.caption("New exclusions: " + ", ".join(new_bl))

    # ── API key check ─────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY") or ""
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            api_key = ""
    if not api_key:
        st.warning(
            "**ANTHROPIC_API_KEY not found.** "
            "Set it as an environment variable or add it to `.streamlit/secrets.toml`."
        )

    st.markdown("---")

    # ── Already have a result? Show it ───────────────────────
    if st.session_state.result:
        _show_results(st.session_state.result)
        return

    # ── Generate button ───────────────────────────────────────
    col_back, _, col_run = st.columns([1, 1, 2])
    with col_back:
        if st.button("← Back"):
            _go(7)  # Anchor Pool
    with col_run:
        run_disabled = not api_key
        if st.button(
            "🎵  Generate Recommendations",
            type="primary",
            disabled=run_disabled,
            use_container_width=True,
        ):
            _execute_run()


def _execute_run():
    p = st.session_state.params
    state: ProjectState = st.session_state.state_obj or ProjectState()
    pool: AnchorPool = st.session_state.anchor_pool

    # Apply blacklist additions
    if st.session_state.new_blacklist:
        state.add_to_blacklist(st.session_state.new_blacklist)
        save_state(state, _state_file(st.session_state.project))

    config = RunConfig(
        history_csv=Path(st.session_state.tmp_csv_path),
        source=st.session_state.source,
        lane=st.session_state.lane,
        project=st.session_state.project,
        vibe_focus=st.session_state.vibe_focus,
        max_artist_plays=p["max_artist_plays"],
        max_unique_tracks=p["max_unique_tracks"],
        anchor_pool_size=p["anchor_pool_size"],
        batch_size=p["batch_size"],
    )

    with st.spinner(f"Asking Claude to discover music for lane: **{config.lane}**…"):
        try:
            result = get_recommendations(config, pool, state)
        except Exception as e:
            st.error(f"**Claude API error:** {e}")
            return

    if not result.recommendations:
        st.warning("No recommendations returned. Discovery space may be exhausted for this lane.")
        return

    # Persist state
    state.add_recommendations([
        {"artist": r.artist, "track": r.track}
        for r in result.recommendations
    ])
    state.run_count += 1
    state.last_lane = config.lane

    # Record anchor pool artists for freshness penalty on next run
    pool = st.session_state.get("anchor_pool")
    if pool:
        state.record_anchor_pool([t["artist"] for t in pool.tracks])

    save_state(state, _state_file(st.session_state.project))
    st.session_state.state_obj = state
    st.session_state.result = result

    # Save to disk so results survive the Spotify OAuth redirect
    try:
        _save_last_result(st.session_state.project, result, config.lane)
    except Exception:
        pass  # non-fatal — Spotify export will just show a note

    st.rerun()


def _show_results(result: RecommendationResult):
    recs = result.recommendations
    st.success(f"**{len(recs)} recommendations generated** — {result.tokens_used:,} tokens used")

    # ── Reset track selection when a new result arrives ───────────────────
    result_id = id(result)
    if st.session_state.get("_result_id") != result_id:
        st.session_state["_result_id"] = result_id
        for i in range(len(recs)):
            st.session_state[f"track_sel_{i}"] = True

    # ── Checkbox track table ───────────────────────────────────────────────
    st.markdown("**Select tracks to include in your playlist** — all selected by default:")

    # Header row
    h0, h1, h2, h3, h4 = st.columns([0.35, 1.8, 2.2, 0.55, 5])
    h0.markdown("<small>**✓**</small>", unsafe_allow_html=True)
    h1.markdown("<small>**Artist**</small>", unsafe_allow_html=True)
    h2.markdown("<small>**Track**</small>", unsafe_allow_html=True)
    h3.markdown("<small>**DCS**</small>", unsafe_allow_html=True)
    h4.markdown("<small>**Rationale**</small>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:4px 0 6px 0'>", unsafe_allow_html=True)

    for i, r in enumerate(recs):
        c0, c1, c2, c3, c4 = st.columns([0.35, 1.8, 2.2, 0.55, 5])
        with c0:
            st.checkbox("", key=f"track_sel_{i}", label_visibility="collapsed")
        c1.markdown(f"<small>{r.artist}</small>", unsafe_allow_html=True)
        c2.markdown(f"<small>{r.track}</small>", unsafe_allow_html=True)
        c3.markdown(
            f"<small>{r.dcs_score:.2f}</small>" if r.dcs_score else "<small>—</small>",
            unsafe_allow_html=True,
        )
        c4.markdown(f"<small>{r.rationale}</small>", unsafe_allow_html=True)

    # Count selected
    selected_recs = [r for i, r in enumerate(recs) if st.session_state.get(f"track_sel_{i}", True)]
    n_sel = len(selected_recs)
    st.markdown(f"<small>*{n_sel} of {len(recs)} tracks selected*</small>", unsafe_allow_html=True)

    fname = _default_output()

    st.markdown("---")

    # ── 1. Send to Spotify (primary action) ───────────────────────────────
    if st.button("🎵  Send to Spotify  →", type="primary", use_container_width=True):
        # Store the filtered list so the Export step only pushes selected tracks
        st.session_state["_export_recs_override"] = selected_recs
        _go(9)

    st.markdown("")

    # ── 2. Soundiiz CSV (alternative streaming apps) ──────────────────────
    st.markdown(
        "If you'd like to create the playlist within another streaming app, download "
        "the playlist here and run it through Soundiiz to push to Apple Music, "
        "Amazon Music, Tidal, etc."
    )
    soundiiz_csv = _recs_to_soundiiz_csv(selected_recs)
    st.download_button(
        "⬇  Download Soundiiz CSV",
        data=soundiiz_csv,
        file_name=fname,
        mime="text/csv",
    )

    st.markdown("")

    # ── 3. Full Detail CSV ────────────────────────────────────────────────
    detail_csv = _recs_to_detail_csv(selected_recs)
    st.download_button(
        "⬇  Full Detail CSV",
        data=detail_csv,
        file_name=fname.replace(".csv", "_detail.csv"),
        mime="text/csv",
    )

    st.markdown("")
    if st.button("Run again  →"):
        st.session_state.result = None
        st.session_state.anchor_pool = None
        st.session_state.pop("_export_recs_override", None)
        _go(6)  # back to Lane selection


# ─────────────────────────────────────────────────────────────
#  Step 9 — Export (Spotify push + Soundiiz instructions)
# ─────────────────────────────────────────────────────────────

def step_export():
    from datetime import date

    st.subheader("Step 9 — Export Your Playlist")

    result = st.session_state.result
    if not result:
        st.warning("No recommendations found in this session. Please run the generator first.")
        if st.button("← Back to Run"):
            _go(8)
        return

    # Use the user-filtered track list if they unchecked any tracks on the results screen
    recs = st.session_state.get("_export_recs_override") or result.recommendations
    lane = st.session_state.get("lane") or getattr(result, "_lane_hint", "") or "Music Discovery"

    # ── Read Spotify credentials from secrets (never hard-code) ──────────
    client_id, client_secret, redirect_uri = _spotify_secrets()
    spotify_configured = bool(client_id and client_secret and redirect_uri)

    left, divider_col, right = st.columns([10, 0.08, 10], gap="small")

    # ══════════════════════════════════════════════════════════
    #  LEFT — Spotify
    # ══════════════════════════════════════════════════════════
    with left:
        st.markdown("### 🎧 Push to Spotify")

        if not spotify_configured:
            st.info(
                "**Spotify not configured.**\n\n"
                "To enable direct Spotify push, add these three keys to "
                "`.streamlit/secrets.toml` (local) and the Streamlit Cloud dashboard:\n\n"
                "```toml\n"
                "SPOTIFY_CLIENT_ID     = \"your_client_id\"\n"
                "SPOTIFY_CLIENT_SECRET = \"your_client_secret\"\n"
                "# Local:\n"
                "SPOTIFY_REDIRECT_URI  = \"http://127.0.0.1:8501/\"\n"
                "# Cloud (use this value in Streamlit Cloud dashboard):\n"
                "# SPOTIFY_REDIRECT_URI = \"https://music-discovery-agent.streamlit.app/\"\n"
                "```\n\n"
                "Register both redirect URIs at "
                "[developer.spotify.com/dashboard](https://developer.spotify.com/dashboard). "
                "Spotify accepts `http://127.0.0.1` (loopback IP) for local development."
            )

        elif st.session_state.get("spotify_push_result"):
            # ── Already pushed ────────────────────────────────────────────
            pr = st.session_state.spotify_push_result
            n_found = len(pr["found"])
            n_total = len(recs)
            st.success(
                f"**Playlist created in Spotify** — {n_found} of {n_total} tracks added"
            )
            st.markdown(
                f"### [🎵 Open \"{pr['playlist_name']}\" in Spotify]({pr['playlist_url']})",
                unsafe_allow_html=False,
            )

            if pr["not_found"]:
                st.markdown(f"**{len(pr['not_found'])} tracks not found on Spotify:**")
                for t in pr["not_found"]:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;• {t}", unsafe_allow_html=True)
                st.caption(
                    "These tracks may be unavailable in your region, "
                    "released under a different title on Spotify, or "
                    "not yet in Spotify's catalog."
                )

            st.markdown("")
            if st.button("Push another playlist", use_container_width=True):
                st.session_state.spotify_push_result = None
                st.session_state.spotify_token = None
                st.session_state.spotify_user  = None
                st.rerun()

        elif st.session_state.get("spotify_token"):
            # ── Connected — show playlist form ───────────────────────────
            user = st.session_state.spotify_user or {}
            display = user.get("display_name") or user.get("id") or "Spotify User"
            st.success(f"✓ Connected as **{display}**")
            st.markdown("")

            default_name = (
                f"Music Discovery — {lane} — {date.today().strftime('%b %Y')}"
            )
            playlist_name = st.text_input(
                "Playlist name",
                value=default_name,
                max_chars=100,
            )
            playlist_desc = st.text_input(
                "Description *(optional)*",
                value=f"Generated by Music Discovery Agent · {lane} lane · {len(recs)} tracks",
                max_chars=300,
                label_visibility="visible",
            )
            public = st.checkbox("Make playlist public", value=True)
            st.markdown("")

            if st.button(
                f"Create playlist with {len(recs)} tracks →",
                type="primary",
                use_container_width=True,
                disabled=not playlist_name.strip(),
            ):
                token_info = st.session_state.spotify_token
                user_id    = (st.session_state.spotify_user or {}).get("id", "")
                prog = st.progress(0)
                status_msg = st.empty()

                def _on_track_progress(i, total):
                    prog.progress(i / total)
                    status_msg.caption(f"Searching Spotify: {i} / {total} tracks…")

                try:
                    from spotify_push import make_client, push_playlist
                    sp = make_client(token_info["access_token"])
                    push_result = push_playlist(
                        sp=sp,
                        user_id=user_id,
                        name=playlist_name.strip(),
                        description=playlist_desc.strip(),
                        recs=recs,
                        public=public,
                        progress_callback=_on_track_progress,
                    )
                    prog.empty()
                    status_msg.empty()
                    st.session_state.spotify_push_result = push_result
                    st.rerun()
                except Exception as e:
                    prog.empty()
                    status_msg.empty()
                    st.error(f"**Spotify error:** {e}")

        else:
            # ── Not connected — show auth flow ───────────────────────────
            if st.session_state.get("spotify_auth_error"):
                st.error(
                    f"**Connection failed:** {st.session_state.spotify_auth_error}\n\n"
                    "Check your Spotify credentials in secrets.toml and try again."
                )
                st.session_state.spotify_auth_error = ""

            st.markdown(
                "Connect your Spotify account to create this playlist directly. "
                "You'll be prompted to log in — **no credentials are stored between sessions**."
            )
            st.markdown("")

            profile = st.session_state.get("project", "user")
            from spotify_push import make_oauth, get_auth_url
            oauth   = make_oauth(client_id, client_secret, redirect_uri)
            url     = get_auth_url(oauth, state=profile)   # state = profile name for session recovery

            st.markdown(
                f"""<a href="{url}" target="_self"
                style="display:inline-block;padding:0.55rem 1.4rem;background:#1DB954;
                       color:#000;font-weight:700;border-radius:6px;text-decoration:none;
                       font-size:0.95rem;">
                Connect to Spotify →
                </a>""",
                unsafe_allow_html=True,
            )
            st.caption(
                "Clicking Connect will briefly redirect you to Spotify's login page. "
                "Your recommendations are saved and will be ready when you return."
            )

    # ── Divider ──────────────────────────────────────────────────────────
    with divider_col:
        st.markdown(
            "<div style='border-left:1px solid #333;min-height:640px;'></div>",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════
    #  RIGHT — CSV download + Soundiiz instructions
    # ══════════════════════════════════════════════════════════
    with right:
        st.markdown("### 📂 Download & Import via Soundiiz")
        st.markdown(
            "Prefer a different streaming service? Download the Soundiiz CSV "
            "and use [soundiiz.com](https://soundiiz.com) — a free web tool that imports "
            "playlists to **Spotify, Apple Music, Amazon Music, Tidal, YouTube Music**, and more."
        )

        soundiiz_csv = _recs_to_soundiiz_csv(recs)
        fname = _default_output()
        st.download_button(
            "⬇  Download Soundiiz CSV",
            data=soundiiz_csv,
            file_name=fname,
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
        detail_csv = _recs_to_detail_csv(recs)
        st.download_button(
            "⬇  Download full detail CSV",
            data=detail_csv,
            file_name=fname.replace(".csv", "_detail.csv"),
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("")
        st.markdown("#### How to import via Soundiiz")
        st.markdown(
            """
1. **Download** the Soundiiz CSV above (two-column: Artist, Track)
2. Go to **[soundiiz.com](https://soundiiz.com)** and sign in or create a free account
3. In the top nav, click **Transfer → Import from file**
4. Choose **CSV** as the file format and upload the file you downloaded
5. Soundiiz will match your tracks — review the results
6. Select your **destination platform** (Spotify, Apple Music, Amazon Music, Tidal, etc.)
7. Click **Transfer** — your new playlist appears in the app of your choice

*Soundiiz's free tier supports up to 200 tracks per playlist.*
"""
        )

    # ── Back button ──────────────────────────────────────────────────────
    st.markdown("")
    if st.button("← Back to results"):
        _go(8)


# ─────────────────────────────────────────────────────────────
#  Sidebar — session info
# ─────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.markdown("### Session")
        if st.session_state.project:
            st.markdown(f"**Profile:** `{st.session_state.project}`")
        if st.session_state.csv_filename:
            st.markdown(f"**File:** `{st.session_state.csv_filename}`")
        if st.session_state.lane:
            st.markdown(f"**Lane:** {st.session_state.lane}")

        state = st.session_state.state_obj
        if state:
            st.markdown(f"**Runs:** {state.run_count}")
            st.markdown(f"**Collision memory:** {len(state.collision_memory)}")
            st.markdown(f"**Blacklisted:** {len(state.blacklist)}")

        st.markdown("---")
        st.markdown("### Jump to step")
        for i, label in enumerate(STEP_LABELS, 1):
            if st.button(label, key=f"sidebar_{i}", use_container_width=True):
                _go(i)

        st.markdown("---")
        if st.button("↺  Start over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ─────────────────────────────────────────────────────────────
#  Router
# ─────────────────────────────────────────────────────────────

_handle_spotify_callback()   # must run before sidebar/steps so OAuth redirect is captured
_sidebar()
_progress_bar()

step = st.session_state.step
if   step == 1: step_overview()
elif step == 2: step_history()
elif step == 3: step_profile()
elif step == 4: step_parameters()
elif step == 5: step_blacklist()
elif step == 6: step_lane()
elif step == 7: step_anchor_pool()
elif step == 8: step_run()
elif step == 9: step_export()
