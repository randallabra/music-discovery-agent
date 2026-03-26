"""
Music Discovery Agent — Streamlit Interface v2
Run locally: streamlit run app_v2.py
API key:     set ANTHROPIC_API_KEY env var, or add to .streamlit/secrets.toml
"""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from config import RunConfig
from recommender import RecommendationResult, get_recommendations
from state import ProjectState, load_state, save_state

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

AGENT_DIR  = Path.home() / ".music-agent"
ASSETS_DIR = Path(__file__).parent / "assets"
CHARTS_DIR = ASSETS_DIR                     # charts live in music-agent/assets/

# Service logo paths (drop PNGs into music-agent/assets/ to activate)
LOGO_SPOTIFY = ASSETS_DIR / "logo_spotify.png"
LOGO_AMAZON  = ASSETS_DIR / "logo_amazon_music.png"
LOGO_APPLE   = ASSETS_DIR / "logo_apple_music.png"
LOGO_SOUNDIIZ = ASSETS_DIR / "logo_soundiiz.png"

TEMPERATURE_OPTIONS = [
    "Any",
    "Quiet",
    "Aggressive",
    "Electronic",
    "Happy",
    "Mid-Range",
]

TEMPERATURE_POCKET_MAP = {
    "Quiet":      "quiet",
    "Aggressive": "aggressive",
    "Electronic": "electronic",
    "Happy":      "happy",
    "Mid-Range":  "blob",
}

TEMPERATURE_DESCRIPTIONS = {
    "Any":        "No temperature filter — draw from the full anchor pool.",
    "Quiet":      "Introspective, low-energy, acoustic or spare arrangements. Folk, blues, singer-songwriter, quiet rock.",
    "Aggressive": "High-energy, loud, or emotionally intense. Grunge, hard rock, metal, punk, garage.",
    "Electronic": "Synthesizer-driven, rhythmically propulsive, or production-forward. New wave, synth-pop, IDM, post-punk with electronics.",
    "Happy":      "Bright, warm, energetic. Pop-rock, funk, soul, britpop, indie pop, jangle.",
    "Mid-Range":  "Acoustically ambiguous. Artists and tracks that straddle multiple temperatures — Pearl Jam, Pink Floyd, Neil Young.",
}

GENRE_OPTIONS = [
    "Any",
    "Classic Rock",
    "Grunge",
    "Alt-Country",
    "Folk",
    "Experimental",
    "Post-Punk",
    "Blues Rock",
    "Hard Rock",
    "Indie Pop",
    "Garage Rock",
    "Glam Rock",
    "New Wave",
    "Psychedelic Rock",
    "Electronic",
    "Britpop",
]

DECADE_OPTIONS = ["Any", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]

STEP_LABELS = [
    "Overview",
    "History",
    "Profile",
    "Parameters",
    "Blacklist",
    "Temperature",
    "Anchor Pool",
    "Run",
    "Export",
]

# ─────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Music Discovery Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "step":              1,
        "source":            None,
        "tmp_csv_path":      None,
        "csv_filename":      "",
        "project":           "",
        "state_obj":         None,
        "params": {
            "max_artist_plays":  200,
            "max_unique_tracks": 20,
            "min_unique_albums": 3,
            "min_track_plays":   3,
            "batch_size":        35,
            "anchor_pool_size":  20,
        },
        # New discovery selectors
        "temperature":       "Any",
        "genre":             "Any",
        "decade":            "Any",
        # Anchor pool
        "anchor_pool_tracks": None,  # list of dicts after user edits
        "anchor_pool_raw":    None,  # unfiltered list before user checkbox edits
        # Results
        "result":            None,
        # Spotify
        "spotify_token":     None,
        "spotify_user":      None,
        "spotify_push_result": None,
        "spotify_auth_error":  "",
        # UI flags
        "show_product_design": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─────────────────────────────────────────────────────────────
#  Track index loader
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_index(profile: str) -> dict:
    """Load pre-indexed track data for a given profile.
    Checks the app directory first (for cloud deployments), then ~/.music-agent/."""
    app_dir = Path(__file__).parent
    for path in [app_dir / f"{profile}_track_index.json",
                 AGENT_DIR / f"{profile}_track_index.json"]:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return {}

def _filter_index(
    index: dict,
    temperature: str,
    genre: str,
    decade: str,
    blacklist: set,
    min_plays: int,
) -> list[dict]:
    """Return tracks matching all active filters, sorted by play count desc."""
    pocket_filter = TEMPERATURE_POCKET_MAP.get(temperature)
    genre_filter  = genre.lower() if genre != "Any" else None
    decade_filter = decade if decade != "Any" else None

    results = []
    for key, t in index.items():
        if t["plays"] < min_plays:
            continue
        artist = t["artist"]
        if artist in blacklist:
            continue
        # Utility purge
        if any(w in artist.lower() for w in
               ("white noise","sleep","binaural","meditation","rain sounds",
                "nature sounds","baby sleep","lullaby machine")):
            continue
        if pocket_filter and t.get("pocket") != pocket_filter:
            continue
        if genre_filter and t.get("genre","").lower() != genre_filter:
            continue
        if decade_filter and t.get("decade") != decade_filter:
            continue
        results.append({
            "key":    key,
            "artist": artist,
            "track":  t["track"],
            "plays":  t["plays"],
            "pocket": t.get("pocket","blob"),
            "genre":  t.get("genre","Other"),
            "decade": t.get("decade","—"),
        })

    return sorted(results, key=lambda x: -x["plays"])

# ─────────────────────────────────────────────────────────────
#  Spotify helpers (unchanged from v1)
# ─────────────────────────────────────────────────────────────

def _spotify_secrets() -> tuple[str, str, str]:
    try:
        cid  = st.secrets.get("SPOTIFY_CLIENT_ID",     "") or ""
        sec  = st.secrets.get("SPOTIFY_CLIENT_SECRET", "") or ""
        ruri = st.secrets.get("SPOTIFY_REDIRECT_URI",  "") or ""
        return cid, sec, ruri
    except Exception:
        return "", "", ""


def _handle_spotify_callback():
    code    = st.query_params.get("code")
    profile = st.query_params.get("state", "")
    if not code:
        return
    if st.session_state.get("spotify_token"):
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
    if profile:
        st.session_state.project = profile
        sf = _state_dir() / f"{profile}.json"
        if sf.exists():
            st.session_state.state_obj = load_state(sf)
        last = _load_last_result(profile)
        if last:
            st.session_state.result = last
    st.query_params.clear()
    st.session_state.step = 9

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _go(step: int):
    st.session_state.step = step
    st.rerun()

def _state_dir() -> Path:
    return AGENT_DIR

def _state_file(project: str) -> Path:
    return _state_dir() / f"{project}.json"

def _default_output() -> str:
    p   = st.session_state.get("project", "user")
    tmp = st.session_state.get("temperature", "").replace(" / "," ").replace(" ","_").lower() or "discovery"
    g   = st.session_state.get("genre","").replace(" ","_").lower() or "any"
    run = 1
    state = st.session_state.get("state_obj")
    if state:
        run = state.run_count + 1
    return f"{p}_{tmp}_{g}_run{run}.csv"

def _save_last_result(profile: str, result, temperature: str) -> None:
    import json as _json
    path = _state_dir() / f"{profile}_last_result.json"
    _state_dir().mkdir(parents=True, exist_ok=True)
    data = {
        "profile":         profile,
        "temperature":     temperature,
        "recommendations": [
            {"artist": r.artist, "track": r.track,
             "dcs_score": r.dcs_score, "cls_score": r.cls_score,
             "cms_score": r.cms_score, "mes_score": r.mes_score,
             "rationale": r.rationale}
            for r in result.recommendations
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(data, f, indent=2)

def _load_last_result(profile: str):
    import json as _json
    from recommender import Recommendation, RecommendationResult
    path = _state_dir() / f"{profile}_last_result.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
        recs = [
            Recommendation(
                artist=r["artist"], track=r["track"],
                dcs_score=r.get("dcs_score"), cls_score=r.get("cls_score"),
                cms_score=r.get("cms_score"), mes_score=r.get("mes_score"),
                rationale=r.get("rationale",""),
            )
            for r in data.get("recommendations",[])
        ]
        result = RecommendationResult(
            recommendations=recs, raw_response="(restored from disk)",
            tokens_used=0, model="(restored)",
        )
        result._temperature_hint = data.get("temperature","")
        return result
    except Exception:
        return None

def _logo_img(path: Path, max_width: str = "140px") -> str:
    """Return a centred base64 <img> tag, or '' if the file is missing."""
    import base64 as _b64
    if path.exists():
        data = _b64.b64encode(path.read_bytes()).decode()
        ext  = path.suffix.lower().lstrip(".")
        mime = "image/png" if ext == "png" else f"image/{ext}"
        return (
            f'<img src="data:{mime};base64,{data}" '
            f'style="max-width:{max_width};height:auto;display:block;'
            f'margin:0 auto 0.6rem auto;">'
        )
    return ""


def _recs_to_soundiiz_csv(recs: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    for r in recs:
        w.writerow([r.artist, r.track])
    return buf.getvalue()

def _recs_to_detail_csv(recs: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Artist","Track","DCS_Score","CLS_Score","CMS_Score","MES_Score","Rationale"])
    for r in recs:
        w.writerow([r.artist, r.track, r.dcs_score or "", r.cls_score or "",
                    r.cms_score or "", r.mes_score or "", r.rationale])
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────
#  Progress bar
# ─────────────────────────────────────────────────────────────

def _progress_bar():
    step  = st.session_state.step
    total = len(STEP_LABELS)
    cols  = st.columns(total)
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
    img_b64 = ""
    if _HERO_IMAGE.exists():
        img_b64 = base64.b64encode(_HERO_IMAGE.read_bytes()).decode()

    right_html = """
    <style>
      body { margin:0; padding:0; font-family:sans-serif; background:transparent; }
      .rp { padding:2.2rem 2.4rem 1rem 2.4rem; color:#111; }
      .rp h1 { font-size:1.55rem; font-weight:700; margin:0 0 0.15rem 0; color:#C0392B; }
      .rp .sub { font-size:0.82rem; color:#555; margin-bottom:1rem; }
      .rp hr { border:none; border-top:1px solid #ccc; margin:0.8rem 0 1rem 0; }
      .rp h4 { font-size:0.92rem; font-weight:700; color:#C0392B;
               margin:1.3rem 0 0.3rem 0; text-transform:uppercase; letter-spacing:.04em; }
      .rp p, .rp li { font-size:0.88rem; line-height:1.6; color:#111; margin:0.2rem 0; }
      .rp ul { padding-left:1.2rem; margin:0.3rem 0; }
      .rp b { color:#000; font-weight:700; }
      .rp i { color:#333; }
      .rp .score { display:flex; gap:0.5rem; align-items:flex-start; margin:0.4rem 0; }
      .rp .badge { background:#C0392B; color:#fff; font-size:0.72rem; font-weight:700;
                   padding:0.15rem 0.45rem; border-radius:4px; white-space:nowrap;
                   margin-top:0.1rem; flex-shrink:0; }
    </style>
    <div class="rp">
      <h1>Music Discovery Agent</h1>
      <p class="sub">A behavior-grounded, song-level discovery engine</p>
      <hr>
      <p>This agent turns a deeply indexed listening history into a
      <b>song-level music discovery engine</b> — recommending new tracks that are
      statistically and critically adjacent to your taste without falling back on
      genre tags, musical scenes, or similar-artist shortcuts.</p>
      <p>The goal is <b>discovery</b> of new artists and songs, not <i>reinforcement</i>
      of existing listening behaviors.</p>

      <h4>How it works</h4>
      <p>Choose a <b>Temperature</b> (the emotional register you want — introspective,
      aggressive, danceable, upbeat, or open), a <b>Genre</b> lane, and an optional
      <b>Decade</b> to shape the anchor pool. Review and prune the pool, then let the
      engine build your playlist.</p>

      <h4>Decision Science</h4>
      <div class="score"><span class="badge">TGE 45%</span>
        <p><b>Taste Graph Expansion</b> — Song-to-song co-listening neighborhoods
        from Last.fm track.getSimilar</p></div>
      <div class="score"><span class="badge">CMS 35%</span>
        <p><b>Credible Mention Strength</b> — Explicit critic references from
        Pitchfork, NPR, Rolling Stone, Stereogum, AllMusic, NYT</p></div>
      <div class="score"><span class="badge">MES 20%</span>
        <p><b>Mechanics Evidence Score</b> — Only applied when critics describe
        what actually happens in the song: tension, release, arrangement, motion</p></div>

      <h4>Fine-Point Features</h4>
      <ul>
        <li><b>Blacklist</b> excludes artists of no interest</li>
        <li><b>Anti-collapse rule</b> prevents any single artist, era, or subgenre
            from dominating the output</li>
        <li><b>Collision memory</b> ensures no track is ever recommended twice</li>
        <li>Fully <b>self-hosted</b> — nothing stored in the cloud</li>
      </ul>

      <h4>Outputs</h4>
      <ul>
        <li><b>Direct Spotify push</b> — playlist created in your account instantly</li>
        <li><b>Soundiiz CSV</b> — import to Apple Music, Tidal, Amazon Music, etc.</li>
        <li><b>Detail CSV</b> — full rationale for every recommendation</li>
      </ul>
    </div>
    """

    st.markdown(f"""
    <style>
      section.main .block-container {{ padding-top:1rem !important; padding-bottom:0 !important; }}
      .ov-wrap {{ display:flex; width:100%; min-height:88vh; gap:0; }}
      .ov-left {{ flex:0 0 57%; max-width:57%; position:sticky; top:70px;
                  height:calc(100vh - 80px); display:flex; align-items:center;
                  justify-content:center; overflow:hidden; padding:1rem; box-sizing:border-box; }}
      .ov-left img {{ max-height:calc(100vh - 120px); max-width:100%;
                      object-fit:contain; display:block; }}
      .ov-div {{ flex:0 0 1px; background:#333; align-self:stretch; }}
      .ov-right {{ flex:1; min-width:0; overflow-y:visible; }}
    </style>
    <div class="ov-wrap">
      <div class="ov-left">
        {'<img src="data:image/png;base64,' + img_b64 + '" alt="Music Discovery Agent">' if img_b64 else '<p style="color:#666">Image not found</p>'}
      </div>
      <div class="ov-div"></div>
      <div class="ov-right">{right_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Two buttons positioned under the right-hand description panel
    _, bc1, bc2, _ = st.columns([57, 18, 22, 3])
    with bc1:
        if st.button("Get started  →", type="primary", key="overview_start"):
            _go(2)
    with bc2:
        if st.button("See Product Design  →", key="overview_design"):
            st.session_state.show_product_design = True
            st.rerun()

# ─────────────────────────────────────────────────────────────
#  Product Design — flowchart overview (not part of step flow)
# ─────────────────────────────────────────────────────────────

def step_product_design():
    import base64 as _b64

    st.title("Product Design")
    st.caption(
        "Three-stage pipeline: Data Intelligence  →  Anchor Pool  →  Playlist Recommendation"
    )
    st.markdown("---")

    chart_files = [
        CHARTS_DIR / "chart1_data_intelligence.png",
        CHARTS_DIR / "chart2_anchor_pool.png",
        CHARTS_DIR / "chart3_playlist_recommendation.png",
    ]
    chart_labels = [
        "1.  Building Data Intelligence",
        "2.  Building the Anchor Pool",
        "3.  Creating Playlist Recommendations",
    ]

    cols = st.columns(3, gap="medium")
    for col, path, label in zip(cols, chart_files, chart_labels):
        with col:
            st.markdown(f"**{label}**")
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.warning(f"Chart not found: `{path.name}`")

    st.markdown("---")
    btn_back, _, btn_start = st.columns([2, 6, 2])
    with btn_back:
        if st.button("← Back to Overview", key="design_back"):
            st.session_state.show_product_design = False
            st.rerun()
    with btn_start:
        if st.button("Get started  →", type="primary", key="design_start"):
            st.session_state.show_product_design = False
            _go(2)


# ─────────────────────────────────────────────────────────────
#  Step 2 — Listening History
# ─────────────────────────────────────────────────────────────

def step_history():
    st.title("Music Discovery Agent")
    st.subheader("Step 2 — Listening History")

    project = st.session_state.get("project","")
    if project:
        index_path = AGENT_DIR / f"{project}_track_index.json"
        if index_path.exists():
            idx = _load_index(project)
            st.info(
                f"**{project}** — {len(idx):,} anchor-eligible tracks indexed. "
                "Your anchor pool will draw from this index, but **please upload your "
                "listening history CSV below** so the recommendation engine can score "
                "candidates and avoid suggesting tracks you've already heard."
            )

    st.caption(
        "Upload your full listening history CSV. "
        "The anchor pool is drawn from your pre-indexed data; "
        "this CSV powers the recommendation candidate scoring."
    )

    left, mid, right = st.columns([10, 0.08, 10], gap="small")
    with left:
        st.markdown("#### Last.fm")
        st.markdown(
            "Export your complete lifetime history at "
            "[lastfm.ghan.nl/export](https://lastfm.ghan.nl/export/)\n\n"
            "⏱ Usually ready in under a minute"
        )
        lastfm_file = st.file_uploader("Upload Last.fm CSV", type=["csv"], key="lastfm_upload")
        if lastfm_file:
            st.success(f"✓  {lastfm_file.name}  ({lastfm_file.size:,} bytes)")
    with mid:
        st.markdown("<div style='border-left:1px solid #333;height:360px;'></div>",
                    unsafe_allow_html=True)
    with right:
        st.markdown("#### Spotify")
        st.markdown(
            "Request your data at "
            "[spotify.com/us/account/privacy](https://www.spotify.com/us/account/privacy/)\n\n"
            "⏱ Could take up to **30 days** for full lifetime history"
        )
        spotify_file = st.file_uploader("Upload Spotify CSV", type=["csv"], key="spotify_upload")
        if spotify_file:
            st.success(f"✓  {spotify_file.name}  ({spotify_file.size:,} bytes)")

    uploaded = None; source = None
    if lastfm_file and spotify_file:
        st.warning("Please upload from one service at a time.")
    elif lastfm_file:
        uploaded, source = lastfm_file, "lastfm"
    elif spotify_file:
        uploaded, source = spotify_file, "spotify"

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back"):
            _go(1)
    with col_next:
        if st.button("Next →", type="primary", disabled=(uploaded is None)):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(uploaded.read()); tmp.flush()
            st.session_state.tmp_csv_path = tmp.name
            st.session_state.csv_filename = uploaded.name
            st.session_state.source = source
            _go(3)

# ─────────────────────────────────────────────────────────────
#  Step 3 — Profile
# ─────────────────────────────────────────────────────────────

def step_profile():
    st.subheader("Step 3 — Listener Profile")
    st.caption("The profile name maps to your pre-indexed listening data, blacklist, and session history.")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        project = st.text_input(
            "Profile name",
            value=st.session_state.project or "",
            placeholder="e.g.  marlonrando",
        )
        project = project.strip().replace(" ","_")
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back"):
                _go(2)
        with col_next:
            if st.button("Next →", type="primary", disabled=not project):
                st.session_state.project   = project
                st.session_state.state_obj = load_state(_state_file(project))
                _go(4)
    with right:
        if project:
            index_path = AGENT_DIR / f"{project}_track_index.json"
            sf = _state_file(project)
            if index_path.exists():
                idx = _load_index(project)
                if sf.exists():
                    s = load_state(sf)
                    st.info(
                        f"**{project}** — indexed\n\n"
                        f"- Tracks in index: **{len(idx):,}**\n"
                        f"- Previous runs: **{s.run_count}**\n"
                        f"- Collision memory: **{len(s.collision_memory)}** tracks\n"
                        f"- Blacklisted artists: **{len(s.blacklist)}**"
                    )
                else:
                    st.info(f"**{project}** — {len(idx):,} indexed tracks. No run history yet.")
            else:
                st.warning(
                    f"No index found for **{project}**. "
                    "Run the ingestion workflow first, or upload a listening history CSV."
                )

# ─────────────────────────────────────────────────────────────
#  Step 4 — Parameters
# ─────────────────────────────────────────────────────────────

def step_parameters():
    st.subheader("Step 4 — Discovery Parameters")
    st.info(
        "**Goal:** Recommend new songs by under-explored or new-to-you artists, "
        "driven by song-level taste adjacency — not 'similar artists.'\n\n"
        "These settings control the size of your anchor pool and your output playlist."
    )
    p = st.session_state.params
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Anchor pool**")
        anchor_size = st.number_input(
            "Tracks in your anchor pool",
            min_value=5, max_value=100, step=5,
            value=p["anchor_pool_size"],
            help="These are your strongest taste signals after applying Temperature + Genre filters. Default 20.",
        )
        min_track_plays = st.number_input(
            "Min plays per track to qualify",
            min_value=1, max_value=50, step=1,
            value=p["min_track_plays"],
            help="A track must have been played at least this many times to be considered a taste signal.",
        )

    with right:
        st.markdown("**Recommendation output**")
        batch_size = st.number_input(
            "Tracks to recommend per batch",
            min_value=5, max_value=100, step=5,
            value=p["batch_size"],
            help="Default 35. You can uncheck any tracks before pushing to Spotify.",
        )

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back"):
            _go(3)
    with col_next:
        if st.button("Next →", type="primary"):
            st.session_state.params = {
                **p,
                "batch_size":        int(batch_size),
                "anchor_pool_size":  int(anchor_size),
                "min_track_plays":   int(min_track_plays),
            }
            st.session_state.anchor_pool_raw    = None
            st.session_state.anchor_pool_tracks = None
            _go(5)

# ─────────────────────────────────────────────────────────────
#  Step 5 — Blacklist
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
        raw = st.text_area("Artists to exclude", height=240,
                           placeholder="The Beatles, Bob Dylan, Fleetwood Mac",
                           label_visibility="collapsed")
    with right:
        existing = sorted(state.blacklist)
        if existing:
            st.markdown(f"**Currently blacklisted — {len(existing)} artists**")
            st.dataframe({"Artist": existing}, use_container_width=True,
                         height=220, hide_index=True)
        else:
            st.markdown("**Currently blacklisted**")
            st.caption("No artists blacklisted yet.")

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back"):
            _go(4)
    with col_next:
        if st.button("Next →  (set discovery filters)", type="primary"):
            if raw.strip():
                additions = [a.strip() for line in raw.replace(",","\n").splitlines()
                             if (a := line.strip())]
                if additions:
                    state.add_to_blacklist(additions)
                    st.session_state.state_obj = state
            st.session_state.anchor_pool_raw    = None
            st.session_state.anchor_pool_tracks = None
            _go(6)

# ─────────────────────────────────────────────────────────────
#  Step 6 — Discovery (Temperature + Genre + Decade)
# ─────────────────────────────────────────────────────────────

def step_discovery():
    st.subheader("Step 6 — Temperature")
    st.caption(
        "Set the emotional temperature, genre, and era for this run. "
        "These filters shape your anchor pool — the taste signals the engine builds from."
    )

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("#### Temperature")
        st.caption("The emotional register and energy level of the tracks you want to anchor from.")
        _t = st.session_state.temperature
        _t_idx = TEMPERATURE_OPTIONS.index(_t) if _t in TEMPERATURE_OPTIONS else 0
        temperature = st.radio(
            "Temperature",
            options=TEMPERATURE_OPTIONS,
            index=_t_idx,
            label_visibility="collapsed",
        )

        st.markdown("")
        st.markdown("#### Genre")
        st.caption("Top 15 genres by catalog size. 'Any' draws across all genres.")
        genre = st.radio(
            "Genre",
            options=GENRE_OPTIONS,
            index=GENRE_OPTIONS.index(st.session_state.genre)
                  if st.session_state.genre in GENRE_OPTIONS else 0,
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown("")
        st.markdown("#### Decade  *(~90% of catalog has year data)*")
        decade = st.radio(
            "Decade",
            options=DECADE_OPTIONS,
            index=DECADE_OPTIONS.index(st.session_state.decade)
                  if st.session_state.decade in DECADE_OPTIONS else 0,
            horizontal=True,
            label_visibility="collapsed",
        )

    with right:
        st.markdown("#### Temperature guide")
        for label, desc in TEMPERATURE_DESCRIPTIONS.items():
            style = "font-weight:700;color:#C0392B;" if label == temperature else "color:#555;"
            st.markdown(
                f"<p style='font-size:0.85rem;margin:0.4rem 0;'>"
                f"<span style='{style}'>{label}</span>"
                f"{'<br>' if label == temperature else ' — '}"
                f"<span style='color:#444;font-size:0.82rem;'>{desc}</span></p>",
                unsafe_allow_html=True,
            )

        # Live pool size preview
        if st.session_state.project:
            index = _load_index(st.session_state.project)
            state = st.session_state.state_obj or ProjectState()
            p = st.session_state.params
            preview = _filter_index(
                index, temperature, genre, decade,
                blacklist=set(state.blacklist),
                min_plays=p["min_track_plays"],
            )
            st.markdown("")
            if len(preview) >= p["anchor_pool_size"]:
                st.success(f"**{len(preview):,}** tracks match — {p['anchor_pool_size']} will be sampled for the anchor pool.")
            elif len(preview) > 0:
                st.warning(f"**{len(preview)}** tracks match — smaller than your anchor pool size of {p['anchor_pool_size']}. All will be used.")
            else:
                st.error("No tracks match this combination. Try broadening the filters.")

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back"):
            _go(5)
    with col_next:
        filters_changed = (
            temperature != st.session_state.temperature or
            genre       != st.session_state.genre       or
            decade      != st.session_state.decade
        )
        if st.button("Next →  (build anchor pool)", type="primary"):
            st.session_state.temperature = temperature
            st.session_state.genre       = genre
            st.session_state.decade      = decade
            if filters_changed:
                st.session_state.anchor_pool_raw    = None
                st.session_state.anchor_pool_tracks = None
            _go(7)

# ─────────────────────────────────────────────────────────────
#  Step 7 — Anchor Pool (checkbox-based)
# ─────────────────────────────────────────────────────────────

def step_anchor_pool():
    st.subheader("Step 7 — Anchor Pool")
    temperature = st.session_state.temperature
    genre       = st.session_state.genre
    decade      = st.session_state.decade

    filter_parts = [f"Temperature: **{temperature}**", f"Genre: **{genre}**"]
    if decade != "Any":
        filter_parts.append(f"Decade: **{decade}**")
    st.caption(
        "  ·  ".join(filter_parts) + "\n\n"
        "These tracks are your strongest taste signals. They drive adjacency scoring — "
        "they are **not** candidates for recommendation. Uncheck any you'd like to remove."
    )

    state: ProjectState = st.session_state.state_obj or ProjectState()
    p = st.session_state.params

    # Build pool if not cached
    if st.session_state.anchor_pool_raw is None:
        index = _load_index(st.session_state.project)
        if not index:
            st.error(f"No track index found for profile **{st.session_state.project}**.")
            if st.button("← Back"):
                _go(6)
            return

        filtered = _filter_index(
            index, temperature, genre, decade,
            blacklist=set(state.blacklist),
            min_plays=p["min_track_plays"],
        )
        # Cap at anchor_pool_size
        raw = filtered[:p["anchor_pool_size"]]
        st.session_state.anchor_pool_raw = raw

        # Initialise all checkboxes to True
        for i in range(len(raw)):
            key = f"anchor_sel_{i}"
            if key not in st.session_state:
                st.session_state[key] = True

    raw: list = st.session_state.anchor_pool_raw

    if not raw:
        st.warning(
            "No tracks matched your filters. "
            "Go back and try broadening Temperature, Genre, or Decade."
        )
        if st.button("← Adjust filters"):
            _go(6)
        return

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Tracks available", len(raw))
    n_checked = sum(1 for i in range(len(raw)) if st.session_state.get(f"anchor_sel_{i}", True))
    c2.metric("Selected for anchor pool", n_checked)
    c3.metric("Total plays (selected)",
              sum(raw[i]["plays"] for i in range(len(raw))
                  if st.session_state.get(f"anchor_sel_{i}", True)))

    st.markdown("")

    # Column headers
    h0, h1, h2, h3, h4, h5 = st.columns([0.4, 2.2, 2.8, 0.7, 1.4, 1.2])
    h0.markdown("<small>**✓**</small>",      unsafe_allow_html=True)
    h1.markdown("<small>**Artist**</small>", unsafe_allow_html=True)
    h2.markdown("<small>**Track**</small>",  unsafe_allow_html=True)
    h3.markdown("<small>**Plays**</small>",  unsafe_allow_html=True)
    h4.markdown("<small>**Genre**</small>",  unsafe_allow_html=True)
    h5.markdown("<small>**Decade**</small>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:4px 0 6px 0'>", unsafe_allow_html=True)

    for i, t in enumerate(raw):
        c0, c1, c2, c3, c4, c5 = st.columns([0.4, 2.2, 2.8, 0.7, 1.4, 1.2])
        with c0:
            st.checkbox("", key=f"anchor_sel_{i}", label_visibility="collapsed")
        c1.markdown(f"<small>{t['artist']}</small>",         unsafe_allow_html=True)
        c2.markdown(f"<small>{t['track']}</small>",          unsafe_allow_html=True)
        c3.markdown(f"<small>{t['plays']}</small>",          unsafe_allow_html=True)
        c4.markdown(f"<small>{t['genre']}</small>",          unsafe_allow_html=True)
        c5.markdown(f"<small>{t.get('decade','—')}</small>", unsafe_allow_html=True)

    st.markdown(f"<small>*{n_checked} of {len(raw)} tracks selected*</small>",
                unsafe_allow_html=True)

    st.markdown("")
    col_back, col_adj, col_next = st.columns([1, 1, 2])
    with col_back:
        if st.button("← Back"):
            _go(6)
    with col_adj:
        if st.button("Reset selections"):
            for i in range(len(raw)):
                st.session_state[f"anchor_sel_{i}"] = True
            st.rerun()
    with col_next:
        if st.button("Looks good  →  Review & Run", type="primary", disabled=(n_checked == 0)):
            # Persist the user's selected tracks
            selected = [raw[i] for i in range(len(raw))
                        if st.session_state.get(f"anchor_sel_{i}", True)]
            st.session_state.anchor_pool_tracks = selected
            _go(8)

# ─────────────────────────────────────────────────────────────
#  Step 8 — Review & Generate
# ─────────────────────────────────────────────────────────────

def step_run():
    st.subheader("Step 8 — Review & Generate")

    p      = st.session_state.params
    state  = st.session_state.state_obj or ProjectState()
    pool   = st.session_state.anchor_pool_tracks or []

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Run configuration**")
        st.markdown(f"""
| | |
|---|---|
| Profile | `{st.session_state.project}` |
| Temperature | {st.session_state.temperature} |
| Genre | {st.session_state.genre} |
| Decade | {st.session_state.decade} |
| Anchor pool | {len(pool)} tracks |
| Batch size | {p['batch_size']} songs |
""")
    with right:
        st.markdown("**Profile status**")
        st.markdown(f"""
| | |
|---|---|
| Previous runs | {state.run_count} |
| Collision memory | {len(state.collision_memory)} tracks |
| Blacklisted | {len(state.blacklist)} artists |
""")

    api_key = os.environ.get("ANTHROPIC_API_KEY","")
    if not api_key:
        try:    api_key = st.secrets.get("ANTHROPIC_API_KEY","")
        except: api_key = ""
    if not api_key:
        st.warning("**ANTHROPIC_API_KEY not found.** Add it to `.streamlit/secrets.toml`.")

    st.markdown("---")

    if st.session_state.result:
        _show_results(st.session_state.result)
        return

    col_back, _, col_run = st.columns([1, 1, 2])
    with col_back:
        if st.button("← Back"):
            _go(7)
    with col_run:
        if st.button("🎵  Generate Recommendations", type="primary",
                     disabled=not api_key or not pool, use_container_width=True):
            _execute_run(pool, state, p, api_key)


def _execute_run(pool, state, p, api_key):
    temperature = st.session_state.temperature
    genre       = st.session_state.genre

    # Require the uploaded CSV — it's the source of candidate scoring data
    csv_path_str = st.session_state.get("tmp_csv_path")
    if not csv_path_str:
        st.error(
            "**No listening history CSV found.** "
            "Please go back to Step 2 and upload your Last.fm or Spotify CSV. "
            "The CSV is needed to score recommendation candidates and avoid "
            "suggesting tracks you've already heard."
        )
        if st.button("← Back to Step 2"):
            _go(2)
        return

    csv_path = Path(csv_path_str)
    source   = st.session_state.get("source") or "lastfm"

    # Build a lane string for the existing recommender interface
    lane_str = f"{temperature}" + (f" · {genre}" if genre != "Any" else "")

    # Parse the full CSV to get known_tracks, known_tracks_by_plays, known_titles.
    # These drive the "do not recommend already-heard tracks" safety filters.
    from history import AnchorPool as _AP, parse_history_csv
    try:
        stats, total_scrobbles, _fmt = parse_history_csv(csv_path, source)
    except Exception as e:
        st.error(f"**Could not parse listening history CSV:** {e}")
        return

    # known_tracks: full set of (artist_lower, track_lower) from history
    known_tracks: frozenset = frozenset(
        (artist.strip().lower(), track.strip().lower())
        for artist, s in stats.items()
        for track, _ in s.top_tracks
    )

    # known_tracks_by_plays: top 300 by plays for the prompt's "do not recommend" list
    _all_ktp = sorted(
        (
            (artist, track, plays)
            for artist, s in stats.items()
            for track, plays in s.top_tracks
        ),
        key=lambda x: -x[2],
    )
    known_tracks_by_plays = _all_ktp[:300]

    # known_titles: normalised track titles cross-artist (cover dedup filter)
    _title_plays: dict[str, int] = {}
    for artist, s in stats.items():
        for track, plays in s.top_tracks:
            key = track.strip().lower()
            _title_plays[key] = _title_plays.get(key, 0) + plays
    known_titles: frozenset = frozenset(_title_plays.keys())

    # Build AnchorPool — tracks come from the pre-indexed JSON filter,
    # but known_* fields come from the full CSV history.
    ap = _AP.__new__(_AP)
    ap.tracks               = [{"artist": t["artist"], "track": t["track"], "plays": t["plays"]}
                                for t in pool]
    ap.purged_artists       = []
    ap.total_scrobbles      = total_scrobbles
    ap.total_artists        = len(stats)
    ap.eligible_count       = len({t["artist"] for t in pool})
    ap.known_tracks         = known_tracks
    ap.known_tracks_by_plays = known_tracks_by_plays
    ap.known_titles         = known_titles

    config = RunConfig(
        history_csv=csv_path,
        source=source,
        lane=lane_str,
        project=st.session_state.project,
        vibe_focus="",
        max_artist_plays=p["max_artist_plays"],
        max_unique_tracks=p["max_unique_tracks"],
        anchor_pool_size=len(pool),
        batch_size=p["batch_size"],
    )

    with st.spinner(f"Generating {p['batch_size']} recommendations…"):
        try:
            result = get_recommendations(config, ap, state)
        except Exception as e:
            st.error(f"**Claude API error:** {e}")
            return

    if not result.recommendations:
        st.warning("No recommendations returned.")
        return

    state.add_recommendations([{"artist": r.artist, "track": r.track}
                                for r in result.recommendations])
    state.run_count += 1
    state.last_lane  = lane_str
    state.record_anchor_pool([t["artist"] for t in pool])
    save_state(state, _state_file(st.session_state.project))
    st.session_state.state_obj = state
    st.session_state.result    = result

    try:
        _save_last_result(st.session_state.project, result, temperature)
    except Exception:
        pass
    st.rerun()


def _show_results(result: RecommendationResult):
    recs = result.recommendations
    batch_size = st.session_state.get("params", {}).get("batch_size", len(recs))
    n_backfill = sum(1 for r in recs if r.backfill)
    n_confident = len(recs) - n_backfill

    # Header summary
    summary = f"**{len(recs)} recommendations generated** — {result.tokens_used:,} tokens used"
    st.success(summary)

    if n_backfill:
        st.info(
            f"**{n_confident} tracks** met full confidence thresholds. "
            f"**{n_backfill} track{'s' if n_backfill != 1 else ''} backfilled** "
            f"to reach your requested batch of {batch_size} — shown with a ↓ marker below."
        )

    result_id = id(result)
    if st.session_state.get("_result_id") != result_id:
        st.session_state["_result_id"] = result_id
        for i in range(len(recs)):
            st.session_state[f"track_sel_{i}"] = True

    st.markdown("**Select tracks to include in your playlist** — all selected by default:")
    st.markdown(
        "<small style='color:#888'>DCS = Discovery Confidence Score &nbsp;|&nbsp; "
        "↓ = backfilled to meet batch size</small>",
        unsafe_allow_html=True,
    )

    h0, h1, h2, h3, h4 = st.columns([0.35, 1.8, 2.2, 0.55, 5])
    h0.markdown("<small>**✓**</small>",          unsafe_allow_html=True)
    h1.markdown("<small>**Artist**</small>",      unsafe_allow_html=True)
    h2.markdown("<small>**Track**</small>",       unsafe_allow_html=True)
    h3.markdown("<small>**DCS**</small>",         unsafe_allow_html=True)
    h4.markdown("<small>**Rationale**</small>",   unsafe_allow_html=True)
    st.markdown("<hr style='margin:4px 0 6px 0'>", unsafe_allow_html=True)

    for i, r in enumerate(recs):
        c0, c1, c2, c3, c4 = st.columns([0.35, 1.8, 2.2, 0.55, 5])
        with c0:
            st.checkbox("", key=f"track_sel_{i}", label_visibility="collapsed")
        artist_label = f"<small>{r.artist}</small>"
        track_label  = (
            f"<small style='color:#aaa'>{r.track} ↓</small>"
            if r.backfill else
            f"<small>{r.track}</small>"
        )
        c1.markdown(artist_label, unsafe_allow_html=True)
        c2.markdown(track_label,  unsafe_allow_html=True)
        c3.markdown(
            f"<small>{r.dcs_score:.2f}</small>" if r.dcs_score else "<small>—</small>",
            unsafe_allow_html=True)
        c4.markdown(f"<small>{r.rationale}</small>", unsafe_allow_html=True)

    selected_recs = [r for i, r in enumerate(recs)
                     if st.session_state.get(f"track_sel_{i}", True)]
    n_sel = len(selected_recs)
    st.markdown(f"<small>*{n_sel} of {len(recs)} tracks selected*</small>",
                unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🎵  Send to Spotify  →", type="primary", use_container_width=True):
        st.session_state["_export_recs_override"] = selected_recs
        _go(9)

    st.markdown("")
    soundiiz_csv = _recs_to_soundiiz_csv(selected_recs)
    st.download_button("⬇  Download Soundiiz CSV", data=soundiiz_csv,
                       file_name=_default_output(), mime="text/csv")
    st.markdown("")
    detail_csv = _recs_to_detail_csv(selected_recs)
    st.download_button("⬇  Full Detail CSV", data=detail_csv,
                       file_name=_default_output().replace(".csv","_detail.csv"),
                       mime="text/csv")
    st.markdown("")
    if st.button("Run again  →"):
        st.session_state.result             = None
        st.session_state.anchor_pool_raw    = None
        st.session_state.anchor_pool_tracks = None
        st.session_state.pop("_export_recs_override", None)
        _go(6)

# ─────────────────────────────────────────────────────────────
#  Step 9 — Export
# ─────────────────────────────────────────────────────────────

def step_export():
    from datetime import date

    st.subheader("Step 9 — Export Your Playlist")
    result = st.session_state.result
    if not result:
        st.warning("No recommendations found. Please run the generator first.")
        if st.button("← Back to Run"):
            _go(8)
        return

    recs = st.session_state.get("_export_recs_override") or result.recommendations
    temperature = (st.session_state.get("temperature")
                   or getattr(result, "_temperature_hint", "")
                   or "Music Discovery")

    client_id, client_secret, redirect_uri = _spotify_secrets()
    spotify_configured = bool(client_id and client_secret and redirect_uri)

    left, divider_col, right = st.columns([10, 0.08, 10], gap="small")

    with left:
        # ── Spotify logo header ──
        _sp_logo = _logo_img(LOGO_SPOTIFY, "150px")
        if _sp_logo:
            st.markdown(_sp_logo, unsafe_allow_html=True)
        else:
            st.markdown("### Push to Spotify")

        if not spotify_configured:
            st.info(
                "**Spotify not configured.**\n\n"
                "Add to `.streamlit/secrets.toml`:\n\n"
                "```toml\n"
                "SPOTIFY_CLIENT_ID     = \"your_client_id\"\n"
                "SPOTIFY_CLIENT_SECRET = \"your_client_secret\"\n"
                "SPOTIFY_REDIRECT_URI  = \"http://127.0.0.1:8501/\"\n"
                "```\n\n"
                "Register both URIs at "
                "[developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)."
            )
        elif st.session_state.get("spotify_push_result"):
            pr = st.session_state.spotify_push_result
            st.success(f"**Playlist created** — {len(pr['found'])} of {len(recs)} tracks added")
            st.markdown(f"### [Open \"{pr['playlist_name']}\" in Spotify]({pr['playlist_url']})")
            if pr["not_found"]:
                st.markdown(f"**{len(pr['not_found'])} tracks not found:**")
                for t in pr["not_found"]:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;• {t}", unsafe_allow_html=True)
            st.markdown("")
            if st.button("Push another playlist", use_container_width=True):
                st.session_state.spotify_push_result = None
                st.session_state.spotify_token       = None
                st.session_state.spotify_user        = None
                st.rerun()
        elif st.session_state.get("spotify_token"):
            user    = st.session_state.spotify_user or {}
            display = user.get("display_name") or user.get("id") or "Spotify User"
            st.success(f"✓ Connected as **{display}**")
            st.markdown("")
            default_name = f"Music Discovery — {temperature} — {date.today().strftime('%b %Y')}"
            playlist_name = st.text_input("Playlist name", value=default_name, max_chars=100)
            playlist_desc = st.text_input(
                "Description *(optional)*",
                value=f"Generated by Music Discovery Agent · {temperature} · {len(recs)} tracks",
                max_chars=300,
            )
            public = st.checkbox("Make playlist public", value=True)
            st.markdown("")
            if st.button(f"Create playlist with {len(recs)} tracks →", type="primary",
                         use_container_width=True, disabled=not playlist_name.strip()):
                token_info = st.session_state.spotify_token
                user_id    = (st.session_state.spotify_user or {}).get("id","")
                prog       = st.progress(0)
                status_msg = st.empty()
                def _on_progress(i, total):
                    prog.progress(i/total)
                    status_msg.caption(f"Searching Spotify: {i} / {total} tracks…")
                try:
                    from spotify_push import make_client, push_playlist
                    sp = make_client(token_info["access_token"])
                    push_result = push_playlist(
                        sp=sp, user_id=user_id,
                        name=playlist_name.strip(), description=playlist_desc.strip(),
                        recs=recs, public=public, progress_callback=_on_progress,
                    )
                    prog.empty(); status_msg.empty()
                    st.session_state.spotify_push_result = push_result
                    st.rerun()
                except Exception as e:
                    prog.empty(); status_msg.empty()
                    st.error(f"**Spotify error:** {e}")
        else:
            if st.session_state.get("spotify_auth_error"):
                st.error(f"**Connection failed:** {st.session_state.spotify_auth_error}")
                st.session_state.spotify_auth_error = ""
            st.markdown("Connect your Spotify account to create this playlist directly.")
            st.markdown("")
            profile = st.session_state.get("project","user")
            from spotify_push import make_oauth, get_auth_url
            oauth = make_oauth(client_id, client_secret, redirect_uri)
            url   = get_auth_url(oauth, state=profile)
            st.markdown(
                f'<a href="{url}" target="_self" style="display:inline-block;'
                'padding:0.55rem 1.4rem;background:#1DB954;color:#000;font-weight:700;'
                'border-radius:6px;text-decoration:none;font-size:0.95rem;">'
                'Connect to Spotify →</a>',
                unsafe_allow_html=True,
            )
            st.caption("Clicking Connect will briefly redirect you to Spotify's login page. "
                       "Your recommendations are saved and will be ready when you return.")
            with st.expander("🔍 Debug: OAuth URL (temporary)"):
                st.caption(f"**Redirect URI in use:** `{redirect_uri}`")
                st.caption(f"**Full auth URL:** `{url}`")

    with divider_col:
        st.markdown("<div style='border-left:1px solid #333;min-height:640px;'></div>",
                    unsafe_allow_html=True)

    with right:
        # ── Soundiiz logo header ──
        _sz_logo = _logo_img(LOGO_SOUNDIIZ, "150px")
        if _sz_logo:
            st.markdown(_sz_logo, unsafe_allow_html=True)
        else:
            st.markdown("### Download for Soundiiz")

        st.markdown(
            "Not on Spotify? Download the CSV and import to your preferred service "
            "via **[soundiiz.com](https://soundiiz.com)**."
        )

        fname = _default_output()
        st.download_button("⬇  Download Soundiiz CSV",
                           data=_recs_to_soundiiz_csv(recs),
                           file_name=fname, mime="text/csv",
                           use_container_width=True, type="primary")
        st.download_button("⬇  Download full detail CSV",
                           data=_recs_to_detail_csv(recs),
                           file_name=fname.replace(".csv","_detail.csv"),
                           mime="text/csv", use_container_width=True)
        st.markdown("")

        # ── Supported services ──
        st.markdown("#### Imports to:")
        _amz_logo = _logo_img(LOGO_AMAZON, "130px")
        _apl_logo = _logo_img(LOGO_APPLE, "130px")
        if _amz_logo or _apl_logo:
            logo_row = "".join([
                f'<div style="flex:1;text-align:center;">{_amz_logo}</div>' if _amz_logo else "",
                f'<div style="flex:1;text-align:center;">{_apl_logo}</div>' if _apl_logo else "",
            ])
            st.markdown(
                f'<div style="display:flex;gap:1rem;align-items:center;'
                f'margin:0.6rem 0 1rem 0;">{logo_row}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Amazon Music · Apple Music · Tidal · YouTube Music")

        st.markdown("#### How to import")
        st.markdown("""
1. **Download** the Soundiiz CSV above
2. Go to **[soundiiz.com](https://soundiiz.com)** and sign in
3. Click **Transfer → Import from file**, choose **CSV**
4. Select your destination platform and click **Transfer**

*Free tier: up to 200 tracks per playlist.*
""")

    st.markdown("")
    if st.button("← Back to results"):
        _go(8)

# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.markdown("### Session")
        if st.session_state.project:
            st.markdown(f"**Profile:** `{st.session_state.project}`")
        if st.session_state.temperature and st.session_state.temperature != "Any":
            st.markdown(f"**Temperature:** {st.session_state.temperature}")
        if st.session_state.genre and st.session_state.genre != "Any":
            st.markdown(f"**Genre:** {st.session_state.genre}")
        if st.session_state.decade and st.session_state.decade != "Any":
            st.markdown(f"**Decade:** {st.session_state.decade}")
        pool = st.session_state.anchor_pool_tracks
        if pool:
            st.markdown(f"**Anchor pool:** {len(pool)} tracks")
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

_handle_spotify_callback()
_sidebar()

# Product Design is a full-page overlay — skip progress bar when active
if st.session_state.get("show_product_design"):
    step_product_design()
else:
    _progress_bar()
    step = st.session_state.step
    if   step == 1: step_overview()
    elif step == 2: step_history()
    elif step == 3: step_profile()
    elif step == 4: step_parameters()
    elif step == 5: step_blacklist()
    elif step == 6: step_discovery()
    elif step == 7: step_anchor_pool()
    elif step == 8: step_run()
    elif step == 9: step_export()
