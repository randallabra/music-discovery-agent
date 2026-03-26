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
        "genre":             [],
        "decade":            [],
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
        "demo_mode":           False,
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
    genres: list[str],
    decades: list[str],
    blacklist: set,
    min_plays: int,
) -> list[dict]:
    """Return tracks matching all active filters, sorted by play count desc."""
    pocket_filter = TEMPERATURE_POCKET_MAP.get(temperature)
    genre_lower_set = {g.lower() for g in genres} if genres else None
    decade_set = set(decades) if decades else None

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
        if genre_lower_set and t.get("genre","").lower() not in genre_lower_set:
            continue
        if decade_set and t.get("decade") not in decade_set:
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

    # Return full filtered list sorted by plays — caller applies any artist cap
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
    error   = st.query_params.get("error", "")

    # Spotify returned an explicit error (e.g. user denied access)
    if error and not code:
        st.session_state.spotify_auth_error = f"Spotify returned: {error}"
        st.query_params.clear()
        st.session_state.step = 9
        st.rerun()

    if not code:
        return
    if st.session_state.get("spotify_token"):
        st.query_params.clear()
        st.rerun()
    client_id, client_secret, redirect_uri = _spotify_secrets()
    if not client_id or not client_secret:
        st.session_state.spotify_auth_error = "Spotify credentials missing from app secrets."
        st.query_params.clear()
        st.session_state.step = 9
        st.rerun()
    try:
        from spotify_push import make_oauth, exchange_code as _exc, make_client, get_current_user
        oauth      = make_oauth(client_id, client_secret, redirect_uri)
        token_info = _exc(oauth, code)
        if not token_info or "access_token" not in token_info:
            raise ValueError(f"Token exchange returned no access_token. Response: {token_info}")
        sp         = make_client(token_info["access_token"])
        user       = get_current_user(sp)
        st.session_state.spotify_token = token_info
        st.session_state.spotify_user  = user
    except Exception as e:
        st.session_state.spotify_auth_error = str(e)
        st.query_params.clear()
        st.session_state.step = 9   # land on Export so the error is visible
        st.rerun()
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
    st.rerun()   # clean rerun — lands on Export with token already in session state

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
    _g  = st.session_state.get("genre", [])
    _g_str = "_".join(_g) if isinstance(_g, list) and _g else (_g if isinstance(_g, str) else "")
    g   = _g_str.replace(" ","_").lower() or "any"
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

    st.markdown("---")

    # ── Demo mode ────────────────────────────────────────────────
    data_mode = st.radio(
        "Data source",
        options=[
            "I'll upload my own listening history",
            "I don't have my data yet — let's work off the existing track_universe.db",
        ],
        index=1 if st.session_state.demo_mode else 0,
        label_visibility="collapsed",
        key="data_mode_radio",
    )
    demo_chosen = data_mode.startswith("I don't have")

    if demo_chosen:
        st.info(
            "**Demo mode** — you'll be exploring the app using **marlonrando**'s "
            "pre-indexed listening history (12,899 tracks). "
            "All filters, Temperature, Anchor Pool, and recommendation engine work exactly "
            "as they would with your own data."
        )

    st.markdown("---")
    st.markdown("#### Or fetch directly from Last.fm")
    st.caption(
        "**Direct Last.fm API import coming soon** — no CSV download needed.\n\n"
        "In the meantime, export your full history instantly at "
        "[lastfm.ghan.nl/export](https://lastfm.ghan.nl/export/)."
    )

    uploaded = None; source = None
    if not demo_chosen:
        if lastfm_file and spotify_file:
            st.warning("Please upload from one service at a time.")
        elif lastfm_file:
            uploaded, source = lastfm_file, "lastfm"
        elif spotify_file:
            uploaded, source = spotify_file, "spotify"

    next_enabled = demo_chosen or (uploaded is not None)

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back"):
            _go(1)
    with col_next:
        if st.button("Next →", type="primary", disabled=not next_enabled):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            if demo_chosen:
                st.session_state.demo_mode    = True
                st.session_state.tmp_csv_path = None
                st.session_state.csv_filename = "(demo — marlonrando)"
                st.session_state.source       = "lastfm"
                st.session_state.project      = "marlonrando"
            else:
                st.session_state.demo_mode    = False
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp.write(uploaded.read()); tmp.flush()
                st.session_state.tmp_csv_path = tmp.name
                st.session_state.csv_filename = uploaded.name
                st.session_state.source       = source
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
            idx = _load_index(project)
            sf = _state_file(project)
            if idx:
                if sf.exists():
                    s = load_state(sf)

                    # Count unique tracks from uploaded CSV
                    csv_path = st.session_state.get("tmp_csv_path")
                    unique_track_count = None
                    if csv_path:
                        try:
                            import csv as _csv
                            tracks_seen = set()
                            with open(csv_path, encoding="utf-8", errors="replace") as f:
                                reader = _csv.DictReader(f)
                                for row in reader:
                                    artist = (row.get("artist") or row.get("Artist") or "").strip().lower()
                                    track = (row.get("track") or row.get("Track") or row.get("track_name") or "").strip().lower()
                                    if artist and track:
                                        tracks_seen.add((artist, track))
                            unique_track_count = len(tracks_seen)
                        except Exception:
                            pass

                    lines = [f"**{project}** — indexed"]
                    if unique_track_count is not None:
                        lines.append(f"- Total unique tracks in your upload: **{unique_track_count:,}**")
                    lines.append(f"- Tracks in index (taste signals): **{len(idx):,}**")
                    lines.append(f"- Previous runs: **{s.run_count}**")
                    lines.append(f"- Tracks withheld from anchor pool (prior runs): **{len(s.collision_memory)}**")
                    lines.append(f"- Blacklisted artists: **{len(s.blacklist)}**")
                    st.info("\n\n".join(lines))

                    if s.run_count > 0:
                        st.markdown("")
                        collision_choice = st.radio(
                            "Prior run history",
                            options=[
                                "Withhold previously recommended tracks from future anchor pools and recommendations",
                                "Do not withhold previously selected tracks from anchor pools and recommendations moving forwards",
                            ],
                            index=0 if not s.ignore_collision_memory else 1,
                            key="collision_memory_choice",
                            label_visibility="collapsed",
                        )
                        if collision_choice.startswith("Do not withhold"):
                            s.ignore_collision_memory = True
                            st.session_state.state_obj = s
                        else:
                            s.ignore_collision_memory = False
                            st.session_state.state_obj = s
                else:
                    st.info(f"**{project}** — {len(idx):,} indexed tracks. No run history yet.")
            else:
                st.warning(
                    f"No index found for **{project}**. "
                    "The index file `{project}_track_index.json` must be present in the "
                    "app directory or `~/.music-agent/`."
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
        st.markdown("**Anchor pool:** *the seed data from your listening history which will inform song recommendations*")
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
        "Artists listed here will be excluded from your anchor pool and will never appear in recommendations. "
        "Include either artists you know inside out, or artists that fit your tastes but you know you don't like. "
        "Enter one artist per line — or comma-separated.\n\n"
        "This blacklist will live locally on your hard drive to inform all future sessions and playlist creations. "
        "You can expand it over time."
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
        "These filters will shape your anchor pool — the taste signals the engine builds from, "
        "and through that, the songs that are recommended back to you."
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
        st.caption("Top 15 genres by catalog size. Check one or more — or leave all unchecked for any.")
        _saved_genres = st.session_state.genre if isinstance(st.session_state.genre, list) else []
        _genre_opts = GENRE_OPTIONS[1:]   # exclude "Any"
        # Lay out checkboxes in 3 columns
        _gcols = st.columns(3)
        genre_selection = [
            g for i, g in enumerate(_genre_opts)
            if _gcols[i % 3].checkbox(g, value=(g in _saved_genres), key=f"genre_cb_{g}")
        ]

        st.markdown("")
        st.markdown("#### Decade  *(~90% of catalog has year data)*")
        st.caption("Check one or more — or leave all unchecked for any decade.")
        _saved_decades = st.session_state.decade if isinstance(st.session_state.decade, list) else []
        _decade_opts = DECADE_OPTIONS[1:]  # exclude "Any"
        _dcols = st.columns(4)
        decade_selection = [
            d for i, d in enumerate(_decade_opts)
            if _dcols[i % 4].checkbox(d, value=(d in _saved_decades), key=f"decade_cb_{d}")
        ]

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
                index, temperature, genre_selection, decade_selection,
                blacklist=set(state.blacklist),
                min_plays=p["min_track_plays"],
            )
            st.markdown("")
            total_in_index = len(index)
            if len(preview) >= p["anchor_pool_size"]:
                st.success(f"**{len(preview):,}** of {total_in_index:,} tracks match — {p['anchor_pool_size']} will be sampled for the anchor pool.")
            elif len(preview) > 0:
                st.warning(f"**{len(preview)}** of {total_in_index:,} tracks match — smaller than your anchor pool size of {p['anchor_pool_size']}. All will be used.")
            else:
                st.error(f"No tracks match this combination (out of {total_in_index:,}). Try broadening the filters.")

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back"):
            _go(5)
    with col_next:
        filters_changed = (
            temperature != st.session_state.temperature or
            genre_selection != st.session_state.genre or
            decade_selection != st.session_state.decade
        )
        if st.button("Next →  (build anchor pool)", type="primary"):
            st.session_state.temperature = temperature
            st.session_state.genre       = genre_selection
            st.session_state.decade      = decade_selection
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

    genre_display = ", ".join(genre) if isinstance(genre, list) and genre else "Any"
    decade_display = ", ".join(decade) if isinstance(decade, list) and decade else "Any"
    filter_parts = [f"Temperature: **{temperature}**", f"Genre: **{genre_display}**"]
    if (isinstance(decade, list) and decade) or (isinstance(decade, str) and decade != "Any"):
        filter_parts.append(f"Decade: **{decade_display}**")
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
        # Per-artist cap of 2 (highest-played first), then slice to anchor_pool_size
        _ac: dict[str, int] = {}
        _capped = []
        for _t in filtered:
            _a = _t["artist"]
            if _ac.get(_a, 0) < 2:
                _capped.append(_t)
                _ac[_a] = _ac.get(_a, 0) + 1
        raw = _capped[:p["anchor_pool_size"]]
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

    _genre_val = st.session_state.get("genre", [])
    _genre_display = ", ".join(_genre_val) if isinstance(_genre_val, list) and _genre_val else (_genre_val if isinstance(_genre_val, str) and _genre_val != "Any" else "Any")
    _decade_val = st.session_state.get("decade", [])
    _decade_display = ", ".join(_decade_val) if isinstance(_decade_val, list) and _decade_val else (_decade_val if isinstance(_decade_val, str) and _decade_val != "Any" else "Any")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Run configuration**")
        st.markdown(f"""
| | |
|---|---|
| Profile | `{st.session_state.project}` |
| Temperature | {st.session_state.temperature} |
| Genre | {_genre_display} |
| Decade | {_decade_display} |
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
    temperature = st.session_state.get("temperature", "Any")
    genre_val = st.session_state.get("genre", [])
    genre_str = ", ".join(genre_val) if isinstance(genre_val, list) and genre_val else (genre_val if isinstance(genre_val, str) and genre_val != "Any" else "Any")
    decade_val = st.session_state.get("decade", [])
    decade_str = ", ".join(decade_val) if isinstance(decade_val, list) and decade_val else (decade_val if isinstance(decade_val, str) and decade_val != "Any" else "Any")

    # Build a lane string for the existing recommender interface
    lane_str = f"{temperature} / {genre_str} / {decade_str}"

    from history import AnchorPool as _AP, parse_history_csv

    csv_path_str = st.session_state.get("tmp_csv_path")
    demo_mode    = st.session_state.get("demo_mode", False)

    if csv_path_str:
        # ── Normal path: parse uploaded CSV ──────────────────────
        csv_path = Path(csv_path_str)
        source   = st.session_state.get("source") or "lastfm"
        try:
            stats, total_scrobbles, _fmt = parse_history_csv(csv_path, source)
        except Exception as e:
            st.error(f"**Could not parse listening history CSV:** {e}")
            return

        known_tracks: frozenset = frozenset(
            (artist.strip().lower(), track.strip().lower())
            for artist, s in stats.items()
            for track, _ in s.top_tracks
        )
        _all_ktp = sorted(
            ((artist, track, plays)
             for artist, s in stats.items()
             for track, plays in s.top_tracks),
            key=lambda x: -x[2],
        )
        known_tracks_by_plays = _all_ktp[:300]
        _title_plays: dict[str, int] = {}
        for artist, s in stats.items():
            for track, plays in s.top_tracks:
                _k = track.strip().lower()
                _title_plays[_k] = _title_plays.get(_k, 0) + plays
        known_titles: frozenset = frozenset(_title_plays.keys())
        total_artists = len(stats)

    elif demo_mode:
        # ── Demo path: build known_* directly from the pre-indexed JSON ──
        _idx = _load_index(st.session_state.project)
        known_tracks = frozenset(
            (t["artist"].strip().lower(), t["track"].strip().lower())
            for t in _idx.values()
        )
        _sorted_idx = sorted(_idx.values(), key=lambda x: -x["plays"])
        known_tracks_by_plays = [
            (t["artist"], t["track"], t["plays"]) for t in _sorted_idx[:300]
        ]
        known_titles = frozenset(t["track"].strip().lower() for t in _idx.values())
        total_scrobbles = sum(t["plays"] for t in _idx.values())
        total_artists   = len({t["artist"] for t in _idx.values()})

    else:
        st.error(
            "**No listening history found.** "
            "Please go back to Step 2 and upload your Last.fm or Spotify CSV, "
            "or choose the demo mode option."
        )
        if st.button("← Back to Step 2"):
            _go(2)
        return

    # Build AnchorPool — tracks come from the pre-indexed JSON filter,
    # but known_* fields come from the full CSV history.
    ap = _AP.__new__(_AP)
    ap.tracks               = [{"artist": t["artist"], "track": t["track"], "plays": t["plays"]}
                                for t in pool]
    ap.purged_artists       = []
    ap.total_scrobbles      = total_scrobbles
    ap.total_artists        = total_artists
    ap.eligible_count       = len({t["artist"] for t in pool})
    ap.known_tracks         = known_tracks
    ap.known_tracks_by_plays = known_tracks_by_plays
    ap.known_titles         = known_titles

    _csv_path_for_config = Path(csv_path_str) if csv_path_str else Path("/dev/null")
    _source_for_config   = st.session_state.get("source") or "lastfm"
    config = RunConfig(
        history_csv=_csv_path_for_config,
        source=_source_for_config,
        lane=lane_str,
        project=st.session_state.project,
        vibe_focus="",
        decade=decade_str,
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
    # Clear any leftover checkbox state so every new result starts fully checked
    for _k in list(st.session_state.keys()):
        if _k.startswith("track_sel_"):
            del st.session_state[_k]
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

    # Ensure every track starts checked — value=True is the fallback when a key
    # is absent; _execute_run clears these keys before each new run so returning
    # users always see a fresh, fully-checked list.
    st.markdown("**Select tracks to include in your playlist** — uncheck any you'd like to remove:")
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
            st.checkbox("", value=True, key=f"track_sel_{i}", label_visibility="collapsed")
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

    export_left, export_right = st.columns([1, 1], gap="large")

    with export_left:
        _sp_logo = _logo_img(LOGO_SPOTIFY, "120px")
        if _sp_logo:
            st.markdown(_sp_logo, unsafe_allow_html=True)
        if st.button("Send to Spotify →", type="primary", use_container_width=True):
            st.session_state["_export_recs_override"] = selected_recs
            _go(9)
        st.markdown("")
        detail_csv = _recs_to_detail_csv(selected_recs)
        st.download_button("⬇  Full Detail CSV", data=detail_csv,
                           file_name=_default_output().replace(".csv","_detail.csv"),
                           mime="text/csv", use_container_width=True)
        st.markdown("")
        if st.button("Run again →", use_container_width=True):
            st.session_state.result             = None
            st.session_state.anchor_pool_raw    = None
            st.session_state.anchor_pool_tracks = None
            st.session_state.pop("_export_recs_override", None)
            _go(6)

    with export_right:
        _sz_logo = _logo_img(LOGO_SOUNDIIZ, "120px")
        if _sz_logo:
            st.markdown(_sz_logo, unsafe_allow_html=True)
        st.markdown("Not on Spotify? Download below and import via **[soundiiz.com](https://soundiiz.com)**.")
        soundiiz_csv = _recs_to_soundiiz_csv(selected_recs)
        st.download_button("⬇  Download Soundiiz CSV", data=soundiiz_csv,
                           file_name=_default_output(), mime="text/csv",
                           use_container_width=True, type="primary")
        st.markdown("")
        # Platform logos
        _amz_logo = _logo_img(LOGO_AMAZON, "110px")
        _apl_logo = _logo_img(LOGO_APPLE,  "110px")
        if _amz_logo or _apl_logo:
            logo_row = "".join([
                f'<div style="flex:1;text-align:center;">{_amz_logo}</div>' if _amz_logo else "",
                f'<div style="flex:1;text-align:center;">{_apl_logo}</div>' if _apl_logo else "",
            ])
            st.markdown(
                f'<div style="display:flex;gap:1rem;align-items:center;margin:0.4rem 0;">'
                f'{logo_row}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Amazon Music · Apple Music · Tidal · YouTube Music")
        st.markdown("**How to import:** Download CSV → [soundiiz.com](https://soundiiz.com) "
                    "→ Transfer → Import from file → CSV → pick your platform.")

# ─────────────────────────────────────────────────────────────
#  Step 9 — Export
# ─────────────────────────────────────────────────────────────

def step_export():
    from datetime import date

    st.subheader("Step 9 — Export Your Playlist")

    # Surface any Spotify auth error immediately — shown before any other content
    # so it's visible even if result is None after the OAuth redirect.
    _auth_err = st.session_state.get("spotify_auth_error", "")
    if _auth_err:
        st.error(
            f"**Spotify connection failed.** Here's the exact error so we can diagnose it:\n\n"
            f"```\n{_auth_err}\n```\n\n"
            "Common causes: Spotify rejected the authorization code (try connecting again), "
            "or your app may require updated permissions per Spotify's 2026 platform changes. "
            "Copy the error above and share it."
        )
        st.session_state.spotify_auth_error = ""   # clear after display

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

    _sp_logo = _logo_img(LOGO_SPOTIFY, "150px")
    if _sp_logo:
        st.markdown(_sp_logo, unsafe_allow_html=True)

    if not spotify_configured:
        st.info(
            "**Spotify not configured.**\n\n"
            "Add to `.streamlit/secrets.toml`:\n\n"
            "```toml\n"
            "SPOTIFY_CLIENT_ID     = \"your_client_id\"\n"
            "SPOTIFY_CLIENT_SECRET = \"your_client_secret\"\n"
            "SPOTIFY_REDIRECT_URI  = \"https://music-discovery-agent.streamlit.app/\"\n"
            "```\n\n"
            "Register both redirect URIs at "
            "[developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)."
        )
    elif st.session_state.get("spotify_push_result"):
        pr = st.session_state.spotify_push_result
        st.success(f"**Playlist created** — {len(pr['found'])} of {len(recs)} tracks added")
        st.markdown(f"### [Open \"{pr['playlist_name']}\" in Spotify]({pr['playlist_url']})")
        if pr["not_found"]:
            st.markdown(f"**{len(pr['not_found'])} tracks not found on Spotify:**")
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
        # (auth errors are surfaced at the top of this function, not here)
        st.markdown("Connect your Spotify account to create this playlist directly.")
        st.markdown("")
        profile = st.session_state.get("project","user")
        from spotify_push import make_oauth, get_auth_url
        import streamlit.components.v1 as _components
        oauth = make_oauth(client_id, client_secret, redirect_uri)
        url   = get_auth_url(oauth, state=profile)

        # Use a JS-driven button so the click navigates window.top directly —
        # st.markdown anchor tags with target="_top" can be blocked by Streamlit
        # Cloud's iframe sandbox policy.
        _sp_logo = _logo_img(ASSETS_DIR / "logo_spotify.png", max_width="110px")
        _components.html(f"""
<style>
  body {{ margin:0; padding:0; background:transparent; }}
  .sp-wrap {{ display:flex; align-items:center; gap:18px; }}
  .sp-btn {{
    display:inline-flex; align-items:center; gap:10px;
    background:#1DB954; color:#000; border:none;
    padding:11px 28px; border-radius:50px;
    font-size:15px; font-weight:700; cursor:pointer;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    transition:background 0.15s;
    white-space:nowrap;
  }}
  .sp-btn:hover {{ background:#1ed760; }}
</style>
<div class="sp-wrap">
  {_sp_logo}
  <button class="sp-btn" onclick="window.top.location.href='{url}'">
    Connect to Spotify →
  </button>
</div>
""", height=70)

        st.caption("Clicking Connect will briefly redirect you to Spotify's login page. "
                   "Your recommendations are saved and will be ready when you return.")
        with st.expander("🔍 Debug: OAuth details"):
            st.caption(f"**Redirect URI in use:** `{redirect_uri}`")
            st.caption(f"**Full auth URL:**")
            st.code(url, language=None)

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
        _sb_genre = st.session_state.genre
        _sb_genre_str = ", ".join(_sb_genre) if isinstance(_sb_genre, list) else _sb_genre
        if _sb_genre_str and _sb_genre_str != "Any":
            st.markdown(f"**Genre:** {_sb_genre_str}")
        _sb_decade = st.session_state.decade
        _sb_decade_str = ", ".join(_sb_decade) if isinstance(_sb_decade, list) else _sb_decade
        if _sb_decade_str and _sb_decade_str != "Any":
            st.markdown(f"**Decade:** {_sb_decade_str}")
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
