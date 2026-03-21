# Music Discovery Agent

A behavior-grounded music discovery engine that analyzes your listening history
and recommends new songs by under-explored artists — driven by song-level taste
adjacency, not "similar artists."

Outputs a Soundiiz-importable playlist CSV scored by the DCSv2 model:
CLS (Co-Listening Strength, 45%) · CMS (Credible Mention Strength, 35%) · MES (Mechanics Evidence Score, 20%)

---

## Requirements

- Python 3.9 or later
- An Anthropic API account with credits → https://console.anthropic.com
- A Last.fm API key (free) → https://www.last.fm/api/account/create
- Your listening history: Last.fm username or a Spotify CSV export

---

## Setup

**1. Install dependencies**

```
pip install -r requirements.txt
```

**2. Create your secrets file**

Create a folder called `.streamlit` inside this `music-agent` folder, then
create a file inside it called `secrets.toml` with the following contents:

```
ANTHROPIC_API_KEY = "sk-ant-api03-..."
LASTFM_API_KEY    = "your-lastfm-api-key"
```

Replace the values with your own keys. Never share this file.

**3. Run the app**

```
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Getting your listening history

**Last.fm**
Enter your Last.fm username directly in the app — no export needed.

**Spotify**
Request your data export at https://spotify.com/us/account/privacy
May take up to 5 days (past year) or 30 days (full history).
Upload the CSV in the app when prompted.

---

## What the wizard does

| Step | What happens |
|---|---|
| 1 · History | Connect Last.fm username or upload Spotify CSV |
| 2 · Profile | Name your listener profile (stores blacklist + session history) |
| 3 · Parameters | Set purge thresholds and batch size |
| 4 · Blacklist | Permanently exclude artists from recommendations |
| 5 · Lane | Choose a musical direction for this session |
| 6 · Anchor Pool | Review the taste signals that will drive recommendations |
| 7 · Run | Generate and download your playlist |

---

## Output files

| File | Use |
|---|---|
| Soundiiz CSV | Import directly at soundiiz.com to sync to Spotify, Apple Music, etc. |
| Full detail CSV | Includes DCS, CLS, CMS, MES scores and rationale per track |

---

## Notes

- Each run is remembered — previously recommended tracks are never repeated
- The blacklist persists across sessions per profile
- State files are stored locally at `~/.music-agent/{profile}.json`
