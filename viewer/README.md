# Side-by-side Replay Viewer (Task 18.5)

Compares two CityFlow replays in parallel — synchronized timeline, live
metrics (AWT / AQL / Throughput), Chart.js trends, and per-intersection
winner dots.

## Quick start

1. Build the manifest (rerun whenever new replays land):

   ```bash
   python scripts/build_viewer_manifest.py
   ```

   Writes `results/replays/manifest.json` describing every
   `<base>.txt` + `<base>_roadnet.json` pair found and links the matching
   `results/metrics/*.json` for the ground-truth panel.

2. Serve the project root (NOT `viewer/`, since the viewer reads
   `../results/...`):

   ```bash
   python -m http.server 8000
   ```

3. Open `http://localhost:8000/viewer/` in Chrome or Edge.

4. Pick a replay in each panel → click **Load**. Panel A indexes its replay
   first (streaming, no full-file parse), panel B follows. When both are
   ready, press **Play** (or `Space`) to scrub both maps in sync.

## Controls

| Action | UI / Key |
|---|---|
| Play / Pause | `▶ Play` button or `Space` / `P` |
| Step forward / back | `]` / `[` |
| Scrub | timeline slider |
| Speed | `Speed` dropdown |
| Vehicle size | top-bar slider, `1x` – `5x` |
| High-contrast theme | top-bar checkbox |
| Pan | drag the map |
| Zoom | mouse wheel over the map |

## Architecture (brief)

```
viewer/
├── index.html         layout
├── css/style.css      themes
├── js/main.js         orchestrator + sync timeline + UI wiring
├── js/renderer.js     PixiJS map renderer (roads / intersections / signals / vehicles)
├── js/replay-loader.js streaming index + lazy step parser
├── js/metrics.js      live AWT/AQL/Throughput + ground-truth loader
└── js/compare.js      per-intersection winner dots
```

* Replay parsing uses `fetch().body.getReader()` to stream chunks; only one
  pass over the file builds an in-memory **byte-offset index** of every
  step start. Random access is then O(1) `subarray` + `TextDecoder`.
* `MapRenderer` keeps separate Pixi containers for road / intersection /
  signal / vehicle / compare layers so individual layers can be re-painted
  without touching the others.
* Live metrics derive vehicle speed from frame-to-frame position deltas
  (CityFlow's replay format does not store speed directly). Ground-truth
  ATT/AQL/AWT come from `results/metrics/*.json`, which the runner writes
  out automatically.
