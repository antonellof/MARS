# MARS Visual Demo — Predictive Failure Prevention

A 35-second video demonstrating MARS cross-modal memory using **real factory footage**, a **3D KUKA robot arm**, and **real audio**.

## Data sources

| Source | What | How |
|--------|------|-----|
| FANUC factory tour (YouTube) | Real camera frames of robot arms on a production line | Downloaded via yt-dlp, extracted at 10fps |
| PyBullet KUKA iiwa | 3D robot arm model rendered offscreen (digital twin) | Built-in URDF model, TinyRenderer backend |
| Factory audio | Real ambient factory sounds as waveform visualization | Extracted from video via ffmpeg |

## Scenario

1. **Normal ops** (4-13s) — Robot arm working, sensors stream visual/vibration/thermal to MARS
2. **Oil leak** (13-17s) — Camera detects oil near bearing #3 (visual memory stored)
3. **Vibration anomaly** (17-22s) — 340 Hz harmonic appears (ambiguous alone)
4. **Cross-modal retrieval** (22-27s) — MARS BFS traverses vibration→visual bridge, retrieves oil leak
5. **Preventive shutdown** (27-31s) — Correlated evidence → bearing failure predicted → safe stop

**Key insight:** Neither sensor alone triggers shutdown. The NSN cross-modal bridge correlates observations across modalities in < 1ms.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (downloads video, extracts data, generates demo)
python generate_demo.py

# Reuse already-downloaded data
python generate_demo.py --skip-download

# Preview a single frame
python generate_demo.py --skip-download --preview-frame 720
```

Requirements: `ffmpeg` must be installed and on PATH.

## Video layout (1920x1080)

```
┌─────────────────┬──────────────────┬─────────────────┐
│  CAMERA FEED    │  3D DIGITAL TWIN │  MEMORY GRAPH   │
│  (real FANUC)   │  (KUKA PyBullet) │  (NSN nodes)    │
├─────────────────┼──────────────────┼─────────────────┤
│  AUDIO WAVEFORM │  TIMELINE        │  LATENCY METRICS│
│  (real audio)   │  (danger meter)  │  (p50/p99/hist) │
├─────────────────┴──────────────────┴─────────────────┤
│  NARRATIVE + ALERT BAR                               │
├──────────────────────────────────────────────────────│
│  MARS PIPELINE DETAILS + MEMORY STATS                │
└──────────────────────────────────────────────────────┘
```

## On vast.ai

```bash
pip install opencv-python-headless numpy pybullet yt-dlp
apt-get install -y ffmpeg  # if not already available
python generate_demo.py
# Download mars_demo.mp4
```
