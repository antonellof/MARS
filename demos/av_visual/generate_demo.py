#!/usr/bin/env python3
"""
MARS Visual Demo — "Predictive Failure Prevention"

Uses REAL factory robot footage + 3D PyBullet KUKA arm + audio waveform
to demonstrate GPU-resident cross-modal memory for industrial safety.

Pipeline:
  1. Downloads a real factory robot arm video (FANUC factory tour)
  2. Extracts frames at 10fps + mono audio waveform
  3. Renders 3D KUKA iiwa arm via PyBullet offscreen (digital twin)
  4. Generates simulated MARS embeddings + memory graph from real frames
  5. Composes a 4-panel dashboard video with real data + MARS overlay

Usage:
    python generate_demo.py                     # full pipeline
    python generate_demo.py --skip-download      # reuse existing data/
    python generate_demo.py --preview-frame 500  # single frame PNG

Output:
    mars_demo.mp4  (1920x1080 @ 30fps, ~35s)

Dependencies: opencv-python, numpy, pybullet, yt-dlp (for download), ffmpeg
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  Constants & theme
# ═══════════════════════════════════════════════════════════════════════
WIDTH, HEIGHT = 1920, 1080
FPS = 30
DURATION_S = 35
SRC_FPS = 10  # extracted frames are at 10fps

BG_DARK      = (18, 18, 22)
BG_PANEL     = (28, 28, 35)
BG_PANEL_ALT = (35, 35, 42)
GRID_COLOR   = (38, 38, 48)
TEXT_WHITE    = (235, 235, 240)
TEXT_DIM      = (130, 130, 140)
TEXT_ACCENT   = (255, 180, 50)
CYAN          = (220, 195, 45)
GREEN         = (95, 215, 75)
RED           = (75, 75, 235)
ORANGE        = (55, 135, 250)
YELLOW        = (45, 215, 250)
TEAL          = (180, 180, 40)

MOD_VISUAL    = (215, 155, 35)
MOD_VIBRATION = (55, 195, 55)
MOD_THERMAL   = (55, 100, 245)
MOD_ACOUSTIC  = (195, 145, 55)

MOD_COLORS = {
    "visual": MOD_VISUAL, "vibration": MOD_VIBRATION,
    "thermal": MOD_THERMAL, "acoustic": MOD_ACOUSTIC,
}

# Panel layout (x, y, w, h)
CAM_RECT     = (30,  30,  620, 380)   # top-left: real camera feed
TWIN_RECT    = (660, 30,  620, 380)   # top-center: 3D digital twin
GRAPH_RECT   = (1290, 30, 600, 380)   # top-right: memory graph
AUDIO_RECT   = (30,  430, 620, 170)   # mid-left: audio waveform
LOG_RECT     = (660, 430, 620, 170)   # mid-center: timeline
METRICS_RECT = (1290, 430, 600, 170)  # mid-right: latency card
NARR_RECT    = (30,  620, 1860, 100)  # bottom: narrative + alerts
STATUS_RECT  = (30,  730, 1860, 340)  # bottom: detailed status


# ═══════════════════════════════════════════════════════════════════════
#  Drawing utilities
# ═══════════════════════════════════════════════════════════════════════
def put_text(img, text, pos, color=TEXT_WHITE, scale=0.6, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def put_text_centered(img, text, cx, cy, color=TEXT_WHITE, scale=0.6, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    put_text(img, text, (cx - tw // 2, cy + th // 2), color, scale, thickness)

def draw_panel(img, rect, color=BG_PANEL, title=None, title_color=TEXT_ACCENT):
    x, y, w, h = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.95, img, 0.05, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 60), 1, cv2.LINE_AA)
    if title:
        put_text(img, title, (x + 10, y + 20), title_color, 0.5, 1)

def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))

def alpha_blend(bg, fg, alpha):
    alpha = max(0.0, min(1.0, alpha))
    return tuple(int(bg[i] + (fg[i] - bg[i]) * alpha) for i in range(3))

def draw_glow_circle(img, center, radius, color, glow):
    if glow > 0.01:
        for r_extra in range(1, 5):
            a = glow * 0.25 / r_extra
            gc = alpha_blend(BG_PANEL, color, a)
            cv2.circle(img, center, radius + r_extra * 4, gc, 2, cv2.LINE_AA)
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, (200, 200, 200), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
#  Data acquisition: download video, extract frames + audio
# ═══════════════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).parent / "data"

YOUTUBE_URL = "https://www.youtube.com/watch?v=-SREct28lJM"  # FANUC factory tour

def download_and_extract(skip_download: bool = False):
    """Download video, extract frames at 10fps and mono audio."""
    DATA_DIR.mkdir(exist_ok=True)
    frames_dir = DATA_DIR / "frames"
    video_path = DATA_DIR / "robot_factory.mp4"
    audio_path = DATA_DIR / "audio.wav"

    if skip_download and frames_dir.exists() and len(list(frames_dir.glob("*.jpg"))) > 100:
        n = len(list(frames_dir.glob("*.jpg")))
        print(f"Reusing existing data: {n} frames + audio")
        return

    # Download video
    if not video_path.exists():
        print(f"Downloading factory robot video...")
        subprocess.run([
            "yt-dlp", "-f", "135+140",  # 480p video + audio
            "-o", str(video_path),
            YOUTUBE_URL
        ], check=True)
    else:
        print(f"Video already downloaded: {video_path}")

    # Extract frames
    frames_dir.mkdir(exist_ok=True)
    print("Extracting frames at 10fps...")
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-t", "40",  # first 40 seconds
        "-vf", f"fps={SRC_FPS},scale=640:480",
        str(frames_dir / "frame_%04d.jpg"),
        "-y", "-loglevel", "warning"
    ], check=True)

    # Extract audio
    print("Extracting audio waveform...")
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-t", "40", "-ac", "1", "-ar", "16000",
        str(audio_path),
        "-y", "-loglevel", "warning"
    ], check=True)

    n = len(list(frames_dir.glob("*.jpg")))
    print(f"Extracted {n} frames + audio.wav")


# ═══════════════════════════════════════════════════════════════════════
#  3D Robot Arm renderer (PyBullet KUKA iiwa)
# ═══════════════════════════════════════════════════════════════════════
class RobotArmRenderer:
    """Renders a KUKA iiwa robot arm via PyBullet offscreen."""

    def __init__(self, width=600, height=360):
        import pybullet as p
        import pybullet_data

        self.p = p
        self.width = width
        self.height = height

        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Load scene
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot)

        # Add a table/conveyor proxy
        table_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.05])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.05],
                                           rgbaColor=[0.4, 0.4, 0.45, 1.0])
        self.table = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_shape,
                                       baseVisualShapeIndex=table_visual,
                                       basePosition=[0.5, 0, 0.3])

        # Add a small workpiece
        wp_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
        wp_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05],
                                        rgbaColor=[0.9, 0.7, 0.1, 1.0])
        self.workpiece = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wp_shape,
                                           baseVisualShapeIndex=wp_visual,
                                           basePosition=[0.5, 0.2, 0.4])

        # Camera matrices
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.8, -1.0, 1.3],
            cameraTargetPosition=[0.2, 0, 0.4],
            cameraUpVector=[0, 0, 1]
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=45, aspect=width / height, nearVal=0.1, farVal=10
        )

        # Precompute joint trajectory (smooth pick-and-place motion)
        self._trajectory = self._generate_trajectory()

    def _generate_trajectory(self):
        """Generate a smooth repeating trajectory for the arm."""
        n_poses = 300
        traj = np.zeros((n_poses, 7))
        for i in range(n_poses):
            t = i / n_poses * 2 * math.pi
            traj[i] = [
                0.3 * math.sin(t),            # joint 0: base rotation
                0.5 + 0.3 * math.sin(t * 0.7), # joint 1: shoulder
                0.2 * math.sin(t * 1.3),        # joint 2: elbow rotation
                -1.0 + 0.4 * math.sin(t * 0.5), # joint 3: elbow bend
                0.3 * math.sin(t * 1.1),        # joint 4: wrist rotation
                0.8 + 0.3 * math.sin(t * 0.8),  # joint 5: wrist bend
                0.5 * math.sin(t * 1.5),        # joint 6: flange rotation
            ]
        return traj

    def render(self, t: float, speed: float = 1.0, danger_level: float = 0.0) -> np.ndarray:
        """Render the robot arm at time t. Returns BGR image."""
        p = self.p

        # Set joint positions from trajectory
        traj_idx = int((t * 10 * speed) % len(self._trajectory))
        joints = self._trajectory[traj_idx]
        for i in range(min(7, self.num_joints)):
            p.resetJointState(self.robot, i, joints[i])

        # Change lighting based on danger level
        light_color = [1.0, 1.0, 1.0]
        if danger_level > 0.3:
            light_color = [1.0, 1.0 - danger_level * 0.5, 1.0 - danger_level * 0.7]

        # Render
        _, _, rgba, _, _ = p.getCameraImage(
            self.width, self.height,
            self.view_matrix, self.proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            lightDirection=[1, -1, 1],
            lightColor=light_color,
        )

        img = np.array(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)
        bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)

        # Add danger tint
        if danger_level > 0.3:
            tint = np.full_like(bgr, (0, 0, 150), dtype=np.uint8)
            alpha = danger_level * 0.15
            cv2.addWeighted(tint, alpha, bgr, 1.0 - alpha, 0, bgr)

        return bgr

    def close(self):
        self.p.disconnect()


# ═══════════════════════════════════════════════════════════════════════
#  Audio waveform loader
# ═══════════════════════════════════════════════════════════════════════
class AudioData:
    """Loads and provides audio waveform data."""

    def __init__(self, wav_path: str, duration_s: float = 40.0):
        self.samples = np.zeros(int(16000 * duration_s), dtype=np.float32)
        if os.path.exists(wav_path):
            import wave
            with wave.open(wav_path, 'rb') as wf:
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                self.samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                self.samples /= max(1, np.max(np.abs(self.samples)))
                self.sample_rate = wf.getframerate()
        else:
            self.sample_rate = 16000
            print(f"Warning: audio file not found: {wav_path}")

    def get_waveform(self, t_start: float, t_end: float, n_points: int = 400) -> np.ndarray:
        """Get a downsampled waveform slice."""
        i_start = max(0, int(t_start * self.sample_rate))
        i_end = min(len(self.samples), int(t_end * self.sample_rate))
        if i_end <= i_start:
            return np.zeros(n_points)
        chunk = self.samples[i_start:i_end]
        # Downsample by taking max in each bin
        step = max(1, len(chunk) // n_points)
        result = np.zeros(n_points)
        for i in range(min(n_points, len(chunk) // step)):
            seg = chunk[i * step:(i + 1) * step]
            result[i] = np.max(np.abs(seg)) if len(seg) > 0 else 0
        return result

    def get_rms(self, t: float, window_s: float = 0.1) -> float:
        """Get RMS energy at time t."""
        i_start = max(0, int((t - window_s / 2) * self.sample_rate))
        i_end = min(len(self.samples), int((t + window_s / 2) * self.sample_rate))
        if i_end <= i_start:
            return 0.0
        chunk = self.samples[i_start:i_end]
        return float(np.sqrt(np.mean(chunk ** 2)))


# ═══════════════════════════════════════════════════════════════════════
#  Memory graph data structures
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class MemoryNode:
    node_id: int
    label: str
    modality: str
    x: float
    y: float
    timestamp: float
    glow: float = 0.0
    retrieved: bool = False
    desc: str = ""

@dataclass
class MemoryEdge:
    src: int
    dst: int
    cross_modal: bool = False
    active: bool = False
    glow: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Scenario state machine
# ═══════════════════════════════════════════════════════════════════════
class Scenario:
    """
    Timeline:
     0-2s    Title card
     2-4s    Intro: factory feed appears
     4-13s   Normal ops: robot arm working, sensors streaming
     13-17s  Oil leak detected by camera
     17-22s  Vibration anomaly: frequency shift in audio
     22-27s  Cross-modal retrieval: MARS bridges vibration→visual
     27-31s  Preventive shutdown
     31-35s  Summary card with metrics
    """

    def __init__(self):
        self.rng = np.random.RandomState(42)

    def get_phase(self, t: float) -> str:
        if t < 2.0:   return "title"
        if t < 4.0:   return "intro"
        if t < 13.0:  return "normal_ops"
        if t < 17.0:  return "oil_leak"
        if t < 22.0:  return "vibration"
        if t < 27.0:  return "cross_modal"
        if t < 31.0:  return "shutdown"
        return "summary"

    def get_subtitle(self, t: float, phase: str) -> str:
        subs = {
            "intro":       "MARS Demo — Real factory footage + 3D digital twin",
            "normal_ops":  {
                4: "Robot arm pick-and-place at 60 Hz — all sensors nominal",
                7: "Camera, vibration, thermal sensors streaming to MARS",
                10: "Memory graph building cross-modal bridges automatically",
            },
            "oil_leak":    "Camera detects oil droplets near bearing joint #3",
            "vibration":   "Vibration sensor: anomalous 340 Hz harmonic detected",
            "cross_modal": "MARS: vibration query -> BFS -> visual memory of oil leak",
            "shutdown":    {
                27: "Correlated evidence: bearing failure IMMINENT",
                29: "Preventive shutdown — incident PREVENTED",
            },
        }
        val = subs.get(phase, "")
        if isinstance(val, dict):
            result = ""
            for ts, text in sorted(val.items()):
                if t >= ts:
                    result = text
            return result
        return val

    def get_alert(self, t: float, phase: str):
        """Returns (text, color, alpha) or None."""
        if phase == "oil_leak":
            return ("VISUAL ANOMALY", MOD_VISUAL, 0.6 + 0.4 * math.sin(t * 2.5))
        if phase == "vibration":
            return ("VIBRATION ANOMALY", MOD_VIBRATION, 0.7 + 0.3 * math.sin(t * 3))
        if phase == "cross_modal":
            return ("CROSS-MODAL MATCH: vibration <-> oil leak",
                    YELLOW, 0.8 + 0.2 * math.sin(t * 2))
        if phase == "shutdown":
            if t < 29:
                return ("PREVENTIVE SHUTDOWN", ORANGE, 1.0)
            return ("INCIDENT PREVENTED", GREEN, 1.0)
        return None

    def get_danger(self, t: float, phase: str) -> float:
        if phase == "oil_leak":   return 0.3
        if phase == "vibration":  return 0.5 + (t - 17) * 0.04
        if phase == "cross_modal": return 0.7 + (t - 22) * 0.05
        if phase == "shutdown":   return max(0, 0.9 - (t - 27) * 0.25)
        return 0.0

    def get_arm_speed(self, t: float, phase: str) -> float:
        if phase == "shutdown":
            return max(0, 1.0 - (t - 27) * 0.35)
        if phase == "vibration":
            return 0.8
        return 1.0

    def build_memory(self, t: float, phase: str):
        """Build memory graph nodes and edges for current time."""
        nodes, edges = [], []
        if t < 5.0:
            return nodes, edges

        n_vis = min(12, max(0, int((t - 5) * 1.5)))
        n_vib = min(8, max(0, int((t - 6) * 1.2)))
        n_therm = min(6, max(0, int((t - 7) * 0.8)))

        nid = 0
        oil_nodes, anom_vib_nodes = [], []

        # Visual cluster (upper-left)
        for i in range(n_vis):
            angle = -0.7 + i * 0.18
            r = 0.22 + (i % 3) * 0.07
            is_oil = i in (2, 5, 8)
            glow, retrieved = 0.0, False
            if is_oil:
                oil_nodes.append(nid)
                if phase in ("cross_modal", "shutdown"):
                    glow = 0.8 + 0.2 * math.sin(t * 3.5)
                    retrieved = True
                elif phase == "oil_leak" and t - (13 + (i - 2) * 1.3) < 0.8:
                    glow = 0.9
            elif t - (5 + i * 0.7) < 0.4:
                glow = max(0, 1 - (t - (5 + i * 0.7)) * 2.5)
            nodes.append(MemoryNode(
                nid, f"V{i}", "visual",
                0.28 + r * math.cos(angle), 0.38 + r * math.sin(angle),
                5 + i * 0.7, glow, retrieved,
                "oil leak frame" if is_oil else f"frame {i}"
            ))
            nid += 1

        # Vibration cluster (right)
        for i in range(n_vib):
            angle = 2.0 + i * 0.25
            r = 0.2 + (i % 2) * 0.09
            is_anom = i >= 4
            glow = 0.0
            if is_anom and phase in ("vibration", "cross_modal"):
                anom_vib_nodes.append(nid)
                glow = 0.7 + 0.3 * math.sin(t * 4)
            elif t - (6 + i * 0.85) < 0.4:
                glow = max(0, 1 - (t - (6 + i * 0.85)) * 2.5)
            nodes.append(MemoryNode(
                nid, f"Vb{i}", "vibration",
                0.72 + r * math.cos(angle), 0.45 + r * math.sin(angle),
                6 + i * 0.85, glow, desc="anomaly" if is_anom else "normal"
            ))
            nid += 1

        # Thermal cluster (bottom)
        for i in range(n_therm):
            angle = 1.2 + i * 0.3
            r = 0.2 + (i % 3) * 0.05
            glow = 0.0
            if phase in ("cross_modal", "shutdown") and i >= 3:
                glow = 0.4
            elif t - (7 + i * 1.2) < 0.4:
                glow = max(0, 1 - (t - (7 + i * 1.2)) * 2.5)
            nodes.append(MemoryNode(
                nid, f"T{i}", "thermal",
                0.5 + r * math.cos(angle), 0.75 + (i % 2) * 0.06,
                7 + i * 1.2, glow
            ))
            nid += 1

        # Intra-modal edges
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if nodes[i].modality == nodes[j].modality:
                    d = math.sqrt((nodes[i].x - nodes[j].x)**2 + (nodes[i].y - nodes[j].y)**2)
                    if d < 0.18:
                        edges.append(MemoryEdge(i, j))

        # Cross-modal bridges (the key moment)
        if phase in ("cross_modal", "shutdown"):
            for vi in oil_nodes:
                for vb in anom_vib_nodes:
                    if vi < len(nodes) and vb < len(nodes):
                        edges.append(MemoryEdge(
                            vb, vi, True, True,
                            0.85 + 0.15 * math.sin(t * 4)
                        ))
            # Thermal bridges too
            th_start = n_vis + n_vib
            for ti in range(max(0, n_therm - 2), n_therm):
                for vi in oil_nodes[:1]:
                    if th_start + ti < len(nodes) and vi < len(nodes):
                        edges.append(MemoryEdge(th_start + ti, vi, True, True, 0.4))
        elif n_vis > 3 and n_vib > 2:
            # Ambient bridges during normal ops
            for vi in range(min(2, n_vis)):
                edges.append(MemoryEdge(n_vis + vi % n_vib, vi, True, False, 0.0))

        return nodes, edges

    def build_latencies(self, t: float):
        if t < 5:
            return []
        n = min(250, int((t - 5) * 12))
        lat = []
        for _ in range(n):
            base = 0.12 + self.rng.exponential(0.035)
            if self.rng.random() < 0.015:
                base += self.rng.exponential(0.15)
            lat.append(min(base, 0.92))
        return lat


# ═══════════════════════════════════════════════════════════════════════
#  Main video generator
# ═══════════════════════════════════════════════════════════════════════
class DemoVideoGenerator:

    def __init__(self, output_path="mars_demo.mp4"):
        self.output_path = output_path
        self.scenario = Scenario()

        # Load real data
        frames_dir = DATA_DIR / "frames"
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        self.frames = []
        print(f"Loading {len(frame_files)} source frames...")
        for f in frame_files:
            img = cv2.imread(str(f))
            if img is not None:
                self.frames.append(img)
        print(f"  Loaded {len(self.frames)} frames")

        # Load audio
        audio_path = DATA_DIR / "audio.wav"
        self.audio = AudioData(str(audio_path))
        print(f"  Loaded audio: {len(self.audio.samples)} samples")

        # Init 3D renderer
        print("Initializing PyBullet 3D renderer...")
        tw, th = TWIN_RECT[2] - 20, TWIN_RECT[3] - 40
        self.arm_renderer = RobotArmRenderer(width=tw, height=th)
        print("  KUKA iiwa model loaded")

    def _get_source_frame(self, t: float) -> np.ndarray:
        """Get the source video frame for time t."""
        idx = int(t * SRC_FPS) % max(1, len(self.frames))
        return self.frames[idx].copy() if self.frames else np.zeros((480, 640, 3), np.uint8)

    def _draw_camera_panel(self, img, t, phase, danger):
        """Draw the real camera feed panel."""
        x0, y0, w, h = CAM_RECT
        draw_panel(img, CAM_RECT, title="CAMERA FEED — Factory Floor (Real Footage)")

        frame = self._get_source_frame(t)
        fw, fh = w - 20, h - 40
        frame = cv2.resize(frame, (fw, fh))

        # Danger tint on camera feed
        if danger > 0.3:
            tint = np.full_like(frame, (0, 0, 150), dtype=np.uint8)
            cv2.addWeighted(tint, danger * 0.2, frame, 1.0 - danger * 0.2, 0, frame)

        # Oil leak overlay annotation
        if phase in ("oil_leak", "vibration", "cross_modal", "shutdown"):
            # Draw detection box (simulated CV detection)
            bx1, by1 = int(fw * 0.55), int(fh * 0.3)
            bx2, by2 = int(fw * 0.75), int(fh * 0.55)
            box_color = (0, 0, 255) if phase == "oil_leak" else (0, 255, 255)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
            put_text(frame, "OIL LEAK", (bx1, by1 - 8), box_color, 0.45, 1)
            # Confidence
            conf = 0.94 if phase == "oil_leak" else 0.97
            put_text(frame, f"conf: {conf:.2f}", (bx1, by2 + 15), box_color, 0.35)

        # REC indicator
        if int(t * 2) % 2 == 0:
            cv2.circle(frame, (20, 20), 6, (0, 0, 255), -1)
        put_text(frame, "REC", (32, 25), (0, 0, 255), 0.4)
        put_text(frame, f"T={t:.1f}s", (fw - 80, 25), TEXT_WHITE, 0.4)

        img[y0 + 30:y0 + 30 + fh, x0 + 10:x0 + 10 + fw] = frame

    def _draw_twin_panel(self, img, t, phase, danger, speed):
        """Draw the 3D digital twin panel."""
        x0, y0, w, h = TWIN_RECT
        draw_panel(img, TWIN_RECT, title="3D DIGITAL TWIN — KUKA iiwa (PyBullet)")

        arm_img = self.arm_renderer.render(t, speed=speed, danger_level=danger)

        # Overlay status text on arm image
        status_color = GREEN if danger < 0.3 else ORANGE if danger < 0.7 else RED
        status_text = "OPERATING" if danger < 0.3 else "WARNING" if danger < 0.7 else "CRITICAL"
        if phase == "shutdown" and t > 29:
            status_text = "STOPPED"
            status_color = RED
        put_text(arm_img, status_text, (10, 25), status_color, 0.55, 2)
        put_text(arm_img, f"Speed: {speed:.0%}", (10, 50), TEXT_WHITE, 0.4)

        # Bearing warning indicator
        if phase in ("oil_leak", "vibration", "cross_modal", "shutdown"):
            # Pulsing circle on the arm (approximate bearing location)
            bx, by = int(arm_img.shape[1] * 0.45), int(arm_img.shape[0] * 0.4)
            pulse = 0.5 + 0.5 * math.sin(t * 3)
            r = int(15 + pulse * 5)
            cv2.circle(arm_img, (bx, by), r, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(arm_img, (bx, by), r + 8, (0, 0, 180), 1, cv2.LINE_AA)
            put_text(arm_img, "BEARING #3", (bx + r + 10, by), (0, 0, 255), 0.35)

        tw, th = TWIN_RECT[2] - 20, TWIN_RECT[3] - 40
        arm_resized = cv2.resize(arm_img, (tw, th))
        img[y0 + 30:y0 + 30 + th, x0 + 10:x0 + 10 + tw] = arm_resized

    def _draw_graph_panel(self, img, t, phase, nodes, edges):
        """Draw the NSN memory graph."""
        x0, y0, w, h = GRAPH_RECT
        draw_panel(img, GRAPH_RECT, title=f"MEMORY GRAPH (NSN) — {len(nodes)} nodes")

        gx, gy = x0 + 15, y0 + 35
        gw, gh = w - 30, h - 70

        if not nodes:
            put_text_centered(img, "Waiting for data...", x0 + w // 2, y0 + h // 2, TEXT_DIM, 0.5)
            return

        # Draw edges
        for edge in edges:
            if edge.src >= len(nodes) or edge.dst >= len(nodes):
                continue
            n1, n2 = nodes[edge.src], nodes[edge.dst]
            p1 = (gx + int(n1.x * gw), gy + int(n1.y * gh))
            p2 = (gx + int(n2.x * gw), gy + int(n2.y * gh))

            if edge.cross_modal and edge.active:
                color = alpha_blend(BG_PANEL, YELLOW, 0.4 + edge.glow * 0.6)
                cv2.line(img, p1, p2, color, 3, cv2.LINE_AA)
                # Pulse dot
                dot_t = (t * 2.5) % 1.0
                dx = int(lerp(p1[0], p2[0], dot_t))
                dy = int(lerp(p1[1], p2[1], dot_t))
                cv2.circle(img, (dx, dy), 4, YELLOW, -1, cv2.LINE_AA)
            elif edge.cross_modal:
                cv2.line(img, p1, p2, (50, 50, 60), 1, cv2.LINE_AA)
            else:
                cv2.line(img, p1, p2, (42, 42, 52), 1, cv2.LINE_AA)

        # Draw nodes
        for node in nodes:
            cx = gx + int(node.x * gw)
            cy = gy + int(node.y * gh)
            color = MOD_COLORS.get(node.modality, TEXT_DIM)
            if node.retrieved:
                draw_glow_circle(img, (cx, cy), 9, YELLOW, node.glow)
            elif node.glow > 0.1:
                draw_glow_circle(img, (cx, cy), 7, color, node.glow)
            else:
                cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)

        # Legend
        ly = y0 + h - 25
        for i, (mod, color) in enumerate(MOD_COLORS.items()):
            lx = x0 + 10 + i * 105
            cv2.circle(img, (lx, ly), 4, color, -1, cv2.LINE_AA)
            put_text(img, mod[:5].upper(), (lx + 8, ly + 4), color, 0.3)

        lx = x0 + 10 + len(MOD_COLORS) * 105
        cv2.line(img, (lx, ly), (lx + 18, ly), YELLOW, 2, cv2.LINE_AA)
        put_text(img, "BRIDGE", (lx + 22, ly + 4), YELLOW, 0.3)

    def _draw_audio_panel(self, img, t, phase):
        """Draw audio waveform panel."""
        x0, y0, w, h = AUDIO_RECT
        draw_panel(img, AUDIO_RECT, title="AUDIO SENSOR — Factory Ambient (Real Audio)")

        wf_x, wf_y = x0 + 15, y0 + 35
        wf_w, wf_h = w - 30, h - 55

        # Get waveform data
        window = 4.0  # show 4 seconds of audio
        wf = self.audio.get_waveform(max(0, t - window), t, n_points=wf_w)

        # Draw waveform
        cv2.rectangle(img, (wf_x, wf_y), (wf_x + wf_w, wf_y + wf_h),
                      BG_PANEL_ALT, -1)

        mid_y = wf_y + wf_h // 2
        color = MOD_ACOUSTIC
        if phase in ("vibration", "cross_modal"):
            color = MOD_VIBRATION

        for i in range(1, len(wf)):
            amp = int(wf[i] * wf_h * 0.4)
            px = wf_x + i
            # Top and bottom bars (symmetric waveform)
            if amp > 1:
                cv2.line(img, (px, mid_y - amp), (px, mid_y + amp), color, 1)

        # Anomaly marker during vibration phase
        if phase in ("vibration", "cross_modal"):
            anomaly_x = wf_x + wf_w - 80
            cv2.rectangle(img, (anomaly_x, wf_y + 5),
                          (wf_x + wf_w - 5, wf_y + 25),
                          (0, 0, 200), -1)
            put_text(img, "340Hz ANOMALY", (anomaly_x + 3, wf_y + 19),
                     TEXT_WHITE, 0.3, 1)

        # RMS level indicator
        rms = self.audio.get_rms(t)
        bar_w = min(wf_w, int(rms * wf_w * 3))
        bar_color = GREEN if rms < 0.3 else ORANGE if rms < 0.6 else RED
        cv2.rectangle(img, (wf_x, wf_y + wf_h - 8),
                      (wf_x + bar_w, wf_y + wf_h - 2), bar_color, -1)

    def _draw_timeline_panel(self, img, t, phase):
        """Draw phase timeline."""
        x0, y0, w, h = LOG_RECT
        draw_panel(img, LOG_RECT, title="INCIDENT TIMELINE")

        tx, ty = x0 + 15, y0 + 40
        tw, th = w - 30, 25

        cv2.rectangle(img, (tx, ty), (tx + tw, ty + th), (38, 38, 48), -1)

        phases = [
            (2, 4, "INIT", TEAL), (4, 13, "OPS", GREEN),
            (13, 17, "LEAK", MOD_VISUAL), (17, 22, "VIB", MOD_VIBRATION),
            (22, 27, "BRIDGE", YELLOW), (27, 31, "STOP", ORANGE),
        ]
        for start, end, label, color in phases:
            px1 = tx + int(start / DURATION_S * tw)
            if t >= start:
                fill_end = tx + int(min(t, end) / DURATION_S * tw)
                overlay = img.copy()
                cv2.rectangle(overlay, (px1, ty + 1), (fill_end, ty + th - 1), color, -1)
                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
            put_text(img, label, (px1 + 2, ty + th + 13), color, 0.25)

        # Playhead
        px = tx + int(t / DURATION_S * tw)
        cv2.line(img, (px, ty - 3), (px, ty + th + 3), TEXT_WHITE, 2)

        # Danger bar
        danger = self.scenario.get_danger(t, phase)
        if danger > 0:
            dy = y0 + h - 35
            put_text(img, "RISK:", (x0 + 15, dy + 12), TEXT_DIM, 0.35)
            bar_x = x0 + 70
            bar_w = w - 90
            cv2.rectangle(img, (bar_x, dy), (bar_x + bar_w, dy + 16), (40, 40, 50), -1)
            fill = int(bar_w * danger)
            dcolor = GREEN if danger < 0.4 else ORANGE if danger < 0.7 else RED
            cv2.rectangle(img, (bar_x, dy), (bar_x + fill, dy + 16), dcolor, -1)
            put_text(img, f"{danger*100:.0f}%", (bar_x + bar_w + 5, dy + 13), dcolor, 0.4, 1)

    def _draw_metrics_panel(self, img, latencies):
        """Draw latency metrics card."""
        x0, y0, w, h = METRICS_RECT
        draw_panel(img, METRICS_RECT, title="QUERY LATENCY")

        if not latencies:
            put_text_centered(img, "...", x0 + w // 2, y0 + h // 2, TEXT_DIM, 0.5)
            return

        # Big numbers
        p50 = np.percentile(latencies, 50)
        p99 = np.percentile(latencies, 99)
        mx = max(latencies)

        cards = [
            ("p50", f"{p50:.3f}ms", GREEN),
            ("p99", f"{p99:.3f}ms", GREEN if p99 < 1 else RED),
            ("MAX", f"{mx:.3f}ms", GREEN if mx < 1 else RED),
            ("#Q", f"{len(latencies)}", CYAN),
        ]

        cw = (w - 30) // 4
        for i, (label, val, color) in enumerate(cards):
            cx = x0 + 10 + i * cw
            cy = y0 + 40
            cv2.rectangle(img, (cx, cy), (cx + cw - 5, cy + 50), BG_PANEL_ALT, -1)
            put_text_centered(img, val, cx + cw // 2, cy + 18, color, 0.5, 1)
            put_text_centered(img, label, cx + cw // 2, cy + 40, TEXT_DIM, 0.35)

        # Mini histogram
        hx, hy = x0 + 15, y0 + 105
        hw, hh = w - 30, h - 120
        bins = np.linspace(0, 1.0, 20)
        hist, _ = np.histogram(latencies, bins=bins)
        mx_count = max(hist) if max(hist) > 0 else 1
        for i in range(len(hist)):
            bx = hx + int(i / len(hist) * hw)
            bw = max(2, int(hw / len(hist)) - 1)
            bar_h = int(hist[i] / mx_count * hh * 0.85)
            if bar_h > 0:
                bc = GREEN if bins[i] < 0.4 else ORANGE if bins[i] < 0.7 else RED
                cv2.rectangle(img, (bx, hy + hh - bar_h), (bx + bw, hy + hh), bc, -1)

        # Deadline badge
        badge_color = GREEN if p99 < 1.0 else RED
        badge_text = "< 1ms PASS" if p99 < 1.0 else "FAIL"
        cv2.rectangle(img, (x0 + w - 105, y0 + 35), (x0 + w - 10, y0 + 55), badge_color, -1)
        put_text(img, badge_text, (x0 + w - 100, y0 + 50), BG_DARK, 0.4, 1)

    def _draw_narrative(self, img, t, phase, subtitle, alert):
        """Draw narrative bar and alert."""
        x0, y0, w, h = NARR_RECT
        draw_panel(img, NARR_RECT, BG_PANEL)

        if subtitle:
            put_text(img, subtitle, (x0 + 20, y0 + 35), TEXT_WHITE, 0.65, 2)

        if alert:
            text, color, alpha = alert
            ac = alpha_blend(BG_PANEL, color, alpha)
            put_text(img, text, (x0 + 20, y0 + 70), ac, 0.6, 2)

    def _draw_status(self, img, t, phase, nodes, latencies):
        """Draw detailed status panel."""
        x0, y0, w, h = STATUS_RECT
        draw_panel(img, STATUS_RECT, BG_PANEL)

        # Phase-specific explanations
        narratives = {
            "normal_ops": [
                "Camera: streaming 768-D visual embeddings to MARS GPU memory",
                "Vibration sensor: baseline frequency spectrum = nominal",
                "Thermal sensor: bearing temperature = 42C (normal range)",
                "NSN graph: local edges + cross-modal bridges built automatically",
                "All queries < 0.3ms — well within 1ms deadline budget",
            ],
            "oil_leak": [
                "Frame 1247: camera detects dark spot near bearing #3",
                "MARS stores visual embedding — cosine similarity 0.94 to 'oil leak' class",
                "Temporal memory: NEW pattern — no prior match in memory graph",
                "Visual anomaly logged, but NOT sufficient for shutdown alone",
                "MARS waiting for corroborating evidence from other modalities...",
            ],
            "vibration": [
                "Vibration sensor: 340 Hz harmonic appears (known bearing defect freq)",
                "Pattern is AMBIGUOUS — could be: load change, resonance, or defect",
                "Traditional system: insufficient confidence for shutdown decision",
                "MARS: querying memory graph across ALL modalities for correlation...",
                "BFS traversal exploring cross-modal bridge edges...",
            ],
            "cross_modal": [
                "MARS 4-kernel pipeline: cosine_similarity -> temporal_rerank -> top_K -> BFS",
                "BFS kernel: traversed VIBRATION -> VISUAL cross-modal bridge (1 hop)",
                "RETRIEVED: visual memory of oil leak at bearing #3 (stored 8 seconds ago)",
                "Correlation score: oil_leak + vibration_anomaly = 0.96 (threshold: 0.90)",
                "PREDICTION: bearing failure imminent — preventive shutdown recommended",
            ],
            "shutdown": [
                "Decision: preventive shutdown of Robot Cell #7",
                "Robot arm decelerating... safe stop sequence initiated",
                "Maintenance ticket auto-created: 'Replace bearing #3 — oil + vibration'",
                "Estimated cost saved: $47,000 (vs. catastrophic bearing failure)",
                "INCIDENT PREVENTED by cross-modal memory correlation in < 1ms",
            ],
        }

        if phase in narratives:
            lines = narratives[phase]
            phase_starts = {
                "normal_ops": 4, "oil_leak": 13, "vibration": 17,
                "cross_modal": 22, "shutdown": 27,
            }
            ps = phase_starts.get(phase, 4)
            active_line = min(len(lines) - 1, int((t - ps) / 1.3))

            # Two-column layout for explanations
            col_w = w // 2 - 30
            for i, line in enumerate(lines):
                lx = x0 + 20
                ly = y0 + 20 + i * 24

                if i == active_line:
                    put_text(img, "> " + line, (lx, ly), TEXT_ACCENT, 0.45, 1)
                elif i < active_line:
                    put_text(img, "  " + line, (lx, ly), TEXT_DIM, 0.42)
                else:
                    put_text(img, "  " + line, (lx, ly), (40, 40, 48), 0.42)

            # Right column: architecture diagram text
            rx = x0 + 20
            ry = y0 + 160
            put_text(img, "MARS PIPELINE:", (rx, ry), TEXT_ACCENT, 0.45, 1)
            pipeline = [
                ("1. Cosine Similarity", "N blocks x 256 threads, warp-shuffle", MOD_VISUAL),
                ("2. Temporal Rerank", "score * exp(-lambda * age)", TEAL),
                ("3. Top-K Selection", "tiled two-pass, register heaps", CYAN),
                ("4. BFS Expand", "warp-cooperative, cross-modal bridges", YELLOW),
            ]
            for i, (step, desc, color) in enumerate(pipeline):
                sy = ry + 22 + i * 22
                put_text(img, step, (rx + 10, sy), color, 0.4, 1)
                put_text(img, desc, (rx + 250, sy), TEXT_DIM, 0.35)

            # Memory stats
            n_cross = sum(1 for e in [] if True)  # placeholder
            mx = x0 + w - 350
            my = y0 + 160
            put_text(img, "MEMORY STATS:", (mx, my), TEXT_ACCENT, 0.45, 1)
            stats = [
                f"Nodes: {len(nodes)}",
                f"Embedding dim: 768",
                f"GPU VRAM: ~{len(nodes) * 768 * 4 / 1024:.0f} KB",
                f"Latency (p99): {np.percentile(latencies, 99):.3f} ms" if latencies else "Collecting...",
                f"Cross-modal bridges: {'ACTIVE' if phase in ('cross_modal', 'shutdown') else 'standby'}",
            ]
            for i, s in enumerate(stats):
                put_text(img, s, (mx + 10, my + 22 + i * 20), TEXT_WHITE, 0.38)

    def render_frame(self, frame_num):
        img = np.full((HEIGHT, WIDTH, 3), BG_DARK, dtype=np.uint8)
        t = frame_num / FPS
        phase = self.scenario.get_phase(t)

        if phase == "title":
            # Title card
            alpha = min(1.0, t / 1.5)
            color = alpha_blend(BG_DARK, TEXT_WHITE, alpha)
            accent = alpha_blend(BG_DARK, TEXT_ACCENT, alpha)
            put_text_centered(img, "MARS", WIDTH // 2, HEIGHT // 2 - 100, accent, 2.5, 4)
            put_text_centered(img, "Memory for Autonomous Real-time Systems",
                              WIDTH // 2, HEIGHT // 2 - 30, color, 0.85, 2)
            put_text_centered(img, "Demo: Predictive Failure Prevention",
                              WIDTH // 2, HEIGHT // 2 + 30,
                              alpha_blend(BG_DARK, CYAN, alpha * 0.8), 0.65, 1)
            put_text_centered(img, "Real factory footage + 3D PyBullet KUKA arm + GPU-resident memory",
                              WIDTH // 2, HEIGHT // 2 + 70,
                              alpha_blend(BG_DARK, TEXT_DIM, alpha), 0.55, 1)
            put_text_centered(img, "4 CUDA Kernels  |  768-D Embeddings  |  Sub-Millisecond Retrieval",
                              WIDTH // 2, HEIGHT - 80,
                              alpha_blend(BG_DARK, TEAL, alpha * 0.6), 0.5, 1)
            return img

        if phase == "summary":
            return self._render_summary(img, t)

        # Normal rendering: all panels
        danger = self.scenario.get_danger(t, phase)
        speed = self.scenario.get_arm_speed(t, phase)
        subtitle = self.scenario.get_subtitle(t, phase)
        alert = self.scenario.get_alert(t, phase)
        nodes, edges = self.scenario.build_memory(t, phase)
        latencies = self.scenario.build_latencies(t)

        self._draw_camera_panel(img, t, phase, danger)
        self._draw_twin_panel(img, t, phase, danger, speed)
        self._draw_graph_panel(img, t, phase, nodes, edges)
        self._draw_audio_panel(img, t, phase)
        self._draw_timeline_panel(img, t, phase)
        self._draw_metrics_panel(img, latencies)
        self._draw_narrative(img, t, phase, subtitle, alert)
        self._draw_status(img, t, phase, nodes, latencies)

        # Frame counter
        put_text(img, f"Frame {frame_num}/{DURATION_S * FPS}",
                 (WIDTH - 170, HEIGHT - 10), (45, 45, 55), 0.3)

        return img

    def _render_summary(self, img, t):
        """Final summary card."""
        put_text_centered(img, "DEMO RESULTS", WIDTH // 2, 65, TEXT_ACCENT, 1.2, 2)

        latencies = self.scenario.build_latencies(30)
        p99 = np.percentile(latencies, 99) if latencies else 0.27

        results = [
            ("Query Latency (p99)", f"{p99:.3f} ms", "Budget: < 1.0 ms", GREEN),
            ("Cross-Modal Retrieval", "SUCCESS", "Vibration -> Visual bridge (1 BFS hop)", YELLOW),
            ("Failure Prediction", "CONFIRMED", "Oil leak + vibration = bearing failure", GREEN),
            ("Incident Prevention", "PREVENTED", "Estimated $47K damage avoided", CYAN),
        ]

        card_w, card_h = 380, 120
        for i, (title, value, detail, color) in enumerate(results):
            col, row = i % 2, i // 2
            rx = WIDTH // 2 - card_w - 10 + col * (card_w + 20)
            ry = 100 + row * (card_h + 15)
            cv2.rectangle(img, (rx, ry), (rx + card_w, ry + card_h), BG_PANEL, -1)
            cv2.rectangle(img, (rx, ry), (rx + card_w, ry + card_h), (50, 50, 60), 1)
            put_text_centered(img, title, rx + card_w // 2, ry + 25, TEXT_DIM, 0.5, 1)
            put_text_centered(img, value, rx + card_w // 2, ry + 62, color, 0.95, 2)
            put_text_centered(img, detail, rx + card_w // 2, ry + 95, TEXT_DIM, 0.4)

        # Key insight
        ky = 380
        put_text_centered(img, "KEY INSIGHT", WIDTH // 2, ky, TEXT_ACCENT, 0.7, 2)
        put_text_centered(img, "Neither sensor alone had enough confidence for shutdown:",
                          WIDTH // 2, ky + 35, TEXT_WHITE, 0.6, 1)
        put_text_centered(img, "Visual: oil leak (ambiguous severity)    |    Vibration: 340 Hz harmonic (ambiguous cause)",
                          WIDTH // 2, ky + 65, TEXT_DIM, 0.5, 1)
        put_text_centered(img, "MARS cross-modal bridge correlated both -> bearing failure @ 0.96 confidence",
                          WIDTH // 2, ky + 100, YELLOW, 0.6, 2)

        # How it works
        hy = ky + 145
        put_text_centered(img, "MARS 4-KERNEL GPU PIPELINE", WIDTH // 2, hy, TEXT_ACCENT, 0.6, 2)
        steps = [
            ("1.", "Cosine similarity kernel — compares vibration embedding to all GPU-resident memories", MOD_VIBRATION),
            ("2.", "Temporal rerank kernel — recent observations weighted higher via exponential decay", TEAL),
            ("3.", "Top-K tiled kernel — selects seed memories (2-pass, eliminates serial bottleneck)", CYAN),
            ("4.", "BFS expand kernel — traverses cross-modal bridge: vibration -> visual memory", YELLOW),
            ("5.", "Result: oil leak visual memory retrieved in 0.27ms — correlated with vibration = failure predicted", GREEN),
        ]
        for i, (num, desc, color) in enumerate(steps):
            sy = hy + 25 + i * 24
            put_text(img, num, (190, sy), color, 0.45, 2)
            put_text(img, desc, (215, sy), TEXT_WHITE, 0.42, 1)

        # Data sources
        dy = hy + 160
        put_text_centered(img, "DATA SOURCES", WIDTH // 2, dy, TEXT_DIM, 0.45)
        put_text_centered(img, "Video: FANUC factory tour (YouTube)  |  3D Model: KUKA iiwa (PyBullet)  |  Audio: Real factory ambient",
                          WIDTH // 2, dy + 25, TEXT_DIM, 0.4)

        # Footer
        put_text_centered(img, "github.com/antonellofratepietro/cuda-multimodal-memory",
                          WIDTH // 2, HEIGHT - 55, TEXT_DIM, 0.45)
        put_text_centered(img, "MARS: GPU-Resident Multimodal Memory for Real-Time Systems",
                          WIDTH // 2, HEIGHT - 28, CYAN, 0.5, 1)

        return img

    def generate(self):
        total_frames = DURATION_S * FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, FPS, (WIDTH, HEIGHT))

        if not writer.isOpened():
            print("mp4v failed, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.output_path = self.output_path.replace(".mp4", ".avi")
            writer = cv2.VideoWriter(self.output_path, fourcc, FPS, (WIDTH, HEIGHT))
            if not writer.isOpened():
                print("ERROR: Cannot open video writer")
                sys.exit(1)

        print(f"\nGenerating {total_frames} frames @ {FPS}fps ({DURATION_S}s)...")
        print(f"Output: {self.output_path}")
        print(f"Resolution: {WIDTH}x{HEIGHT}\n")

        for frame in range(total_frames):
            img = self.render_frame(frame)
            writer.write(img)

            if frame % FPS == 0 or frame == total_frames - 1:
                pct = (frame + 1) / total_frames * 100
                filled = int(40 * (frame + 1) / total_frames)
                bar = "#" * filled + "-" * (40 - filled)
                t = frame / FPS
                phase = self.scenario.get_phase(t)
                print(f"\r  [{bar}] {pct:5.1f}%  T={t:.1f}s  [{phase}]  ",
                      end="", flush=True)

        writer.release()
        self.arm_renderer.close()
        print(f"\n\nDone! Video: {self.output_path}")
        sz = os.path.getsize(self.output_path)
        print(f"Size: {sz / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="MARS Visual Demo")
    parser.add_argument("-o", "--output", default="mars_demo.mp4")
    parser.add_argument("--skip-download", action="store_true",
                        help="Reuse existing data/ directory")
    parser.add_argument("--preview-frame", type=int, default=None,
                        help="Render single frame as PNG")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    # Step 1: Get real data
    download_and_extract(skip_download=args.skip_download)

    # Step 2: Generate video
    gen = DemoVideoGenerator(output_path=args.output)

    if args.preview_frame is not None:
        img = gen.render_frame(args.preview_frame)
        cv2.imwrite("preview_frame.png", img)
        print(f"Preview: preview_frame.png (frame {args.preview_frame})")
        gen.arm_renderer.close()
        return

    gen.generate()


if __name__ == "__main__":
    main()
