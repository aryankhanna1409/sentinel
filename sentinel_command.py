"""Sentinel Command: a presentation-grade HUD around the existing best.onnx (model unchanged).

What it adds on top of tactical_dashboard.py
  * multi-target tracking with persistent IDs, motion trails and velocity
  * casualty logic: a person who is still AND lying down (wide, low box) for 3 s is flagged as a possible casualty
  * top-down radar with bearing and range blips, rescue-priority ranking
  * four imaging modes (RGB, night-vision NIR, thermal, thermal+edge fusion) and a 2x2 tri-spectral wall
  * live detection-density heatmap, event log, lock-on reticle, MP4 recording and snapshots

Run:   python sentinel_command.py                    (webcam)
       python sentinel_command.py --source clip.mp4  (video file)
Keys:  1 RGB  2 NIR  3 THERMAL  4 FUSION  v 2x2 wall  h heatmap  a auto-cycle  r record  s snapshot  q quit
"""
import argparse
import collections
import math
import os
import time

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.onnx")
CONF = 0.33
FOCAL, HUMAN_H, FOV_DEG = 500, 1.7, 70
STILL_SECONDS, STILL_FRACTION, LYING_ASPECT = 3.0, 0.18, 0.9   # still for 3 s (moved < 18% of body size) and box wider than 0.9x its height
NAMES = {0: "person", 1: "face", 2: "hand"}
CYAN, AMBER, RED, GREEN, INK, MUTED = (255, 232, 103), (60, 180, 255), (90, 90, 255), (120, 255, 150), (243, 237, 230), (181, 152, 139)


# --------------------------------------------------------------------------- detector
class Detector:
    """Runs best.onnx. Uses Ultralytics when installed, otherwise plain onnxruntime (lighter, good for Jetson)."""

    def __init__(self, path=MODEL_PATH):
        self.yolo = None
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(path, task="detect")
        except Exception:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            self.size = int(self.sess.get_inputs()[0].shape[2])

    def __call__(self, frame, conf=CONF):
        if self.yolo is not None:
            r = self.yolo.predict(frame, conf=conf, verbose=False)[0]
            return [(*map(float, b.xyxy[0]), float(b.conf[0]), int(b.cls[0])) for b in r.boxes]
        h, w = frame.shape[:2]
        s = self.size
        r = min(s / h, s / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        top, left = (s - nh) // 2, (s - nw) // 2
        canvas = np.full((s, s, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0][0].T
        cls, cf = out[:, 4:].argmax(1), out[:, 4:].max(1)
        keep = cf >= conf
        out, cls, cf = out[keep], cls[keep], cf[keep]
        if not len(out):
            return []
        bx = np.stack([(out[:, 0] - out[:, 2] / 2 - left) / r, (out[:, 1] - out[:, 3] / 2 - top) / r,
                       (out[:, 0] + out[:, 2] / 2 - left) / r, (out[:, 1] + out[:, 3] / 2 - top) / r], 1)
        off = cls * 10000.0   # per-class NMS: shift each class far apart
        rects = [[float(a + o), float(b + o), float(c - a), float(d - b)] for (a, b, c, d), o in zip(bx, off)]
        idx = np.array(cv2.dnn.NMSBoxes(rects, cf.tolist(), conf, 0.7)).reshape(-1)
        return [(max(0, bx[i][0]), max(0, bx[i][1]), min(w, bx[i][2]), min(h, bx[i][3]), float(cf[i]), int(cls[i])) for i in idx]


# --------------------------------------------------------------------------- tracker
class Track:
    def __init__(self, tid, box, conf, t):
        self.id, self.box, self.conf, self.born, self.seen, self.hits = tid, box, conf, t, t, 1
        self.trail = collections.deque(maxlen=40)
        self.hist = collections.deque(maxlen=400)   # (time, cx, cy)
        self.still_since = None
        self.casualty = False
        self.confirmed = False
        self.holding = False

    @property
    def c(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def h(self):
        return max(self.box[3] - self.box[1], 1)

    @property
    def dist(self):
        return round(float(HUMAN_H * FOCAL / self.h), 1)


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


class Tracker:
    def __init__(self, log):
        self.tracks, self.next_id, self.log = [], 1, log

    def update(self, dets, t):
        def score(tr, d):
            b = d[:4]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            near = 1 - math.hypot(cx - tr.c[0], cy - tr.c[1]) / (1.2 * tr.h)    # centre-distance gate, helps fast walkers
            return max(iou(tr.box, b), near * 0.6) if near > 0 else iou(tr.box, b)
        pairs = sorted(((score(tr, d), i, j) for i, tr in enumerate(self.tracks) for j, d in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for sc, i, j in pairs:
            if sc < 0.15 or i in used_t or j in used_d:
                continue
            used_t.add(i); used_d.add(j)
            tr, d = self.tracks[i], dets[j]
            tr.box = tuple(0.6 * np.array(d[:4]) + 0.4 * np.array(tr.box))   # light smoothing
            tr.conf, tr.seen, tr.hits = d[4], t, tr.hits + 1
        for j, d in enumerate(dets):
            if j not in used_d:
                tr = Track(self.next_id, tuple(d[:4]), d[4], t); self.next_id += 1
                self.tracks.append(tr)
        for tr in self.tracks:
            if tr.seen == t:
                tr.trail.append(tr.c); tr.hist.append((t, *tr.c))
                if tr.hits >= 4 and not tr.confirmed:
                    tr.confirmed = True; self.log(f"ID{tr.id:02d} acquired")
                self._casualty_logic(tr, t)
        for tr in [x for x in self.tracks if t - x.seen > 1.6]:
            if tr.confirmed: self.log(f"ID{tr.id:02d} lost")
            self.tracks.remove(tr)
        return [tr for tr in self.tracks if tr.seen == t and tr.confirmed]

    def _casualty_logic(self, tr, t):
        window = [(ts, x, y) for ts, x, y in tr.hist if t - ts <= STILL_SECONDS]
        if len(window) < 5 or t - tr.born < STILL_SECONDS:
            tr.still_since = None
            tr.casualty = False
            return
        xs, ys = [p[1] for p in window], [p[2] for p in window]
        moved = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        w_box = tr.box[2] - tr.box[0]
        lying = w_box / tr.h >= LYING_ASPECT           # wide, low box = person on the ground
        tr.holding = moved < STILL_FRACTION * tr.h      # standing still, but upright: not an alarm
        if moved < STILL_FRACTION * tr.h and lying:
            tr.still_since = tr.still_since or t
            if not tr.casualty:
                tr.casualty = True; self.log(f"ID{tr.id:02d} STATIONARY + PRONE: possible casualty")
        else:
            if tr.casualty:
                self.log(f"ID{tr.id:02d} moving again")
            tr.still_since, tr.casualty = None, False


# --------------------------------------------------------------------------- imaging modes
def to_mode(frame, mode):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "NIR":
        eq = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
        g = np.zeros_like(frame); g[:, :, 1] = eq; g[:, :, 0] = eq // 6; g[:, :, 2] = eq // 5
        noise = np.random.randint(0, 22, gray.shape, np.uint8)
        g = cv2.add(g, cv2.merge([noise // 4, noise, noise // 4]))
        h, w = gray.shape
        k = np.outer(cv2.getGaussianKernel(h, h * 0.55), cv2.getGaussianKernel(w, w * 0.55)[:, 0])
        k = (0.25 + 0.75 * k / k.max()).astype(np.float32)
        return (g * k[:, :, None]).astype(np.uint8)
    if mode == "THERMAL":
        eq = cv2.createCLAHE(2.5, (8, 8)).apply(gray)
        return cv2.applyColorMap(eq, cv2.COLORMAP_INFERNO)
    if mode == "FUSION":
        th = cv2.applyColorMap(cv2.createCLAHE(2.5, (8, 8)).apply(gray), cv2.COLORMAP_INFERNO)
        edges = cv2.Canny(gray, 70, 160)
        out = cv2.addWeighted(th, 0.8, frame, 0.2, 0)
        out[edges > 0] = CYAN
        return out
    return frame.copy()


# --------------------------------------------------------------------------- drawing helpers
def panel(img, x1, y1, x2, y2, alpha=0.55):
    sub = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(np.zeros_like(sub), alpha, sub, 1 - alpha, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 55, 37), 1)


def brackets(img, box, col, thick=2, ln=14):
    x1, y1, x2, y2 = map(int, box)
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * ln, py), col, thick); cv2.line(img, (px, py), (px, py + dy * ln), col, thick)


def text(img, s, org, scale=0.5, col=INK, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def reticle(img, c, r, frame_no, col):
    cx, cy = map(int, c)
    a = (frame_no * 6) % 360
    cv2.ellipse(img, (cx, cy), (r, r), 0, a, a + 70, col, 2, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (r, r), 0, a + 180, a + 250, col, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, col, -1, cv2.LINE_AA)


def radar(img, tracks, w, h, frame_no, x0, y0, rad=78):
    cx, cy = x0 + rad, y0 + rad
    panel(img, x0 - 6, y0 - 6, x0 + 2 * rad + 6, y0 + 2 * rad + 6, 0.6)
    for r in (rad // 3, 2 * rad // 3, rad):
        cv2.circle(img, (cx, cy), r, (90, 70, 45), 1, cv2.LINE_AA)
    cv2.line(img, (cx - rad, cy), (cx + rad, cy), (70, 55, 37), 1); cv2.line(img, (cx, cy - rad), (cx, cy + rad), (70, 55, 37), 1)
    sweep = (frame_no * 4) % 360
    ov = img.copy()
    cv2.ellipse(ov, (cx, cy), (rad, rad), 0, sweep - 40, sweep, CYAN, -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.28, img, 0.72, 0, img)
    for tr in tracks:
        bearing = ((tr.c[0] - w / 2) / w) * FOV_DEG
        rng = min(tr.dist, 12.0) / 12.0
        a = math.radians(bearing - 90)
        px, py = int(cx + rng * rad * math.cos(a)), int(cy + rng * rad * math.sin(a))
        col = RED if tr.casualty else CYAN
        r = 4 + (int(2 + 2 * math.sin(frame_no / 4)) if tr.casualty else 0)
        cv2.circle(img, (px, py), r, col, -1, cv2.LINE_AA)
        text(img, str(tr.id), (px + 6, py - 4), 0.35, col)
    text(img, "RADAR  12m", (x0, y0 + 2 * rad + 20), 0.4, MUTED)


# --------------------------------------------------------------------------- HUD
class Hud:
    def __init__(self):
        self.events = collections.deque(maxlen=7)
        self.t0 = time.time()
        self.heat = None

    def log(self, msg, now=None):
        self.events.append((now if now is not None else time.time() - self.t0, msg))

    def render(self, base, tracks, dets, mode, fps, frame_no, t, show_heat, wall_label=None):
        h, w = base.shape[:2]
        out = base
        if show_heat:
            if self.heat is None or self.heat.shape[:2] != (h, w):
                self.heat = np.zeros((h, w), np.float32)
            self.heat *= 0.985
            for tr in tracks:
                cv2.circle(self.heat, tuple(map(int, tr.c)), int(tr.h * 0.35), 1.0, -1)
            hm = cv2.GaussianBlur(self.heat, (0, 0), 22)
            hm8 = np.clip(hm / max(hm.max(), 1e-3) * 255, 0, 255).astype(np.uint8)
            col = cv2.applyColorMap(hm8, cv2.COLORMAP_TURBO)
            m = (hm8 > 70)[:, :, None]
            out[:] = np.where(m, cv2.addWeighted(out, 0.72, col, 0.28, 0), out)
        for d in dets:                                   # faces / hands: small, unobtrusive
            if d[5] != 0:
                brackets(out, d[:4], AMBER, 1, 6)
        casualties = [tr for tr in tracks if tr.casualty]
        ranked = sorted(tracks, key=lambda x: (not x.casualty, x.dist))
        primary = ranked[0] if ranked else None
        for tr in tracks:
            if len(tr.trail) > 1:
                pts = np.array(tr.trail, np.int32)
                for i in range(1, len(pts)):
                    a = i / len(pts)
                    cv2.line(out, tuple(pts[i - 1]), tuple(pts[i]), tuple(int(c * (0.3 + 0.7 * a)) for c in (RED if tr.casualty else CYAN)), 2, cv2.LINE_AA)
            col = RED if tr.casualty else CYAN
            brackets(out, tr.box, col, 2, 16 if tr is primary else 10)
            x1, y1 = int(tr.box[0]), int(tr.box[1])
            tag = f"ID{tr.id:02d}  {tr.dist}m  {tr.conf:.2f}"
            text(out, tag, (x1, max(y1 - 8, 100)), 0.42, col)
            if tr.casualty:
                text(out, "CASUALTY?", (x1, int(tr.box[3]) + 16), 0.5, RED, 2)
            elif tr.holding:
                text(out, "holding", (x1, int(tr.box[3]) + 14), 0.4, AMBER)
        if primary:
            reticle(out, primary.c, int(primary.h * 0.32) + 8, frame_no, RED if primary.casualty else GREEN)
            cv2.line(out, (w // 2, h // 2), tuple(map(int, primary.c)), (90, 90, 40), 1, cv2.LINE_AA)
        cv2.line(out, (w // 2 - 14, h // 2), (w // 2 + 14, h // 2), INK, 1); cv2.line(out, (w // 2, h // 2 - 14), (w // 2, h // 2 + 14), INK, 1)

        # directive
        directive, dcol = "SCANNING...", MUTED
        if casualties:
            tgt = min(casualties, key=lambda x: x.dist)
            directive, dcol = f"PRIORITY: ID{tgt.id:02d} DOWN at {tgt.dist}m: DISPATCH RESCUE", RED
        elif primary:
            err = primary.c[0] - w / 2
            if primary.dist < 1.0:
                directive, dcol = "PROXIMITY: STOP", RED
            elif abs(err) > 50:
                directive, dcol = (f"STEER LEFT {abs(int(err))}px" if err < 0 else f"STEER RIGHT {abs(int(err))}px"), AMBER
            else:
                directive, dcol = "LOCK ENGAGED: APPROACH", GREEN

        # top bar
        panel(out, 0, 0, w, 62, 0.62)
        text(out, "SENTINEL // COMMAND", (14, 24), 0.62, CYAN, 2)
        label = wall_label or f"MODE {mode}"
        text(out, label, (14, 50), 0.46, INK)
        text(out, f"FPS {fps:4.1f}", (w - 340, 24), 0.5, GREEN)
        text(out, f"TARGETS {len(tracks)}", (w - 250, 24), 0.5, CYAN)
        text(out, f"CASUALTY {len(casualties)}", (w - 140, 24), 0.5, RED if casualties else MUTED)
        text(out, directive, (260, 50), 0.52, dcol, 1)
        if casualties and int(frame_no / 8) % 2 == 0:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), RED, 3)

        # event log (bottom left) and radar (bottom right)
        if w >= 600:
            panel(out, 8, h - 8 - 18 * (len(self.events) + 1), 330, h - 8, 0.55)
            text(out, "EVENT LOG", (16, h - 14 - 18 * len(self.events)), 0.4, MUTED)
            for i, (ts, msg) in enumerate(self.events):
                text(out, f"{ts:6.1f}s  {msg}", (16, h - 14 - 18 * (len(self.events) - 1 - i)), 0.4, RED if "casualty" in msg.lower() else INK)
            radar(out, tracks, w, h, frame_no, w - 170, h - 170, 78)
        # scanline sweep
        y = int((frame_no * 5) % h)
        ov = out.copy(); cv2.line(ov, (0, y), (w, y), CYAN, 2); cv2.addWeighted(ov, 0.18, out, 0.82, 0, out)
        return out


# --------------------------------------------------------------------------- main loop
def run(args):
    det = Detector()
    hud = Hud()
    trk = Tracker(lambda m: hud.log(m, now))
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    mode, wall, heat, auto, rec, writer = "RGB", False, args.heat, args.auto, args.out, None
    modes = ["RGB", "NIR", "THERMAL", "FUSION"]
    frame_no, prev, fps = 0, time.time(), 0.0
    snap_dir = os.path.dirname(os.path.abspath(__file__))
    win = "Sentinel Command"
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (args.width, int(frame.shape[0] * args.width / frame.shape[1])))
        now = (frame_no / native_fps) if not args.source.isdigit() else time.time() - hud.t0   # video time keeps timing deterministic
        dets = det(frame)
        tracks = trk.update([d for d in dets if d[5] == 0], now)
        if auto:
            mode = modes[(frame_no // args.cycle) % len(modes)]
        if wall:
            tiles = [to_mode(frame, m) for m in modes]
            small = [cv2.resize(t, (frame.shape[1] // 2, frame.shape[0] // 2)) for t in tiles]
            for t, m in zip(small, modes):
                text(t, m, (10, 22), 0.55, CYAN, 1)
                s = 0.5
                for tr in tracks:
                    brackets(t, np.array(tr.box) * s, RED if tr.casualty else CYAN, 1, 8)
            out = np.vstack([np.hstack(small[:2]), np.hstack(small[2:])])
            panel(out, 0, out.shape[0] - 28, 520, out.shape[0], 0.7)
            text(out, f"SENTINEL // TRI-SPECTRAL WALL   targets {len(tracks)}   fps {fps:.1f}", (10, out.shape[0] - 10), 0.5, INK)
        else:
            out = hud.render(to_mode(frame, mode), tracks, dets, mode, fps, frame_no, now, heat)
        if args.wall_at and frame_no == args.wall_at:
            wall = True
        if rec and writer is None:
            writer = cv2.VideoWriter(rec, cv2.VideoWriter_fourcc(*"mp4v"), native_fps if native_fps < 60 else 30, (out.shape[1], out.shape[0]))
        if writer is not None:
            writer.write(out)
        if not args.no_window:
            cv2.imshow(win, out)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            elif k in (ord("1"), ord("2"), ord("3"), ord("4")): mode, auto = modes[k - ord("1")], False
            elif k == ord("v"): wall = not wall
            elif k == ord("h"): heat = not heat
            elif k == ord("a"): auto = not auto
            elif k == ord("s"): cv2.imwrite(os.path.join(snap_dir, f"snapshot_{int(time.time())}.png"), out)
            elif k == ord("r"):
                if writer is None: rec = os.path.join(snap_dir, f"sentinel_{int(time.time())}.mp4")
                else: writer.release(); writer, rec = None, None
        frame_no += 1
        t = time.time(); fps = 0.9 * fps + 0.1 / max(t - prev, 1e-6) if fps else 1 / max(t - prev, 1e-6); prev = t
        if args.max_frames and frame_no >= args.max_frames:
            break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--out", default=None, help="record to this mp4")
    ap.add_argument("--no-window", action="store_true", help="headless (for rendering a video)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--auto", action="store_true", help="auto-cycle imaging modes")
    ap.add_argument("--cycle", type=int, default=90, help="frames per mode when auto-cycling")
    ap.add_argument("--wall-at", type=int, default=0, help="(demo) switch to the 2x2 wall at this frame")
    ap.add_argument("--heat", action="store_true", help="start with the heatmap on")
    run(ap.parse_args())
