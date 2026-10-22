<div align="center">

# 🛰️ Sentinel

**A portable AI scanner for search and rescue: real-time, occlusion-aware human detection across RGB, near-infrared and thermal views.**

![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF?style=flat-square&logo=yolo&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-training-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-live%20HUD-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-edge%20inference-005CED?style=flat-square&logo=onnx&logoColor=white)
![Colab](https://img.shields.io/badge/Trained%20on-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

</div>

---

<div align="center">

![Sentinel HUD demo](assets/demo.gif)

<sub>Real run of <code>tactical_dashboard.py</code> with the repo's <code>best.onnx</code> on public sample pedestrian footage (RGB → simulated NIR → simulated thermal).</sub>

</div>

---

## 🎯 Why Sentinel

In disaster zones people are partly buried, hidden behind debris or lying in the dark. A detector tuned for clean street scenes misses them. Sentinel trains a YOLO11n model (the shipped `best.onnx` detects three classes: person, face and hand) with tooling for **CrowdHuman**, a dataset full of heavily overlapping and partially hidden people, and wraps it in a handheld-style heads-up display that works across imaging modes.

---

## ✨ Features

| | |
|---|---|
| 👁️ **Occlusion-aware detection** | Detects person, face and hand; the repo includes CrowdHuman tooling for crowded, partially hidden scenes |
| 🌗 **Multi-spectral modes** | Switch live between **RGB**, simulated **NIR** and simulated **Thermal** (false-colour) |
| 📏 **Range estimate** | Distance to the nearest person from bounding-box height (pinhole model, 1.7 m reference height) |
| 🎯 **Targeting HUD** | Crosshair, target lock on the closest person, and a steering directive: steer left, steer right, approach, or stop |
| ⚡ **Fast inference** | ONNX export for lightweight, CPU-friendly runtime |

---

## 🧠 How it works

```mermaid
flowchart LR
    CAM[Camera frame] --> MODE{Mode}
    MODE -- RGB --> M[YOLO<br/>best.onnx]
    MODE -- NIR<br/>histogram equalised --> M
    MODE -- Thermal<br/>colour-mapped --> M
    M --> T[Closest person<br/>= primary target]
    T --> D[Distance estimate]
    T --> S[Steering directive]
    D --> HUD[Live HUD]
    S --> HUD
```

**Steering logic** (`tactical_dashboard.py`): the horizontal offset between the target and the frame centre is compared against a 50 px dead zone. Inside the zone the HUD shows *LOCK ENGAGED – APPROACH*; outside it shows *STEER LEFT / RIGHT* with the pixel error; under 1 m it raises a *PROXIMITY ALERT – STOP*.

**Detection threshold:** confidence is fixed at **0.33**, the point chosen from the model's F1 curve.

> ℹ️ The NIR and Thermal modes are **software simulations** applied to a normal webcam feed (contrast equalisation and a JET colour map). They demonstrate the pipeline; real NIR or LWIR sensors would plug into the same loop.

---

## 📁 Repository layout

| Path | What it is |
|---|---|
| `best.pt` / `best.onnx` | Trained detector weights (PyTorch and ONNX) |
| `dashboard.py` | Simple live HUD: FPS and person count |
| `tactical_dashboard.py` | Full HUD: multi-spectral modes, range, target lock, steering |
| `colab code.ipynb` | End-to-end Colab workflow: setup, dataset prep, YOLO conversion |
| `notebooks/` | Environment check and dataset download notebooks |
| `scripts/convert_crowdhuman_to_yolo.py` | CrowdHuman `.odgt` annotations → YOLO label format |
| `requirements.txt` | Python dependencies |

---

## 🚀 Quick start

```bash
git clone https://github.com/aryankhanna1409/sentinel.git
cd sentinel
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the full tactical HUD on your default webcam:

```bash
python tactical_dashboard.py
```

| Key | Action |
|---|---|
| `1` | RGB mode |
| `2` | Simulated NIR mode |
| `3` | Simulated thermal mode |
| `q` | Quit |

Prefer the minimal overlay?

```bash
python dashboard.py
```

---

## 🏋️ Reproducing the training

1. Open `colab code.ipynb` in Google Colab (badge inside the notebook) and select a GPU runtime.
2. Download CrowdHuman with `notebooks/02_download_datasets.ipynb`.
3. Convert annotations to YOLO format with `scripts/convert_crowdhuman_to_yolo.py`.
4. Train with Ultralytics and export the best checkpoint to ONNX.

---

<div align="center">

Made by [Aryan Khanna](https://github.com/aryankhanna1409)

</div>
