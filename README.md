# Ping Pong Ball Tracker (macOS)

Lightweight Python/OpenCV tracker for a white ping pong ball using your camera or a video file.

**Architecture / diagrams:** see [`docs/CODEBASE_DIAGRAM.md`](docs/CODEBASE_DIAGRAM.md) (Mermaid: modules, pipeline, calibration flow).

## Web UI (browser)

Run the same tracker in a local website (MJPEG video + status + reset button):

```bash
source .venv/bin/activate   # if you use a venv
pip install -r requirements.txt
python web_server.py --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in your browser. The **camera is opened by Python** on the Mac (grant Terminal/Cursor camera access in **System Settings → Privacy & Security → Camera**).

Useful flags (same as the CLI where applicable):

- `--calibration path/to/ball_hsv.json` — HSV from `pingpong_tracker_calibrated.py`
- `--no-mirror`, `--max-speed-px-s`, real-world scale flags, etc.

## 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Run with your webcam

```bash
python pingpong_tracker.py --source 0
```

- Press `q` or `Esc` to quit.
- Press `O` to reset the **displacement origin** (distance-from-start readout).
- The preview is **mirrored horizontally** by default (natural left/right like a selfie). Use **`--no-mirror`** for the raw camera/video orientation (e.g. when analyzing a file).
- You should also see a `Mask` window (debug view of the white-ball segmentation).
- With a ball visible, the overlay shows **velocity** \((v_x, v_y)\), per-frame **step**, and **disp** from the origin. Without extra flags these are in **pixels** / **px/s** (image coordinates: +x right, +y down).

## Real-world units (meters / m/s)

You need **one number**: **meters per pixel** on the processed frame (after `--max-width` resize). Two ways:

1. **Direct scale** — if you know it (e.g. from a calibration sheet):

   ```bash
   python pingpong_tracker.py --source 0 --m-per-pixel 0.002
   ```

2. **From a known length in the image** — measure how many **pixels** a real object spans in the **same** resolution the script uses (the live window after resize). Example: a **2.74 m** table length spans **900 px** in the image:

   ```bash
   python pingpong_tracker.py --source 0 --reference-meters 2.74 --reference-pixels 900
   ```

   That sets `m_per_pixel = 2.74 / 900`.

The overlay then shows **m/s**, **meters** per step, and **meters** displacement. The status bar shows `1px=…mm`. Use `--overlay-pixel-units` to also show the pixel readout under the meter lines.

**Note:** This is a **uniform** scale (same \(x\) and \(y\)). It is approximate if the camera looks at the table obliquely; for lab-grade accuracy you’d use a full camera calibration / homography to the table plane.

### Scale from ball area (no ruler)

With a known **physical ball diameter** (regulation **40 mm**), you can estimate **m/px** from the **contour area**: assume the blob is a disk, so `r_eq = sqrt(area/π)` px and `m/px ≈ (diameter/2) / r_eq`.

- **Step and velocity** average the scale from **this frame and the previous** (better when the ball moves in depth).
- **Displacement from origin** uses **this frame’s** area only.

```bash
python pingpong_tracker.py --source 0 --scale-from-ball-area
```

Use `--ball-diameter-m 0.04` (default) for non-standard balls. **Do not** combine with `--m-per-pixel` or `--reference-*`. This remains approximate (segmentation noise, perspective).

## 3) Run with a video file

```bash
python pingpong_tracker.py --source /path/to/video.mp4
```

## Optional flags

- `--min-radius 5` and `--max-radius 80` tune detection size.
- `--max-width 1280` caps frame width for speed (great for laptop battery/perf).
- `--trail 32` changes trajectory length.
- `--record tracked.mp4` saves annotated output.
- **`--max-speed-px-s`** — drop detections that would require moving faster than this (pixels/s vs the last **accepted** position). Example: `8000`–`15000` depending on resolution and how fast the ball crosses the frame. Omit to disable.

## Auto-calibrated tracker (recommended)

Use the same ball color as the scene: sample a region on the ball, then track with that HSV range.

- **Drag mode (default):** draw a **tight** box on the **ball only** (including lots of white table in the box makes the mask too inclusive). Press **Space** to lock.
- **Click mode:** click the center of the ball (samples a small patch), then **Space**.

Calibration uses **percentile-based** HSV bands with **tighter** limits for low-saturation (white) balls so the result isn’t “white-washed.”

```bash
python pingpong_tracker_calibrated.py --source 0
```

```bash
# Click once on the ball instead of dragging a box
python pingpong_tracker_calibrated.py --source 0 --calibrate-mode click
```

Save calibration to reuse next time (no UI):

```bash
python pingpong_tracker_calibrated.py --source 0 --save-calibration ball_hsv.json
# Later:
python pingpong_tracker_calibrated.py --source 0 --load-calibration ball_hsv.json
```

Use `--skip-calibration` to run with the same default white-ball HSV as `pingpong_tracker.py` (no calibration UI).

## Notes for a 2025 MacBook Air

- The default settings are designed to run smoothly on Apple Silicon.
- Use good lighting and keep the ball brighter than the background for best results.
- If the background is also white, lower false positives by reducing `--max-radius` and increasing `--min-radius`.
