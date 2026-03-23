# Ping Pong Tracker — codebase diagrams

These diagrams use [Mermaid](https://mermaid.js.org/). They render in GitHub, many IDEs (including Cursor), and Markdown previewers.

---

## 1. Module layout (dependencies)

```mermaid
flowchart TB
  subgraph deps["External"]
    CV2[OpenCV cv2]
    NP[numpy]
  end

  subgraph project["pingpong_mac"]
    TC["tracker_core.py<br/>shared library"]
    TP["tracking_pipeline.py"]
    PT["pingpong_tracker.py<br/>CLI: default HSV"]
    PTC["pingpong_tracker_calibrated.py<br/>CLI: ROI/calibration"]
    WS["web_server.py<br/>FastAPI + MJPEG"]
  end

  PT --> TC
  PT --> TP
  PTC --> TC
  WS --> TC
  WS --> TP
  TC --> CV2
  TC --> NP
  PT --> CV2
  PTC --> CV2
  WS --> CV2
```

- **`tracker_core`**: detection, masks, kinematics, scaling, argparse helpers, calibration math.
- **`tracking_pipeline`**: shared per-frame overlay + kinematics (CLI + web).
- **`pingpong_tracker`**: OpenCV windows; fixed white-ball HSV.
- **`pingpong_tracker_calibrated`**: calibration UI + same tracking loop; loads/saves JSON HSV.
- **`web_server`**: browser UI at `/`, MJPEG at `/stream`, JSON `/api/status`, POST `/api/reset-origin`.

---

## 2. What lives in `tracker_core.py` (logical blocks)

```mermaid
flowchart LR
  subgraph io["Frame I/O"]
    PRE[preprocess_frame]
    MIR[maybe_mirror_frame]
  end

  subgraph vision["Detection"]
    DET[detect_ball]
    HSV[hsv_bounds_from_roi]
  end

  subgraph motion["Kinematics"]
    BK[BallKinematics]
    BKT[BallKinematicsTracker]
    VEL[velocity_within_limit]
  end

  subgraph world["Real-world"]
    WFP[world_from_px]
    WBA[world_from_ball_area]
    RSC[resolve_scale_config]
  end

  subgraph ui["Overlay"]
    DKO[draw_kinematics_overlay]
  end

  subgraph persist["Calibration files"]
    SAV[save_calibration]
    LOD[load_calibration]
  end

  PRE --> DET
  MIR --> DET
  BKT --> BK
  BK --> WFP
  BK --> WBA
  BK --> DKO
  HSV --> SAV
```

---

## 3. Per-frame pipeline (both trackers)

```mermaid
flowchart TD
  A[VideoCapture.read] --> B[preprocess_frame max_width]
  B --> C[maybe_mirror_frame unless no-mirror]
  C --> D[detect_ball HSV inRange + contours]
  D --> E{Velocity gate<br/>max-speed-px-s?}
  E -->|reject| F[Skip update / show rejected]
  E -->|accept| G[BallKinematicsTracker.update]
  G --> H[draw_kinematics_overlay<br/>px / uniform mpp / ball-area scale]
  G --> I[Draw circle + trail + FPS]
  C --> J[imshow frames + mask]
  H --> J
  I --> J
```

---

## 4. `pingpong_tracker_calibrated.py` only (startup)

```mermaid
flowchart TD
  START[parse_args] --> A{skip_calibration?}
  A -->|yes| B[DEFAULT_LOWER_HSV]
  A -->|no| C{load_calibration json?}
  C -->|yes| D[load_calibration]
  C -->|no| E[run_calibration UI<br/>drag or click ROI]
  E --> F[hsv_bounds_from_roi]
  F --> G[optional save_calibration]
  B --> LOOP[Main tracking loop]
  D --> LOOP
  G --> LOOP
```

---

## 5. Data artifacts

| File | Role |
|------|------|
| `requirements.txt` | `opencv-python`, `numpy` |
| `README.md` | Usage and flags |
| User `*.json` (optional) | `lower_hsv` / `upper_hsv` from calibration |

---

*Generated for the pingpong_mac project; edit this file if you add modules.*
