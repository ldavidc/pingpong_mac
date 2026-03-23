#!/usr/bin/env python3
"""
Ping pong ball tracker with auto-calibration: sample ball color from a ROI or click.

1. Calibration phase: drag a rectangle on the ball (or click on its center), then
   press SPACE to lock HSV bounds.
2. Tracking phase: same as pingpong_tracker.py but using calibrated ranges.

Optional: --load-calibration file.json to skip the UI, or --save-calibration to persist.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from tracker_core import (
    BallKinematicsTracker,
    DEFAULT_LOWER_HSV,
    DEFAULT_UPPER_HSV,
    add_real_world_scale_arguments,
    add_velocity_gate_arguments,
    detect_ball,
    draw_kinematics_overlay,
    equivalent_radius_from_area,
    hsv_bounds_from_roi,
    load_calibration,
    maybe_mirror_frame,
    preprocess_frame,
    resolve_scale_config,
    save_calibration,
    velocity_within_limit,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Track ping pong balls with HSV auto-calibration."
    )
    p.add_argument(
        "--source",
        default="0",
        help="Camera index or video path. Default: 0",
    )
    p.add_argument("--min-radius", type=int, default=5)
    p.add_argument("--max-radius", type=int, default=80)
    p.add_argument("--trail", type=int, default=32)
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument("--record", default="", help="Save annotated mp4 path.")
    p.add_argument(
        "--calibrate-mode",
        choices=("drag", "click"),
        default="drag",
        help="drag: rectangle on ball. click: single click samples patch. Default: drag",
    )
    p.add_argument(
        "--click-patch",
        type=int,
        default=21,
        help="Half-size of square around click (odd total side). Default: 21",
    )
    p.add_argument(
        "--load-calibration",
        default="",
        help="JSON from prior run; skip calibration UI.",
    )
    p.add_argument(
        "--save-calibration",
        default="",
        help="After calibration, write HSV bounds to this JSON path.",
    )
    p.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Use built-in default white-ball HSV (no UI, no file).",
    )
    add_real_world_scale_arguments(p)
    add_velocity_gate_arguments(p)
    p.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring (default: mirror for natural left/right preview).",
    )
    return p


def open_capture(source_text: str) -> cv2.VideoCapture:
    if source_text.isdigit():
        source = int(source_text)
    else:
        source = source_text
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source_text}")
    return cap


def get_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps if fps > 0 else 30.0, size)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for: {path}")
    return writer


def run_calibration(
    cap: cv2.VideoCapture,
    max_width: int,
    mode: str,
    click_patch: int,
    mirror: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interactive calibration. Returns (lower_hsv, upper_hsv).
    """
    window = "Calibrate: drag/click on ball, then SPACE"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    state: dict = {
        "dragging": False,
        "x0": 0,
        "y0": 0,
        "x1": 0,
        "y1": 0,
        "roi": None,  # (x, y, w, h) or None
        "click": None,  # (cx, cy) for click mode
    }

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if mode == "click":
            if event == cv2.EVENT_LBUTTONDOWN:
                state["click"] = (x, y)
                state["roi"] = None
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["x0"], state["y0"] = x, y
            state["x1"], state["y1"] = x, y
            state["roi"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["x1"], state["y1"] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
            x0 = min(state["x0"], state["x1"])
            y0 = min(state["y0"], state["y1"])
            x1 = max(state["x0"], state["x1"])
            y1 = max(state["y0"], state["y1"])
            w, h = x1 - x0, y1 - y0
            if w >= 4 and h >= 4:
                state["roi"] = (x0, y0, w, h)
            state["click"] = None

    cv2.setMouseCallback(window, on_mouse)

    instructions_drag = (
        "Drag TIGHT on the ball only (avoid white table/clothing). SPACE=confirm  R=reset  Q=quit"
    )
    instructions_click = (
        "Click center of ball. SPACE=confirm  R=reset  Q=quit"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("No frame during calibration (end of video?)")

        frame = preprocess_frame(frame, max_width)
        frame = maybe_mirror_frame(frame, mirror)
        h, w = frame.shape[:2]
        display = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if mode == "drag":
            if state["dragging"]:
                cv2.rectangle(
                    display,
                    (state["x0"], state["y0"]),
                    (state["x1"], state["y1"]),
                    (0, 255, 255),
                    2,
                )
            elif state["roi"] is not None:
                rx, ry, rw, rh = state["roi"]
                cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(
                display,
                instructions_drag,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            if state["click"] is not None:
                cx, cy = state["click"]
                half = max(1, click_patch // 2)
                x0, y0 = max(0, cx - half), max(0, cy - half)
                x1, y1 = min(w, cx + half + 1), min(h, cy + half + 1)
                cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
                cv2.drawMarker(
                    display, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2
                )
            cv2.putText(
                display,
                instructions_click,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            cv2.destroyWindow(window)
            raise SystemExit(0)
        if key in (ord("r"), ord("R")):
            state["roi"] = None
            state["click"] = None
            state["dragging"] = False
            continue

        if key == ord(" "):
            roi_tuple: Optional[tuple[int, int, int, int]] = None
            if mode == "drag" and state["roi"] is not None:
                roi_tuple = state["roi"]
            elif mode == "click" and state["click"] is not None:
                cx, cy = state["click"]
                half = max(1, click_patch // 2)
                x0 = max(0, cx - half)
                y0 = max(0, cy - half)
                rw = min(2 * half + 1, w - x0)
                rh = min(2 * half + 1, h - y0)
                if rw >= 3 and rh >= 3:
                    roi_tuple = (x0, y0, rw, rh)

            if roi_tuple is None:
                continue

            rx, ry, rw, rh = roi_tuple
            lower, upper = hsv_bounds_from_roi(hsv, rx, ry, rw, rh)
            cv2.destroyWindow(window)
            return lower, upper


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    m_per_px, use_ball_area, ball_radius_m = resolve_scale_config(args, parser)
    cap = open_capture(args.source)
    mirror = not args.no_mirror

    if args.skip_calibration:
        lower_hsv, upper_hsv = DEFAULT_LOWER_HSV.copy(), DEFAULT_UPPER_HSV.copy()
    elif args.load_calibration:
        lower_hsv, upper_hsv = load_calibration(args.load_calibration)
    else:
        lower_hsv, upper_hsv = run_calibration(
            cap,
            args.max_width,
            args.calibrate_mode,
            args.click_patch,
            mirror,
        )
        if args.save_calibration:
            save_calibration(args.save_calibration, lower_hsv, upper_hsv)
        # Video file: rewind so tracking sees the full clip after calibration frames.
        if not args.source.isdigit():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    points = deque(maxlen=max(args.trail, 0))
    prev_frame_t = time.time()
    fps_smoothed = 0.0
    writer = None
    kin_tracker = BallKinematicsTracker()
    last_area_px: Optional[float] = None
    last_accept_pos: Optional[tuple[float, float]] = None
    last_accept_t: Optional[float] = None
    window_track = "Ping Pong Ball Tracker (calibrated)"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_t = time.time()
        dt_frame = max(frame_t - prev_frame_t, 1e-6)
        prev_frame_t = frame_t

        frame = preprocess_frame(frame, args.max_width)
        frame = maybe_mirror_frame(frame, mirror)
        hit, mask = detect_ball(
            frame, args.min_radius, args.max_radius, lower_hsv, upper_hsv
        )
        raw_hit = hit

        accepted = False
        if hit is not None:
            x, y, radius, area, circularity = hit
            if velocity_within_limit(
                float(x),
                float(y),
                last_accept_pos,
                last_accept_t,
                frame_t,
                args.max_speed_px_s,
            ):
                accepted = True

        if accepted:
            assert raw_hit is not None
            x, y, radius, area, circularity = raw_hit
            center = (x, y)
            points.appendleft(center)
            area_prev = last_area_px
            dt_kin = (
                max(frame_t - last_accept_t, dt_frame)
                if last_accept_t is not None
                else dt_frame
            )
            kin = kin_tracker.update(float(x), float(y), dt_kin)
            last_accept_pos = (float(x), float(y))
            last_accept_t = frame_t
            last_area_px = float(area)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"r={radius}px area={int(area)} circ={circularity:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (60, 220, 60),
                2,
                cv2.LINE_AA,
            )
            draw_kinematics_overlay(
                frame,
                kin,
                y0=55,
                meters_per_pixel=m_per_px,
                overlay_pixel_units=args.overlay_pixel_units,
                ball_area_scale=use_ball_area,
                area_px=float(area),
                area_px_prev=area_prev,
                ball_radius_m=ball_radius_m,
            )
        else:
            points.appendleft(None)
            if raw_hit is None:
                kin_tracker.on_lost()
                last_area_px = None
                last_accept_pos = None
                last_accept_t = None
                cv2.putText(
                    frame,
                    "No ball detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    "Rejected: velocity too high (false positive?)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )

        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            thickness = (
                int(np.sqrt(args.trail / float(i + 1)) * 2.5) if args.trail > 0 else 1
            )
            cv2.line(frame, points[i - 1], points[i], (255, 0, 0), thickness)

        fps = 1.0 / dt_frame
        fps_smoothed = fps if fps_smoothed == 0.0 else (fps_smoothed * 0.9 + fps * 0.1)
        scale_hint = ""
        if use_ball_area and last_area_px is not None:
            req = equivalent_radius_from_area(last_area_px)
            mpp = ball_radius_m / max(req, 1e-6)
            scale_hint = f"  1px={mpp * 1000:.3f}mm (from ball area)"
        elif m_per_px is not None:
            scale_hint = f"  1px={m_per_px * 1000:.3f}mm"
        cv2.putText(
            frame,
            f"FPS: {fps_smoothed:.1f}   O=reset disp. origin{scale_hint}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_track, frame)
        cv2.imshow("Mask", mask)

        if args.record:
            if writer is None:
                writer = get_writer(
                    args.record,
                    cap.get(cv2.CAP_PROP_FPS),
                    (frame.shape[1], frame.shape[0]),
                )
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("o"), ord("O")):
            kin_tracker.reset_origin()
            last_area_px = None
            last_accept_pos = None
            last_accept_t = None

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
