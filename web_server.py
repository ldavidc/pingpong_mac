#!/usr/bin/env python3
"""
Web UI for the ping pong ball tracker: MJPEG stream + REST controls.

  python web_server.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from tracker_core import (
    DEFAULT_LOWER_HSV,
    DEFAULT_UPPER_HSV,
    add_real_world_scale_arguments,
    add_velocity_gate_arguments,
    load_calibration,
    resolve_scale_config,
)
from tracking_pipeline import PipelineConfig, TrackingPipeline

# --- Shared state ---
pipeline: Optional[TrackingPipeline] = None
pipeline_lock = threading.Lock()
stop_event = threading.Event()
camera_thread: Optional[threading.Thread] = None
latest_jpeg: Optional[bytes] = None
jpeg_lock = threading.Lock()
last_status: str = "starting"
fps_display: float = 0.0


def open_capture(source_text: str) -> cv2.VideoCapture:
    if source_text.isdigit():
        source = int(source_text)
    else:
        source = source_text
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source_text}")
    return cap


def camera_worker(cap: cv2.VideoCapture, pipe: TrackingPipeline) -> None:
    global latest_jpeg, last_status, fps_display
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            last_status = "eof"
            time.sleep(0.05)
            continue
        with pipeline_lock:
            out, _mask, status = pipe.step(frame)
            last_status = status
            fps_display = pipe.fps_smoothed
        ok_enc, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok_enc:
            with jpeg_lock:
                latest_jpeg = buf.tobytes()
        time.sleep(0.001)

    cap.release()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ping pong tracker web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--source", default="0", help="Camera index or video path")
    p.add_argument(
        "--calibration",
        default="",
        help="Optional path to HSV JSON from pingpong_tracker_calibrated.py",
    )
    p.add_argument("--min-radius", type=int, default=5)
    p.add_argument("--max-radius", type=int, default=80)
    p.add_argument("--trail", type=int, default=32)
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument("--no-mirror", action="store_true")
    add_real_world_scale_arguments(p)
    add_velocity_gate_arguments(p)
    return p


def create_app(args: argparse.Namespace) -> FastAPI:
    global pipeline, camera_thread

    parser = build_parser()
    m_per_px, use_ball_area, ball_radius_m = resolve_scale_config(args, parser)

    if args.calibration:
        lower_hsv, upper_hsv = load_calibration(args.calibration)
    else:
        lower_hsv, upper_hsv = DEFAULT_LOWER_HSV.copy(), DEFAULT_UPPER_HSV.copy()

    pipe_conf = PipelineConfig(
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        trail=args.trail,
        max_width=args.max_width,
        mirror=not args.no_mirror,
        max_speed_px_s=args.max_speed_px_s,
        meters_per_pixel=m_per_px,
        use_ball_area=use_ball_area,
        ball_radius_m=ball_radius_m,
        overlay_pixel_units=args.overlay_pixel_units,
        footer_reset_hint="Reset via button →",
    )
    pipeline = TrackingPipeline(lower_hsv, upper_hsv, pipe_conf)

    static_dir = Path(__file__).resolve().parent / "static"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global camera_thread
        stop_event.clear()
        cap = open_capture(args.source)
        camera_thread = threading.Thread(
            target=camera_worker,
            args=(cap, pipeline),
            daemon=True,
        )
        camera_thread.start()
        yield
        stop_event.set()
        if camera_thread is not None:
            camera_thread.join(timeout=3.0)

    app = FastAPI(title="Ping Pong Ball Tracker", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        html_path = static_dir / "index.html"
        if html_path.is_file():
            return html_path.read_text(encoding="utf-8")
        return "<h1>Missing static/index.html</h1>"

    def mjpeg_generator():
        while True:
            with jpeg_lock:
                jpg = latest_jpeg
            if jpg is None:
                time.sleep(0.02)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
            time.sleep(0.025)

    @app.get("/stream")
    async def stream():
        return StreamingResponse(
            mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/status")
    async def status():
        return {
            "status": last_status,
            "fps": round(fps_display, 1),
        }

    @app.post("/api/reset-origin")
    async def reset_origin():
        if pipeline is not None:
            with pipeline_lock:
                pipeline.reset_origin()
        return {"ok": True}

    return app


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
