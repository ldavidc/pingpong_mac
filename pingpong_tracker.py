#!/usr/bin/env python3
"""
Track ping pong balls from webcam or video input.

Optimized for lightweight CPU usage on Apple Silicon laptops.
"""

import argparse
import time

import cv2

from tracker_core import (
    DEFAULT_LOWER_HSV,
    DEFAULT_UPPER_HSV,
    add_real_world_scale_arguments,
    add_velocity_gate_arguments,
    resolve_scale_config,
)
from tracking_pipeline import PipelineConfig, TrackingPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track ping pong balls in webcam/video stream."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or path to video file. Default: 0",
    )
    parser.add_argument(
        "--min-radius",
        type=int,
        default=5,
        help="Minimum detected ball radius in pixels. Default: 5",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=80,
        help="Maximum detected ball radius in pixels. Default: 80",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=32,
        help="Number of previous points to draw as trajectory. Default: 32",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Resize frame width cap for performance. Default: 1280",
    )
    parser.add_argument(
        "--record",
        default="",
        help="Optional output path to save annotated video (mp4).",
    )
    add_real_world_scale_arguments(parser)
    add_velocity_gate_arguments(parser)
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring (default: mirror for natural left/right preview).",
    )
    return parser


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


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    m_per_px, use_ball_area, ball_radius_m = resolve_scale_config(args, parser)
    cap = open_capture(args.source)

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
    )
    pipeline = TrackingPipeline(DEFAULT_LOWER_HSV, DEFAULT_UPPER_HSV, pipe_conf)

    writer = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame, mask, _ = pipeline.step(frame)

        cv2.imshow("Ping Pong Ball Tracker", frame)
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
            pipeline.reset_origin()

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
