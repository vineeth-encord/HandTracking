"""
Hand Tracking Video Pipeline (WiLoR — CVPR 2025)
--------------------------------------------------
Process video files through WiLoR hand tracking and output
wireframe overlays of detected hands.

Usage:
    python process_video.py input.mp4
    python process_video.py input.mp4 --output result.mp4
    python process_video.py input.mp4 --mode wireframe
    python process_video.py videos/           # batch process a directory
    python process_video.py videos/ --output tracked_videos/

Setup (run once):
    pip install torch torchvision
    pip install git+https://github.com/warmshao/WiLoR-mini
    pip install opencv-python tqdm
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )
except ImportError:
    print("[ERROR] Required packages not installed. Run:", file=sys.stderr)
    print("  pip install torch torchvision", file=sys.stderr)
    print("  pip install git+https://github.com/warmshao/WiLoR-mini", file=sys.stderr)
    sys.exit(1)

# ── Hand skeleton connections (MediaPipe/WiLoR 21-landmark layout) ────────────
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),      # Thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),      # Index finger
    (0, 9),  (9, 10), (10, 11),(11, 12),    # Middle finger
    (0, 13),(13, 14),(14, 15),(15, 16),     # Ring finger
    (0, 17),(17, 18),(18, 19),(19, 20),     # Pinky
    (5, 9), (9, 13),(13, 17),               # Palm knuckle connections
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


# ── Device detection ──────────────────────────────────────────────────────────

def get_device() -> str:
    """Auto-select best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    # MPS (Apple Silicon) has tensor stride incompatibilities with WiLoR's ViT backbone
    # — fall back to CPU
    return "cpu"


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_hand(canvas: np.ndarray, keypoints_2d: np.ndarray, mode: str) -> None:
    """Draw skeleton connections and landmark dots for one detected hand.

    Args:
        canvas: BGR image to draw onto (modified in-place).
        keypoints_2d: Array of shape [21, 2] in pixel coordinates.
        mode: 'overlay' or 'wireframe'.
    """
    pts = [(int(kp[0]), int(kp[1])) for kp in keypoints_2d]

    line_color = (255, 255, 255) if mode == "overlay" else (0, 200, 255)
    dot_color  = (0, 255, 0)     if mode == "overlay" else (0, 255, 128)

    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], line_color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(canvas, pt, 4, dot_color, -1, cv2.LINE_AA)


# ── Core pipeline ─────────────────────────────────────────────────────────────

def process_video(
    input_path: Path,
    output_path: Path,
    pipe: WiLorHandPose3dEstimationPipeline,
    mode: str,
    max_hands: int,
) -> bool:
    """Process a single video through WiLoR hand tracking and write the output."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}", file=sys.stderr)
        return False

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"[ERROR] Cannot create output file: {output_path}", file=sys.stderr)
        cap.release()
        return False

    hands_detected_frames = 0

    with tqdm(total=total_frames, desc=input_path.name, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # WiLoR-mini accepts a BGR numpy array (same as cv2 output)
            outputs = pipe.predict(frame)

            canvas = np.zeros_like(frame) if mode == "wireframe" else frame.copy()

            if outputs:
                hands_detected_frames += 1
                for out in outputs[:max_hands]:
                    # keypoints_2d[0] → shape [21, 2], already in pixel coordinates
                    keypoints_2d = out["wilor_preds"]["pred_keypoints_2d"][0]
                    draw_hand(canvas, keypoints_2d, mode)

            writer.write(canvas)
            pbar.update(1)

    cap.release()
    writer.release()

    pct = (hands_detected_frames / total_frames * 100) if total_frames > 0 else 0
    print(f"  Hands detected in {hands_detected_frames}/{total_frames} frames ({pct:.1f}%)")
    print(f"  Output → {output_path}")
    return True


# ── CLI entry point ───────────────────────────────────────────────────────────

def collect_videos(input_path: Path) -> list:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(
        description="Run WiLoR hand tracking on videos and output wireframe overlays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input video file or directory of videos.")
    parser.add_argument("--output", "-o", help="Output file or directory.")
    parser.add_argument(
        "--mode", choices=["overlay", "wireframe"], default="overlay",
        help="'overlay': wireframe on original (default). 'wireframe': skeleton on black.",
    )
    parser.add_argument("--max-hands", type=int, default=2, metavar="N",
                        help="Max hands to draw per frame (default: 2).")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    # Use float32 on MPS/CPU for stability; float16 only on CUDA
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")
    print("Loading WiLoR model (weights download on first run)…\n")
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

    videos = collect_videos(input_path)
    if not videos:
        print(f"[ERROR] No supported video files found in: {input_path}", file=sys.stderr)
        sys.exit(1)

    is_batch = len(videos) > 1 or input_path.is_dir()
    print(f"Found {len(videos)} video(s)  |  mode: {args.mode}  |  max hands: {args.max_hands}\n")

    success_count = 0
    for video in videos:
        if is_batch and args.output:
            out_path = Path(args.output) / (video.stem + "_tracked" + video.suffix)
        elif is_batch:
            out_path = video.with_name(video.stem + "_tracked" + video.suffix)
        else:
            out_path = Path(args.output) if args.output else video.with_name(
                video.stem + "_tracked" + video.suffix
            )

        if process_video(video, out_path, pipe, args.mode, args.max_hands):
            success_count += 1

    print(f"\nDone. {success_count}/{len(videos)} video(s) processed successfully.")


if __name__ == "__main__":
    main()
