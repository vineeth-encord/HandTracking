"""
Hand Tracking Video Pipeline
-----------------------------
Process video files through MediaPipe hand tracking and output
wireframe overlays of detected hands.

Usage:
    python process_video.py input.mp4
    python process_video.py input.mp4 --output result.mp4
    python process_video.py input.mp4 --mode wireframe
    python process_video.py videos/           # batch process a directory
    python process_video.py videos/ --output tracked_videos/
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("[ERROR] mediapipe not installed. Run: pip install mediapipe", file=sys.stderr)
    sys.exit(1)

# ── Hand skeleton connections (MediaPipe 21-landmark layout) ──────────────────
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),     # Thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),     # Index finger
    (0, 9),  (9, 10), (10, 11),(11, 12),   # Middle finger
    (0, 13),(13, 14),(14, 15),(15, 16),    # Ring finger
    (0, 17),(17, 18),(18, 19),(19, 20),    # Pinky
    (5, 9), (9, 13),(13, 17),              # Palm knuckle connections
]

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


# ── Model management ──────────────────────────────────────────────────────────

def ensure_model() -> str:
    """Return path to the hand landmark model, downloading it if needed."""
    if MODEL_PATH.exists():
        return str(MODEL_PATH)

    print(f"Hand landmark model not found at {MODEL_PATH}")
    print(f"Downloading (~29 MB) from Google…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.\n")
    except Exception as exc:
        print(f"\n[ERROR] Auto-download failed: {exc}", file=sys.stderr)
        print("Please download the model manually and place it next to process_video.py:", file=sys.stderr)
        print(f"  curl -o '{MODEL_PATH}' '{MODEL_URL}'", file=sys.stderr)
        sys.exit(1)

    return str(MODEL_PATH)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_hand(canvas: np.ndarray, landmarks, width: int, height: int, mode: str) -> None:
    """Draw skeleton connections and landmark dots for one detected hand."""
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

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
    model_path: str,
    mode: str,
    max_hands: int,
    confidence: float,
) -> bool:
    """Process a single video through hand tracking and write the output."""
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

    base_options = mp_python.BaseOptions(
        model_asset_path=model_path,
        delegate=mp_python.BaseOptions.Delegate.CPU,  # force CPU; GPU requires a display context
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=max_hands,
        min_hand_detection_confidence=confidence,
        min_tracking_confidence=confidence,
        running_mode=mp_vision.RunningMode.VIDEO,  # frame-by-frame, no callback
    )

    hands_detected_frames = 0
    frame_index = 0

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        with tqdm(total=total_frames, desc=input_path.name, unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # VIDEO mode requires monotonically increasing timestamps in ms
                timestamp_ms = int((frame_index / fps) * 1000)
                result = detector.detect_for_video(mp_image, timestamp_ms)

                if mode == "wireframe":
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    canvas = frame.copy()

                if result.hand_landmarks:
                    hands_detected_frames += 1
                    for hand_landmarks in result.hand_landmarks:
                        draw_hand(canvas, hand_landmarks, width, height, mode)

                writer.write(canvas)
                frame_index += 1
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
        description="Run hand tracking on videos and output wireframe overlays.",
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
                        help="Max hands to detect per frame (default: 2).")
    parser.add_argument("--confidence", type=float, default=0.5, metavar="F",
                        help="Min detection/tracking confidence 0–1 (default: 0.5).")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    model_path = ensure_model()

    videos = collect_videos(input_path)
    if not videos:
        print(f"[ERROR] No supported video files found in: {input_path}", file=sys.stderr)
        sys.exit(1)

    is_batch = len(videos) > 1 or input_path.is_dir()
    print(f"Found {len(videos)} video(s)  |  mode: {args.mode}  |  "
          f"max hands: {args.max_hands}  |  confidence: {args.confidence}\n")

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

        if process_video(video, out_path, model_path, args.mode, args.max_hands, args.confidence):
            success_count += 1

    print(f"\nDone. {success_count}/{len(videos)} video(s) processed successfully.")


if __name__ == "__main__":
    main()
