"""
score_media.py

Scans a folder of photos/video clips and scores each for use as auto-selected
b-roll. Outputs a ranked JSON so build_video.py can pull the best unused
shots for each section without re-scoring every run.

Scoring factors:
  - Sharpness (Laplacian variance) - penalizes blurry/out-of-focus shots
  - Exposure (mean brightness, penalize crushed blacks/blown highlights)
  - Face presence (bonus - people-focused content tends to perform better
    on social, and it's useful signal for which shots suit a close crop)

Usage:
  python score_media.py --folder path/to/broll --out scored_media.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v"}

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def sharpness_score(gray: np.ndarray) -> float:
    """Laplacian variance - higher is sharper/more in-focus."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_score(gray: np.ndarray) -> float:
    """
    1.0 = well exposed (mean brightness near the middle of the range),
    lower = too dark or too blown out.
    """
    mean_brightness = float(np.mean(gray))
    # Penalize distance from a healthy midtone target (~128 on 0-255 scale)
    distance = abs(mean_brightness - 128) / 128.0
    return max(0.0, 1.0 - distance)


def has_face(gray: np.ndarray) -> bool:
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0


def score_image(path: str) -> dict:
    img = cv2.imread(path)
    if img is None:
        return {"error": "unreadable"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = sharpness_score(gray)
    expo = exposure_score(gray)
    face = has_face(gray)
    h, w = gray.shape
    # Normalize sharpness roughly - typical in-focus phone/camera photos
    # land well above 100 on this metric, blurry ones well below.
    sharp_norm = min(1.0, sharp / 300.0)
    overall = (sharp_norm * 0.5) + (expo * 0.3) + (0.2 if face else 0.0)
    return {
        "type": "image",
        "width": w,
        "height": h,
        "sharpness_raw": round(sharp, 2),
        "exposure_score": round(expo, 3),
        "has_face": face,
        "overall_score": round(overall, 3),
    }


def score_video(path: str, sample_frames: int = 5) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "unreadable"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = frame_count / fps if fps else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count <= 0:
        cap.release()
        return {"error": "no_frames"}

    sample_indices = np.linspace(0, frame_count - 1, num=min(sample_frames, frame_count), dtype=int)
    sharp_vals, expo_vals, face_hits = [], [], 0

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp_vals.append(sharpness_score(gray))
        expo_vals.append(exposure_score(gray))
        if has_face(gray):
            face_hits += 1

    cap.release()

    if not sharp_vals:
        return {"error": "no_readable_frames"}

    avg_sharp = float(np.mean(sharp_vals))
    avg_expo = float(np.mean(expo_vals))
    sharp_norm = min(1.0, avg_sharp / 300.0)
    face_ratio = face_hits / len(sample_indices)
    overall = (sharp_norm * 0.5) + (avg_expo * 0.3) + (face_ratio * 0.2)

    return {
        "type": "video",
        "width": w,
        "height": h,
        "duration_seconds": round(duration, 2),
        "sharpness_raw": round(avg_sharp, 2),
        "exposure_score": round(avg_expo, 3),
        "has_face": face_ratio > 0,
        "overall_score": round(overall, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Local folder of downloaded b-roll media")
    parser.add_argument("--out", default="scored_media.json")
    args = parser.parse_args()

    results = []
    for fname in sorted(os.listdir(args.folder)):
        ext = os.path.splitext(fname)[1].lower()
        full_path = os.path.join(args.folder, fname)

        if ext in IMAGE_EXTS:
            data = score_image(full_path)
        elif ext in VIDEO_EXTS:
            data = score_video(full_path)
        else:
            continue

        if "error" in data:
            print(f"WARN: skipping {fname} ({data['error']})", file=sys.stderr)
            continue

        data["filename"] = fname
        results.append(data)

    results.sort(key=lambda r: r["overall_score"], reverse=True)

    with open(args.out, "w") as f:
        json.dump({"media": results}, f, indent=2)

    print(f"Scored {len(results)} usable files -> {args.out}")


if __name__ == "__main__":
    main()
