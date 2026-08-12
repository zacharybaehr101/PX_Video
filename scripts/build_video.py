"""
build_video.py

Assembles the final video from:
  - manifest.json         (job definition)
  - aligned.json           (quote timestamps from transcribe_align.py)
  - scored_media.json      (ranked b-roll from score_media.py)

For each speaker/quote:
  - First `facetime_seconds` of the quote's audio -> talking-head video,
    with a lower-third (name/title) composited on top.
  - Remainder of the quote's audio -> auto-picked b-roll plays instead,
    while the speaker's voice keeps playing underneath (L-cut).

Renders one file per aspect ratio in manifest["aspect_ratios"].
9:16, 1:1, 4:5 crops are centered on detected faces (falls back to
center-crop if no face found), clamped to frame edges.

Usage:
  python build_video.py --manifest jobs/example-manifest.json \\
      --aligned aligned.json --scored scored_media.json \\
      --media-dir ./local_media --out-dir ./renders
"""

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.audio.fx.all import volumex, audio_loop

FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts")
BRAND_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "brand.json")

ASPECT_DIMENSIONS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MUSIC_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "music-library-index.json")


def pick_auto_music(mood_tags: list, media_dir: str, target_duration: float) -> str:
    """
    Picks a track from the music library index whose mood tags best overlap
    the requested tags. Prefers tracks close to (or longer than) the video's
    duration so looping is minimal. Returns a local filename or None.
    """
    if not os.path.exists(MUSIC_INDEX_PATH):
        return None
    with open(MUSIC_INDEX_PATH) as f:
        index = json.load(f)

    candidates = []
    for track in index.get("tracks", []):
        local_path = os.path.join(media_dir, track["file"])
        if not os.path.exists(local_path):
            continue  # wasn't downloaded (e.g. index out of sync with library folder)
        overlap = len(set(t.lower() for t in track.get("mood_tags", [])) & set(t.lower() for t in mood_tags))
        duration_gap = abs(track.get("duration_seconds", 0) - target_duration)
        candidates.append((overlap, -duration_gap, track["file"]))

    if not candidates:
        return None

    # Highest tag overlap first, then closest duration match
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return candidates[0][2]


def load_brand():
    with open(BRAND_PATH) as f:
        return json.load(f)


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Face-centered cropping for vertical/square reframes
# ---------------------------------------------------------------------------

def detect_face_center(frame_bgr) -> tuple:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (x + w / 2, y + h / 2)


def smart_crop_clip(clip, target_w: int, target_h: int):
    """
    Crops a clip to target_w x target_h, centered on the detected face when
    present (sampled once from the first frame - stable shots don't need
    per-frame tracking and this avoids jitter), otherwise center-crop.
    """
    src_w, src_h = clip.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Source is wider than target - crop width, keep full height
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)

    # Sample first frame for face position
    sample_frame = clip.get_frame(0)
    sample_bgr = cv2.cvtColor(sample_frame, cv2.COLOR_RGB2BGR)
    face_center = detect_face_center(sample_bgr)

    if face_center:
        cx, cy = face_center
    else:
        cx, cy = src_w / 2, src_h / 2

    x1 = int(np.clip(cx - crop_w / 2, 0, src_w - crop_w))
    y1 = int(np.clip(cy - crop_h / 2, 0, src_h - crop_h))

    cropped = clip.crop(x1=x1, y1=y1, width=crop_w, height=crop_h)
    return cropped.resize((target_w, target_h))


def fit_image_to_frame(clip, target_w: int, target_h: int):
    """For b-roll photos - same smart-crop logic, static image so cheap."""
    return smart_crop_clip(clip, target_w, target_h)


# ---------------------------------------------------------------------------
# Text overlays (lower thirds, title cards) rendered via PIL -> ImageClip
# ---------------------------------------------------------------------------

def render_lower_third(name: str, subtitle: str, frame_w: int, frame_h: int, brand: dict) -> np.ndarray:
    canvas = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    name_font = ImageFont.truetype(os.path.join(FONT_DIR, "Poppins-Bold.ttf"), size=int(frame_h * 0.045))
    sub_font = ImageFont.truetype(os.path.join(FONT_DIR, "Poppins-Regular.ttf"), size=int(frame_h * 0.03))

    bg_color = hex_to_rgb(brand["lower_third"]["bg_color"])
    accent = hex_to_rgb(brand["lower_third"]["accent_color"])
    text_color = hex_to_rgb(brand["lower_third"]["text_color"])

    bar_y = int(frame_h * 0.78)
    bar_h = int(frame_h * 0.12)
    bar_x = int(frame_w * 0.05)
    bar_w = int(frame_w * 0.55)

    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(*bg_color, 230))
    draw.rectangle([bar_x, bar_y, bar_x + 8, bar_y + bar_h], fill=(*accent, 255))

    draw.text((bar_x + 30, bar_y + 12), name, font=name_font, fill=(*text_color, 255))
    draw.text((bar_x + 30, bar_y + 12 + int(frame_h * 0.05)), subtitle, font=sub_font, fill=(*accent, 255))

    return np.array(canvas)


def render_title_card(text: str, frame_w: int, frame_h: int, brand: dict) -> np.ndarray:
    bg = hex_to_rgb(brand["title_card"]["bg_color"])
    text_color = hex_to_rgb(brand["title_card"]["text_color"])
    canvas = Image.new("RGBA", (frame_w, frame_h), (*bg, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(os.path.join(FONT_DIR, "Poppins-Bold.ttf"), size=int(frame_h * 0.08))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((frame_w - tw) / 2, (frame_h - th) / 2), text, font=font, fill=(*text_color, 255))
    return np.array(canvas)


def render_caption(text: str, frame_w: int, frame_h: int, brand: dict) -> np.ndarray:
    canvas = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(os.path.join(FONT_DIR, "Poppins-SemiBold.ttf"), size=int(frame_h * 0.04))
    text_color = hex_to_rgb(brand["caption"]["text_color"])
    outline = hex_to_rgb(brand["caption"]["outline_color"])

    max_width = int(frame_w * 0.8)
    words = text.split()
    lines, current = [], ""
    for w in words:
        test = f"{current} {w}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = w
        else:
            current = test
    if current:
        lines.append(current)

    line_h = int(frame_h * 0.055)
    total_h = line_h * len(lines)
    y = frame_h - total_h - int(frame_h * 0.12)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        x = (frame_w - lw) / 2
        # Simple outline for legibility over varied b-roll
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(*outline, 255))
        draw.text((x, y), line, font=font, fill=(*text_color, 255))
        y += line_h

    return np.array(canvas)


# ---------------------------------------------------------------------------
# Segment building
# ---------------------------------------------------------------------------

def pick_broll(scored_media: list, used: set, media_dir: str, min_duration: float):
    """Returns the next best-scored, unused b-roll file path."""
    for item in scored_media:
        if item["filename"] in used:
            continue
        if item["type"] == "video" and item.get("duration_seconds", 0) < min_duration:
            continue
        used.add(item["filename"])
        return os.path.join(media_dir, item["filename"]), item["type"]
    return None, None


def build_speaker_segment(speaker: dict, quote: dict, media_dir: str, scored_media: list,
                           used_broll: set, frame_w: int, frame_h: int, brand: dict, is_vertical: bool):
    timing = quote["timing"]
    if timing is None:
        return None  # flagged for manual review upstream, skip in render

    start, end = timing["start"], timing["end"]
    duration = end - start
    facetime = min(speaker.get("facetime_seconds", 5), duration)

    talking_head_path = os.path.join(media_dir, speaker["clip"])
    face_clip_full = VideoFileClip(talking_head_path)

    # --- Facetime portion: talking head + lower third ---
    facetime_start_mode = speaker.get("facetime_start", "start")
    if facetime_start_mode == "start":
        ft_offset = start
    elif facetime_start_mode == "mid":
        ft_offset = start + max(0, (duration - facetime) / 2)
    else:
        ft_offset = start + _timestamp_to_seconds(facetime_start_mode)

    face_segment = face_clip_full.subclip(ft_offset, min(ft_offset + facetime, face_clip_full.duration))
    face_segment = smart_crop_clip(face_segment, frame_w, frame_h) if is_vertical else face_segment.resize((frame_w, frame_h))

    lower_third_img = render_lower_third(speaker["name"], speaker.get("subtitle", ""), frame_w, frame_h, brand)
    lower_third_clip = ImageClip(lower_third_img).set_duration(face_segment.duration)
    facetime_composite = CompositeVideoClip([face_segment, lower_third_clip])

    # --- B-roll portion: covers remainder of quote, voice audio continues ---
    remaining = duration - facetime
    broll_clip = None
    if remaining > 0.3:
        broll_path, broll_type = pick_broll(scored_media, used_broll, media_dir, remaining)
        if broll_path:
            if broll_type == "video":
                broll_clip = VideoFileClip(broll_path).subclip(0, min(remaining, VideoFileClip(broll_path).duration))
            else:
                broll_clip = ImageClip(broll_path).set_duration(remaining)
            broll_clip = smart_crop_clip(broll_clip, frame_w, frame_h) if is_vertical else broll_clip.resize((frame_w, frame_h))
        else:
            # No b-roll left - fall back to holding on the talking head
            broll_clip = face_clip_full.subclip(
                ft_offset + facetime, min(ft_offset + facetime + remaining, face_clip_full.duration)
            ).resize((frame_w, frame_h))

    segments = [facetime_composite] + ([broll_clip] if broll_clip else [])
    video_segment = concatenate_videoclips(segments, method="compose")

    # Voice audio for the FULL quote duration, continuous under both parts
    voice_audio = face_clip_full.audio.subclip(start, end) if face_clip_full.audio else None

    return video_segment.set_duration(duration), voice_audio


def _timestamp_to_seconds(ts: str) -> float:
    parts = [float(p) for p in ts.split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0]


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_aspect_ratio(manifest: dict, aligned: dict, scored_media: list, media_dir: str,
                         out_dir: str, aspect: str, brand: dict):
    frame_w, frame_h = ASPECT_DIMENSIONS[aspect]
    is_vertical = frame_h >= frame_w

    used_broll = set()
    clips, voice_tracks = [], []

    # Opening title card
    for card in manifest.get("title_cards", []):
        if card.get("at") == "start":
            img = render_title_card(card["text"], frame_w, frame_h, brand)
            clips.append(ImageClip(img).set_duration(card.get("duration_seconds", 3)))

    for speaker in aligned["speakers"]:
        for quote in speaker["quotes"]:
            result = build_speaker_segment(speaker, quote, media_dir, scored_media, used_broll,
                                            frame_w, frame_h, brand, is_vertical)
            if result is None:
                continue
            video_seg, voice_seg = result
            clips.append(video_seg)
            if voice_seg:
                voice_tracks.append(voice_seg)

    final_video = concatenate_videoclips(clips, method="compose")

    # Mix continuous voice track under bed music
    voice_audio = CompositeAudioClip(voice_tracks) if voice_tracks else None

    music_cfg = manifest.get("music")
    audio_tracks = []
    if voice_audio:
        audio_tracks.append(voice_audio)
    if music_cfg:
        if music_cfg.get("mode") == "auto":
            chosen_file = pick_auto_music(music_cfg.get("mood_tags", []), media_dir, final_video.duration)
            if chosen_file is None:
                print("WARNING: no music library track matched mood_tags (or index.json is out of sync) - rendering without music bed")
            music_filename = chosen_file or ""
        else:
            music_filename = music_cfg.get("track", "")
        music_path = os.path.join(media_dir, os.path.basename(music_filename))
        if os.path.exists(music_path):
            music_clip = AudioFileClip(music_path)
            if music_clip.duration < final_video.duration:
                # Loop shorter tracks to cover the full video rather than crashing
                music_clip = audio_loop(music_clip, duration=final_video.duration)
            else:
                music_clip = music_clip.subclip(0, final_video.duration)
            db = manifest.get("music_volume_db", -18)
            music_clip = music_clip.fx(volumex, 10 ** (db / 20))
            audio_tracks.append(music_clip)
        else:
            print(f"WARNING: music track not found at {music_path}, rendering without music bed")

    if audio_tracks:
        final_video = final_video.set_audio(CompositeAudioClip(audio_tracks))

    os.makedirs(out_dir, exist_ok=True)
    aspect_label = aspect.replace(":", "x")
    out_path = os.path.join(out_dir, f"{manifest['output_prefix']}_{aspect_label}.mp4")
    final_video.write_videofile(out_path, fps=30, codec="libx264", audio_codec="aac")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--scored", required=True)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--out-dir", default="./renders")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.aligned) as f:
        aligned = json.load(f)
    with open(args.scored) as f:
        scored_media = json.load(f)["media"]

    brand = load_brand()

    outputs = []
    for aspect in manifest.get("aspect_ratios", ["16:9"]):
        if aspect not in ASPECT_DIMENSIONS:
            print(f"WARNING: unknown aspect ratio {aspect}, skipping")
            continue
        out_path = render_aspect_ratio(manifest, aligned, scored_media, args.media_dir,
                                        args.out_dir, aspect, brand)
        outputs.append(out_path)
        print(f"Rendered: {out_path}")

    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
