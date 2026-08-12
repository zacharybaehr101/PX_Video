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
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    AudioClip,
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
)
from moviepy.audio.fx.all import volumex, audio_loop, audio_normalize

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

# Same tightened detection parameters as score_media.py - the default
# minNeighbors=5 produces false positives on high-texture non-face imagery
# (verified against real b-roll: a bookshelf photo triggered 9 false
# "faces" at defaults, 0 at these settings, with genuine faces still caught).
FACE_MIN_NEIGHBORS = 15
FACE_MIN_SIZE_RATIO = 0.1

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
        track_duration = track.get("duration_seconds") or 0  # handles missing key AND explicit null
        duration_gap = abs(track_duration - target_duration) if track_duration else 0
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
    h, w = gray.shape
    min_dim = int(min(h, w) * FACE_MIN_SIZE_RATIO)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=FACE_MIN_NEIGHBORS, minSize=(min_dim, min_dim)
    )
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

    # Sized up from the original pass per feedback that the text read too small
    name_font = ImageFont.truetype(os.path.join(FONT_DIR, "Poppins-Bold.ttf"), size=int(frame_h * 0.08))
    sub_font = ImageFont.truetype(os.path.join(FONT_DIR, "Poppins-Regular.ttf"), size=int(frame_h * 0.052))

    bg_color = hex_to_rgb(brand["lower_third"]["bg_color"])
    accent = hex_to_rgb(brand["lower_third"]["accent_color"])
    text_color = hex_to_rgb(brand["lower_third"]["text_color"])

    bar_h = int(frame_h * 0.2)
    bar_y = int(frame_h * 0.72)
    bar_x = int(frame_w * 0.05)
    bar_w = int(frame_w * 0.62)

    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(*bg_color, 230))
    draw.rectangle([bar_x, bar_y, bar_x + 10, bar_y + bar_h], fill=(*accent, 255))

    draw.text((bar_x + 36, bar_y + int(bar_h * 0.16)), name, font=name_font, fill=(*text_color, 255))
    draw.text((bar_x + 36, bar_y + int(bar_h * 0.16) + int(frame_h * 0.09)), subtitle, font=sub_font, fill=(*accent, 255))

    return np.array(canvas)


def chunk_words(text: str, max_per_line: int = 3) -> list:
    """Splits text into short lines of 2-3 words, avoiding a lonely 1-word
    trailing line where possible - used for the transparent title overlay."""
    words = text.split()
    lines = []
    i = 0
    n = len(words)
    while i < n:
        remaining = n - i
        if remaining <= max_per_line:
            size = remaining
        elif remaining == max_per_line + 1:
            size = max_per_line - 1  # avoid stranding a single trailing word
        else:
            size = max_per_line
        lines.append(" ".join(words[i:i + size]))
        i += size
    return lines


def render_title_overlay(text: str, frame_w: int, frame_h: int, brand: dict, vertical_anchor: str = "center") -> np.ndarray:
    """
    Transparent-background title text meant to sit on top of a b-roll clip
    (not a solid full-screen card). 2-3 words per line, bold, outlined for
    legibility over any footage underneath. vertical_anchor moves the text
    block to "upper", "center", or "lower" so it's not always dead-center.
    """
    canvas = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    text_color = hex_to_rgb(brand["title_card"]["text_color"])
    outline = hex_to_rgb(brand["colors"]["black"])

    lines = chunk_words(text, max_per_line=3)
    font_size = int(frame_h * 0.09)
    font = ImageFont.truetype(os.path.join(FONT_DIR, "Poppins-Bold.ttf"), size=font_size)

    line_h = int(font_size * 1.15)
    total_h = line_h * len(lines)

    if vertical_anchor == "upper":
        y = int(frame_h * 0.12)
    elif vertical_anchor == "lower":
        y = int(frame_h * 0.62) - total_h
    else:
        y = (frame_h - total_h) / 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        x = (frame_w - lw) / 2
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(*outline, 220))
        draw.text((x, y), line, font=font, fill=(*text_color, 255))
        y += line_h

    return np.array(canvas)


def render_title_card(text: str, frame_w: int, frame_h: int, brand: dict) -> np.ndarray:
    """Kept for backward compatibility (solid full-screen card) - the default
    opening is now render_title_overlay composited on b-roll instead."""
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
    # Anchored above where the lower third sits (bar starts at ~0.72h) and
    # above typical platform UI safe zones (Reels/TikTok/Shorts controls
    # cluster in the bottom ~15-20% of the frame) - a caption glued to the
    # very bottom edge risks getting covered on-device.
    y = int(frame_h * 0.60) - total_h

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


def pick_broll_chunks(scored_media: list, used: set, media_dir: str, total_duration: float,
                       chunk_target: float = 3.0, chunk_min: float = 1.5):
    """
    Splits `total_duration` into multiple short b-roll chunks (instead of one
    clip holding for the whole remainder) so the cut has some energy/variety.
    Cycles through top-scored unused media; if the pool runs out, reuses from
    the top of the list rather than leaving a gap.
    """
    chunks = []
    remaining = total_duration
    pool_exhausted_reused = set()

    while remaining > 0.2:
        this_chunk = min(chunk_target, remaining)
        if remaining - this_chunk < chunk_min and remaining - this_chunk > 0:
            this_chunk = remaining  # avoid leaving an awkwardly short trailing sliver

        path, media_type = pick_broll(scored_media, used, media_dir, min(this_chunk, chunk_min))
        if path is None:
            # Pool exhausted - reuse best-scored items rather than fall back to nothing
            for item in scored_media:
                if item["filename"] not in pool_exhausted_reused:
                    pool_exhausted_reused.add(item["filename"])
                    path = os.path.join(media_dir, item["filename"])
                    media_type = item["type"]
                    break
        if path is None:
            break  # truly nothing available

        chunks.append((path, media_type, this_chunk))
        remaining -= this_chunk

    return chunks


def build_speaker_segment(speaker: dict, quote: dict, media_dir: str, scored_media: list,
                           used_broll: set, frame_w: int, frame_h: int, brand: dict, is_vertical: bool):
    """
    Returns (video_segments, audio_segments) - parallel lists where each
    audio segment is the EXACT audio that plays under its matching video
    segment. Keeping video and audio paired per-segment (rather than
    building one continuous voice track separately and hoping the offsets
    line up) is what guarantees lip-sync during the facetime portion.
    """
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

    ft_end = min(ft_offset + facetime, face_clip_full.duration)
    face_segment = face_clip_full.subclip(ft_offset, ft_end)
    face_segment_video = smart_crop_clip(face_segment, frame_w, frame_h) if is_vertical else face_segment.resize((frame_w, frame_h))

    lower_third_img = render_lower_third(speaker["name"], speaker.get("subtitle", ""), frame_w, frame_h, brand)
    lower_third_clip = ImageClip(lower_third_img).set_duration(face_segment_video.duration)

    caption_text = quote.get("text", "")
    caption_layers = [face_segment_video, lower_third_clip]
    if caption_text:
        caption_img = render_caption(caption_text, frame_w, frame_h, brand)
        caption_layers.append(ImageClip(caption_img).set_duration(face_segment_video.duration))

    facetime_composite = CompositeVideoClip(caption_layers, size=(frame_w, frame_h)).set_duration(face_segment_video.duration)

    # Audio for the facetime window: the ACTUAL audio under this exact video span
    facetime_audio = face_clip_full.audio.subclip(ft_offset, ft_end) if face_clip_full.audio else None

    video_segments = [facetime_composite]
    audio_segments = [facetime_audio]

    # --- B-roll portion: several short clips rather than one long hold,
    # voice audio continues underneath (L-cut). Captions keep running here
    # too, since the quote is still being spoken. ---
    remaining = duration - facetime
    if remaining > 0.3:
        broll_chunks = pick_broll_chunks(scored_media, used_broll, media_dir, remaining)
        voice_cursor = ft_offset + facetime  # continues from where facetime audio left off

        if broll_chunks:
            for path, media_type, chunk_dur in broll_chunks:
                if media_type == "video":
                    src = VideoFileClip(path)
                    clip = src.subclip(0, min(chunk_dur, src.duration))
                else:
                    clip = ImageClip(path).set_duration(chunk_dur)
                clip = smart_crop_clip(clip, frame_w, frame_h) if is_vertical else clip.resize((frame_w, frame_h))
                clip = clip.set_duration(chunk_dur)

                if caption_text:
                    caption_img = render_caption(caption_text, frame_w, frame_h, brand)
                    caption_clip = ImageClip(caption_img).set_duration(chunk_dur)
                    clip = CompositeVideoClip([clip, caption_clip], size=(frame_w, frame_h)).set_duration(chunk_dur)

                video_segments.append(clip)

                chunk_audio = face_clip_full.audio.subclip(
                    voice_cursor, min(voice_cursor + chunk_dur, face_clip_full.duration)
                ) if face_clip_full.audio else None
                audio_segments.append(chunk_audio)
                voice_cursor += chunk_dur
        else:
            # No b-roll available at all - hold on the talking head instead of a gap
            hold_end = min(voice_cursor + remaining, face_clip_full.duration)
            hold_clip = face_clip_full.subclip(voice_cursor, hold_end)
            hold_video = smart_crop_clip(hold_clip, frame_w, frame_h) if is_vertical else hold_clip.resize((frame_w, frame_h))
            hold_video = hold_video.set_duration(hold_end - voice_cursor)

            if caption_text:
                caption_img = render_caption(caption_text, frame_w, frame_h, brand)
                caption_clip = ImageClip(caption_img).set_duration(hold_video.duration)
                hold_video = CompositeVideoClip([hold_video, caption_clip], size=(frame_w, frame_h)).set_duration(hold_video.duration)

            video_segments.append(hold_video)
            audio_segments.append(hold_clip.audio)

    return video_segments, audio_segments


def build_opening_segment(manifest: dict, scored_media: list, used_broll: set, media_dir: str,
                           frame_w: int, frame_h: int, brand: dict, is_vertical: bool):
    """
    New default opening: 2-3s of b-roll with the title text overlaid
    (transparent background, 2-3 words/line) instead of a solid title card.
    The text starts in a randomized position (upper/center/lower - not
    always dead-center) and slides off (right or bottom) in the last part
    of the opening, so it doesn't just sit static the whole time.
    Position/direction can be forced via the manifest's title_cards entry
    ("position": "upper"|"center"|"lower", "slide_to": "right"|"bottom")
    for a specific look; otherwise each render picks randomly.
    Returns (video_clip, audio_clip_or_None) or None if no title_cards set.
    """
    title_cards = manifest.get("title_cards", [])
    opening = next((c for c in title_cards if c.get("at") == "start"), None)
    if opening is None:
        return None

    duration = opening.get("duration_seconds", 2.5)
    path, media_type = pick_broll(scored_media, used_broll, media_dir, duration)

    if path:
        if media_type == "video":
            src = VideoFileClip(path)
            bg_clip = src.subclip(0, min(duration, src.duration))
        else:
            bg_clip = ImageClip(path).set_duration(duration)
        bg_clip = smart_crop_clip(bg_clip, frame_w, frame_h) if is_vertical else bg_clip.resize((frame_w, frame_h))
    else:
        # No b-roll available at all - fall back to a solid brand-color card
        # rather than failing the render.
        solid = np.array(Image.new("RGBA", (frame_w, frame_h), (*hex_to_rgb(brand["title_card"]["bg_color"]), 255)))
        bg_clip = ImageClip(solid).set_duration(duration)

    vertical_anchor = opening.get("position") or random.choice(["upper", "center", "lower"])
    slide_to = opening.get("slide_to") or random.choice(["right", "bottom"])

    overlay_img = render_title_overlay(opening["text"], frame_w, frame_h, brand, vertical_anchor)
    overlay_clip = ImageClip(overlay_img).set_duration(duration)

    # Static for the first ~60% of the opening, then slides fully off-frame
    # over the remainder - gives it some motion without being distracting
    # for the brief window it's actually readable.
    hold_until = duration * 0.6
    slide_span = duration - hold_until

    def position_fn(t):
        if t <= hold_until or slide_span <= 0:
            return (0, 0)
        progress = (t - hold_until) / slide_span  # 0 -> 1 over the slide window
        if slide_to == "right":
            return (progress * frame_w, 0)
        else:  # "bottom"
            return (0, progress * frame_h)

    overlay_clip = overlay_clip.set_position(position_fn)
    composite = CompositeVideoClip([bg_clip, overlay_clip], size=(frame_w, frame_h)).set_duration(duration)

    # Opening b-roll plays with no dialogue - silence on the voice channel,
    # music (added later as a global bed) is what's heard here.
    return composite, None


def make_silence(duration: float):
    """A silent stereo audio clip of the given duration, for gaps in the
    dialogue track (e.g. under the opening b-roll, where nobody's talking)."""
    return AudioClip(lambda t: [0, 0], duration=duration, fps=44100)


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
    video_segments, audio_segments = [], []

    # New default opening: b-roll + transparent title text, not a solid card
    opening = build_opening_segment(manifest, scored_media, used_broll, media_dir,
                                     frame_w, frame_h, brand, is_vertical)
    if opening:
        opening_video, _ = opening
        video_segments.append(opening_video)
        audio_segments.append(make_silence(opening_video.duration))

    for speaker in aligned["speakers"]:
        for quote in speaker["quotes"]:
            result = build_speaker_segment(speaker, quote, media_dir, scored_media, used_broll,
                                            frame_w, frame_h, brand, is_vertical)
            if result is None:
                continue
            seg_videos, seg_audios = result
            video_segments.extend(seg_videos)
            # Every video segment needs a matching audio segment (silence if
            # the source had no audio track) so concatenation stays in sync.
            for v, a in zip(seg_videos, seg_audios):
                audio_segments.append(a if a is not None else make_silence(v.duration))

    final_video_visual = concatenate_videoclips(video_segments, method="compose")
    voice_track = concatenate_audioclips(audio_segments)

    # Real phone-recorded voice audio tends to be quiet - normalize so the
    # spoken word is clearly audible rather than needing the viewer to
    # crank their volume (this was inaudible-quiet before this fix).
    voice_track = voice_track.fx(audio_normalize)

    music_cfg = manifest.get("music")
    audio_tracks = [voice_track]
    if music_cfg:
        if music_cfg.get("mode") == "auto":
            chosen_file = pick_auto_music(music_cfg.get("mood_tags", []), media_dir, final_video_visual.duration)
            if chosen_file is None:
                print("WARNING: no music library track matched mood_tags (or index.json is out of sync) - rendering without music bed")
            music_filename = chosen_file or ""
        else:
            music_filename = music_cfg.get("track", "")
        music_path = os.path.join(media_dir, os.path.basename(music_filename))
        if os.path.exists(music_path):
            music_clip = AudioFileClip(music_path)
            if music_clip.duration < final_video_visual.duration:
                music_clip = audio_loop(music_clip, duration=final_video_visual.duration)
            else:
                music_clip = music_clip.subclip(0, final_video_visual.duration)
            # Normalize the music bed too before applying the manifest's
            # target dB, so volume is predictable regardless of the source
            # file's original loudness.
            music_clip = music_clip.fx(audio_normalize)
            db = manifest.get("music_volume_db", -18)
            music_clip = music_clip.fx(volumex, 10 ** (db / 20))
            audio_tracks.append(music_clip)
        else:
            print(f"WARNING: music track not found at {music_path}, rendering without music bed")

    final_video = final_video_visual.set_audio(CompositeAudioClip(audio_tracks))

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
