# Pius X Video Pipeline

Turns a script, voice recording, music, and photos/clips into finished 16:9,
9:16, 1:1, and/or 4:5 videos with branded captions, lower thirds, and title
cards — fully automated via GitHub Actions.

## How a video gets made

1. You write a paragraph describing the video (see
   `PARAGRAPH-TO-MANIFEST-EXAMPLE.md`).
2. Claude turns it into a manifest JSON file in `/jobs/`.
3. You review it and commit it to the repo.
4. That commit triggers the GitHub Actions workflow, which:
   - downloads your voice/music/clips/photos from Bunny Storage
   - scores b-roll for quality (sharpness, exposure, faces)
   - transcribes your voice track and times your exact quotes to it
   - builds the video(s): talking head + lower third for the first 5s of
     each quote, b-roll for the rest, captions, title cards, mixed audio
   - uploads finished renders back to Bunny Storage and posts links in the
     GitHub Actions job summary

## One-time setup

### 1. Repo secrets
In GitHub: **Settings → Secrets and variables → Actions**, add:
- `BUNNY_STORAGE_API_KEY` — your Bunny Storage zone API key
- `BUNNY_STORAGE_ZONE` — your storage zone name

### 2. Bunny Storage folder structure
```
your-storage-zone/
  <project-name>/
    voice.mp3
    interview_person1.mp4
    interview_person2.mp4
    broll/
      photo1.jpg
      clip1.mp4
      ...
  music-library/
    uplifting-piano-01.mp3
    cinematic-build-02.mp3
    ...
```
Keep `assets/music-library-index.json` up to date as you add tracks — that's
what lets the auto-picker match mood tags to a manifest's request.

### 3. Fonts / branding
Already baked in: `assets/brand.json` (colors, style) and
`assets/fonts/Poppins-*.ttf` (open-license Century Gothic alternative).
Update `brand.json` if colors or style ever change.

## Running a job manually
Actions tab → **Render Video** → **Run workflow** → paste the manifest path
(e.g. `jobs/homecoming2026.json`). Useful for re-rendering without a new commit.

## Repo layout
```
/jobs/                          one manifest per video
/scripts/
  score_media.py                ranks b-roll for auto-selection
  transcribe_align.py           times your quotes to the voice audio
  build_video.py                assembles the final video(s)
  fetch_bunny_media.py          pulls source media before render
  upload_bunny_renders.py       pushes finished renders back to Bunny
/assets/
  brand.json                    colors, fonts, lower-third/caption styling
  fonts/                        Poppins TTFs (Century Gothic alternative)
  music-library-index.json      tagged catalog of your music-library tracks
/.github/workflows/render-video.yml
PARAGRAPH-TO-MANIFEST-EXAMPLE.md  how to write the paragraph Claude turns into a manifest
```

## Notes / current limitations
- Quote text must be close to verbatim to what's spoken — Whisper times it,
  it doesn't write it. If alignment confidence is low, the manifest output
  flags that quote for manual review instead of guessing.
- Face-centered cropping samples the first frame of a shot, not
  frame-by-frame tracking — fine for static/slow shots, may need a manual
  b-roll override for fast-moving action clips.
- Canva-sourced music: confirm licensing covers extracting the audio
  standalone for reuse across renders before relying on it long-term;
  YouTube Audio Library downloads are unambiguously fine for this use.
- Render time scales roughly linearly with number of aspect ratios
  requested — specify only the ratios you actually need in the manifest.
- Every render produces a matching `.srt` file alongside the `.mp4` (same
  b-roll folder in Bunny, same filename minus extension). Captions are
  burned into the video by default; set `"burn_captions": false` in the
  manifest to get a clean video + .srt only, for hand-finishing in Premiere.
- Music can fade from one volume to another as talking starts — set
  `"music_volume_before_talk_db"`, `"music_volume_db"` (after talking
  starts), and `"music_fade_seconds"` in the manifest.
