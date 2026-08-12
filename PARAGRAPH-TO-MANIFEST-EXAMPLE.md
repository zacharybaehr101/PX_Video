# How This Works

You write one paragraph with the details. I turn it into the manifest JSON
the pipeline actually reads. You never have to touch the JSON yourself —
just review it before we commit and run it.

---

## Example: What You'd Give Me

> Video is for Homecoming 2026, call it "Homecoming 2026 Senior Interviews."
> Voice track is voice.mp3 and music is music.mp3, both in the homecoming2026
> Bunny folder — keep the music pretty quiet under the voice. B-roll photos
> and clips are in homecoming2026/broll/, just pick the best ones.
> Two interviews: first is Zachary Baehr, he's a Senior, clip is
> interview_zbaehr.mp4, use this quote — "This program changed how I see
> storytelling and gave me the confidence to actually put my work out
> there." Second is Mr. Baehr, Digital Media Teacher, clip is
> interview_mrbaehr.mp4, quote is "Every year I'm amazed at what these
> students create when you just give them the tools and get out of the
> way." Open the video with a title card that just says "Homecoming 2026"
> for about 3 seconds. Need both the square-ish landscape version and the
> vertical one for Instagram.

---

## What I'd Generate From That

```json
{
  "title": "Homecoming 2026 Senior Interviews",
  "output_prefix": "homecoming2026_senior_interviews",
  "brand": "piusx",

  "voice_audio_url": "https://your-zone.b-cdn.net/homecoming2026/voice.mp3",
  "music_audio_url": "https://your-zone.b-cdn.net/homecoming2026/music.mp3",
  "music_volume_db": -18,

  "media_folder": "homecoming2026/broll/",

  "aspect_ratios": ["16:9", "9:16"],

  "speakers": [
    {
      "clip": "interview_zbaehr.mp4",
      "name": "Zachary Baehr",
      "subtitle": "Senior",
      "facetime_seconds": 5,
      "facetime_start": "start",
      "quotes": [
        {
          "text": "This program changed how I see storytelling and gave me the confidence to actually put my work out there.",
          "broll": "auto"
        }
      ]
    },
    {
      "clip": "interview_mrbaehr.mp4",
      "name": "Mr. Baehr",
      "subtitle": "Digital Media Teacher",
      "facetime_seconds": 5,
      "facetime_start": "start",
      "quotes": [
        {
          "text": "Every year I'm amazed at what these students create when you just give them the tools and get out of the way.",
          "broll": "auto"
        }
      ]
    }
  ],

  "title_cards": [
    { "text": "Homecoming 2026", "at": "start", "duration_seconds": 3 }
  ]
}
```

---

## What Your Paragraph Needs to Cover

So I can build a clean manifest on the first try, try to include:

| Info | Example |
|---|---|
| Video title | "Homecoming 2026 Senior Interviews" |
| Where voice + music audio live | Bunny folder/filenames |
| Where b-roll lives | Bunny folder path |
| Each speaker: name, grade/title, clip filename | "Zachary Baehr, Senior, interview_zbaehr.mp4" |
| Each speaker's **exact quote** (close to verbatim) | in quotes, word-for-word-ish |
| Any title cards / text overlays and roughly when | "open with a title card that says..." |
| Aspect ratios needed | "both" (default) or just one |
| Anything unusual | e.g. force a specific b-roll shot for one quote, different music volume, non-default facetime timing |

If you skip something, I'll either use the sensible default (both aspect
ratios, auto b-roll, 5s facetime at start of quote) or just ask.
