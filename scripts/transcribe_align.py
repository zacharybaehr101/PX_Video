"""
transcribe_align.py

Transcribes a voice track with word-level timestamps (Whisper), then finds
where each pre-written quote actually occurs in that transcript using fuzzy
matching - so captions/lower-thirds use YOUR exact wording, timed to when
it's actually spoken. Whisper's transcript is a timing aid, never displayed.

Usage:
  python transcribe_align.py --audio voice.mp3 --manifest jobs/example-manifest.json --out aligned.json
"""

import argparse
import json
import re

from rapidfuzz import fuzz

PAD_SECONDS = 0.35  # buffer added before/after a matched quote so words aren't clipped


def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def transcribe(audio_path: str, model_size: str = "small"):
    """Returns a flat list of {word, start, end} dicts for the whole track."""
    from faster_whisper import WhisperModel  # lazy import - heavy dependency, only needed here
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)

    words = []
    for seg in segments:
        for w in seg.words:
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    return words


def find_quote_span(words: list, quote_text: str, hint: str = None):
    """
    Slides a window across the transcribed words to find the best fuzzy match
    for the quote text, returns the (start, end) timestamp span.

    If `hint` (a short distinctive phrase from the quote) is given, narrows
    the search window around the best hint match first - faster and more
    reliable on long tracks with repeated phrasing.
    """
    target = normalize(quote_text)
    target_word_count = len(target.split())
    if target_word_count == 0 or not words:
        return None

    search_words = words
    offset = 0

    if hint:
        hint_norm = normalize(hint)
        hint_len = len(hint_norm.split())
        best_hint_score, best_hint_idx = 0, None
        for i in range(len(words) - hint_len + 1):
            window = " ".join(w["word"] for w in words[i:i + hint_len])
            score = fuzz.ratio(normalize(window), hint_norm)
            if score > best_hint_score:
                best_hint_score, best_hint_idx = score, i
        if best_hint_idx is not None and best_hint_score > 60:
            # Search in a generous window around the hint match
            lo = max(0, best_hint_idx - target_word_count)
            hi = min(len(words), best_hint_idx + target_word_count * 2)
            search_words = words[lo:hi]
            offset = lo

    best_score, best_start, best_end = 0, None, None
    # Slide a window roughly the size of the quote (+/- a few words for
    # filler/false-starts) across the search space.
    window_min = max(1, target_word_count - 3)
    window_max = target_word_count + 5

    for win_len in range(window_min, window_max + 1):
        for i in range(len(search_words) - win_len + 1):
            window = " ".join(w["word"] for w in search_words[i:i + win_len])
            score = fuzz.ratio(normalize(window), target)
            if score > best_score:
                best_score = score
                best_start = search_words[i]["start"]
                best_end = search_words[i + win_len - 1]["end"]

    if best_start is None or best_score < 55:
        return None  # no confident match - flag for manual review

    return {
        "start": round(max(0, best_start - PAD_SECONDS), 2),
        "end": round(best_end + PAD_SECONDS, 2),
        "match_confidence": best_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="aligned.json")
    parser.add_argument("--model-size", default="small")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    words = transcribe(args.audio, args.model_size)

    aligned_speakers = []
    for speaker in manifest.get("speakers", []):
        aligned_quotes = []
        for quote in speaker.get("quotes", []):
            span = find_quote_span(words, quote["text"], quote.get("audio_hint_near"))
            entry = {**quote, "timing": span}
            if span is None:
                entry["_needs_review"] = "No confident match found - check quote wording against actual audio."
            aligned_quotes.append(entry)
        aligned_speakers.append({**speaker, "quotes": aligned_quotes})

    with open(args.out, "w") as f:
        json.dump({"speakers": aligned_speakers}, f, indent=2)

    unmatched = sum(
        1 for s in aligned_speakers for q in s["quotes"] if q["timing"] is None
    )
    if unmatched:
        print(f"WARNING: {unmatched} quote(s) could not be confidently matched. See _needs_review flags in {args.out}.")
    print(f"Alignment complete -> {args.out}")


if __name__ == "__main__":
    main()
