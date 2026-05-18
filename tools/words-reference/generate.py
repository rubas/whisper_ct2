"""Regenerate the word-timestamp golden fixture used by integration tests.

Runs `faster_whisper.WhisperModel` end-to-end on the canonical JFK clip
and dumps every emitted word with its start/end timestamp and
probability. The integration test loads this fixture and asserts that
our transcribe output stays within a fixed tolerance of the reference,
which is the only way to catch alignment drift (e.g. a wrong
`start_sequence` in `align.rs`) that doesn't surface in a plain
"does the transcript contain the right words" check.

Run from the repo root:

    uv run --with faster-whisper tools/words-reference/generate.py

Outputs written to `test/fixtures/words_golden/`:

- `audio_path.txt`  Identifier of the audio used (informational).
- `words.json`      List of `{text, start, end, probability}` objects.
- `meta.json`       Model name, faster-whisper version, generator
                    options, sha256 of the audio file.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import faster_whisper
from faster_whisper import WhisperModel

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_PATH = REPO_ROOT / "test" / "fixtures" / "jfk.wav"
OUT_DIR = REPO_ROOT / "test" / "fixtures" / "words_golden"
MODEL_NAME = "Systran/faster-whisper-tiny.en"
JFK_URL = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"


def ensure_audio() -> Path:
    if AUDIO_PATH.exists() and AUDIO_PATH.stat().st_size > 0:
        return AUDIO_PATH
    AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {JFK_URL} -> {AUDIO_PATH}", file=sys.stderr)
    urllib.request.urlretrieve(JFK_URL, AUDIO_PATH)
    return AUDIO_PATH


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = ensure_audio()

    # `compute_type="default"` mirrors what the Elixir integration test
    # loads. Using the same checkpoint and inference settings is what
    # makes the comparison meaningful; widening either side would let
    # real drift slip through.
    model = WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="default",
    )
    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True,
        beam_size=5,
    )

    words = []
    for segment in segments:
        for word in segment.words or []:
            words.append(
                {
                    "text": word.word,
                    "start": float(word.start),
                    "end": float(word.end),
                    "probability": float(word.probability),
                }
            )

    if not words:
        raise SystemExit("faster-whisper produced no words; fixture would be useless")

    (OUT_DIR / "words.json").write_text(json.dumps(words, indent=2) + "\n")
    (OUT_DIR / "audio_path.txt").write_text(
        str(audio_path.relative_to(REPO_ROOT)) + "\n"
    )

    meta = {
        "model": MODEL_NAME,
        "faster_whisper_version": faster_whisper.__version__,
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "language": info.language,
        "duration": info.duration,
        "word_count": len(words),
        "beam_size": 5,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {len(words)} words to {OUT_DIR}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
