"""Regenerate the golden mel-spectrogram fixtures used by Rust unit tests.

Run this once when the upstream feature extractor changes. The fixtures let
`build_chunks` in `preprocessor.rs` be tested for numerical parity with
`faster-whisper` without depending on Python at test time.

Two fixture directories are written under `test/fixtures/`:

- `mel_golden`            one full 30 s window (the standard Whisper config)
- `mel_golden_multichunk` two 3 s windows: a loud first window and a
                          near-silent partial second window. Pins the
                          whole-audio normalisation max (the quiet window
                          is floored against the loud window's max, not
                          its own) and the tail zero-pad path.

Each directory contains:

- `input.pcm_f32`            raw f32 PCM, 16 kHz mono
- `preprocessor_config.json` config with the explicit mel filterbank
                             faster-whisper used, so the Rust loader
                             cannot drift from upstream
- `mel.f32`                  reference log-mel, one row-major
                             [feature_size, nb_max_frames] block per window
- `meta.json`                shape, generator version, hash for sanity

Run from the repo root:

    uv run --with faster-whisper --with numpy tools/mel-reference/generate.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from faster_whisper.feature_extractor import FeatureExtractor

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "test" / "fixtures"
SAMPLING_RATE = 16_000


def synth_signal(num_samples: int) -> np.ndarray:
    """Deterministic mix of silence + tones, padded to `num_samples`.

    The signal is 3 s of varied content (silence + 440 Hz + silence + 1 kHz),
    then zero-padded out to the full 30 s window. Padding inside the signal
    (not via FeatureExtractor's pad path) keeps Rust and Python framing on
    the same audio-length footing so any mel divergence is filterbank or
    STFT framing, not pad math.
    """
    sr = SAMPLING_RATE
    pieces = [
        np.zeros(sr // 2, dtype=np.float32),
        0.5 * np.sin(2 * math.pi * 440.0 * np.arange(sr, dtype=np.float32) / sr),
        np.zeros(sr // 2, dtype=np.float32),
        0.5 * np.sin(2 * math.pi * 1000.0 * np.arange(sr, dtype=np.float32) / sr),
    ]
    body = np.concatenate(pieces).astype(np.float32, copy=False)
    if body.shape[0] > num_samples:
        return body[:num_samples]
    padded = np.zeros(num_samples, dtype=np.float32)
    padded[: body.shape[0]] = body
    return padded


def synth_multichunk_signal(window_samples: int) -> np.ndarray:
    """One loud window followed by a quiet partial window.

    The first window is loud tones at amplitude 0.5; the tail is ~1 s of a
    700 Hz tone at amplitude 5e-3 (40 dB down) ending mid-hop, so the
    second window is mostly zero pad. With per-window normalisation the
    quiet window's clamp floor would derive from its own max and every
    floored value would diverge from this reference by ~1.0 — the fixture
    fails loudly if the whole-audio max regresses to per-chunk.
    """
    sr = SAMPLING_RATE
    loud = [
        np.zeros(sr // 4, dtype=np.float32),
        0.5 * np.sin(2 * math.pi * 440.0 * np.arange(sr, dtype=np.float32) / sr),
        0.5 * np.sin(2 * math.pi * 1000.0 * np.arange(sr, dtype=np.float32) / sr),
    ]
    body = np.concatenate(loud).astype(np.float32, copy=False)
    first = np.zeros(window_samples, dtype=np.float32)
    first[: min(body.shape[0], window_samples)] = body[:window_samples]
    # 16_037 samples: not a multiple of hop_length (160), so the tail
    # window also exercises the partial-frame rounding.
    quiet = (
        5e-3 * np.sin(2 * math.pi * 700.0 * np.arange(16_037, dtype=np.float32) / sr)
    ).astype(np.float32, copy=False)
    return np.concatenate([first, quiet])


def write_fixture(out_dir: Path, fe: FeatureExtractor, audio: np.ndarray) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_size = int(fe.mel_filters.shape[0])
    n_windows = math.ceil(audio.shape[0] / fe.n_samples)

    # Zero-extend the tail window the way Rust `build_chunks` does, then
    # run FeatureExtractor over the whole signal: one continuous STFT
    # with center=True/reflect, normalised against the global log-mel
    # max — the whole-audio semantics `build_chunks` mirrors. The
    # transcribe pipeline slices `nb_max_frames` windows from the
    # result; we mirror that slicing so the reference matches what Rust
    # produces per window.
    extended = np.zeros(n_windows * int(fe.n_samples), dtype=np.float32)
    extended[: audio.shape[0]] = audio
    raw_mel = np.asarray(fe(extended), dtype=np.float32)
    windows = [
        raw_mel[:, k * fe.nb_max_frames : (k + 1) * fe.nb_max_frames]
        for k in range(n_windows)
    ]
    for window in windows:
        assert window.shape == (feature_size, fe.nb_max_frames), window.shape

    (out_dir / "input.pcm_f32").write_bytes(audio.tobytes(order="C"))
    mel_bytes = b"".join(
        np.ascontiguousarray(w).tobytes(order="C") for w in windows
    )
    (out_dir / "mel.f32").write_bytes(mel_bytes)

    # Embed the exact mel filterbank faster-whisper used so the Rust loader
    # picks it up via `preprocessor_config.json` instead of falling back to
    # the `mel_spec` crate's filterbank.
    config = {
        "feature_size": feature_size,
        "hop_length": int(fe.hop_length),
        "n_fft": int(fe.n_fft),
        "n_samples": int(fe.n_samples),
        "nb_max_frames": int(fe.nb_max_frames),
        "sampling_rate": int(fe.sampling_rate),
        "mel_filters": fe.mel_filters.astype(np.float64).tolist(),
    }
    (out_dir / "preprocessor_config.json").write_text(
        json.dumps(config, separators=(",", ":")) + "\n"
    )

    meta = {
        "generator": "faster-whisper FeatureExtractor",
        "sampling_rate": SAMPLING_RATE,
        "audio_samples": int(audio.shape[0]),
        "n_windows": n_windows,
        "mel_shape": [feature_size, int(fe.nb_max_frames) * n_windows],
        "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "mel_sha256": hashlib.sha256(mel_bytes).hexdigest(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    fe_standard = FeatureExtractor()
    standard = write_fixture(
        FIXTURES_DIR / "mel_golden",
        fe_standard,
        synth_signal(int(fe_standard.n_samples)),
    )

    fe_short = FeatureExtractor(chunk_length=3)
    multichunk = write_fixture(
        FIXTURES_DIR / "mel_golden_multichunk",
        fe_short,
        synth_multichunk_signal(int(fe_short.n_samples)),
    )

    print(json.dumps({"mel_golden": standard, "mel_golden_multichunk": multichunk}, indent=2))


if __name__ == "__main__":
    main()
