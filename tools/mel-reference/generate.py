"""Regenerate the golden mel-spectrogram fixture used by Rust unit tests.

Run this once when the upstream feature extractor changes. The fixture lets
`build_chunks` in `preprocessor.rs` be tested for numerical parity with
`faster-whisper` without depending on Python at test time.

Outputs written to `test/fixtures/mel_golden/`:

- `input.pcm_f32`            raw f32 PCM, 16 kHz mono
- `preprocessor_config.json` config with the explicit mel filterbank
                             faster-whisper used, so the Rust loader
                             cannot drift from upstream
- `mel.f32`                  reference log-mel, row-major [feature_size, n_frames]
- `meta.json`                shape, generator version, hash for sanity

Run from the repo root:

    uv run --with faster-whisper --with numpy tools/mel-reference/generate.py
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
from faster_whisper.feature_extractor import FeatureExtractor

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "test" / "fixtures" / "mel_golden"
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fe = FeatureExtractor()
    feature_size = int(fe.mel_filters.shape[0])
    audio = synth_signal(int(fe.n_samples))

    # FeatureExtractor returns [feature_size, n_samples / hop_length + 1]
    # via librosa STFT with center=True. The transcribe pipeline trims the
    # tail frame so the encoder sees exactly `nb_max_frames`; we mirror
    # that trim so the reference matches what Rust `build_chunks`
    # produces.
    raw_mel = np.asarray(fe(audio), dtype=np.float32)
    mel = raw_mel[:, : fe.nb_max_frames]
    assert mel.shape == (feature_size, fe.nb_max_frames), mel.shape

    # Write inputs and outputs.
    (OUT_DIR / "input.pcm_f32").write_bytes(audio.tobytes(order="C"))
    (OUT_DIR / "mel.f32").write_bytes(np.ascontiguousarray(mel).tobytes(order="C"))

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
    (OUT_DIR / "preprocessor_config.json").write_text(
        json.dumps(config, separators=(",", ":")) + "\n"
    )

    meta = {
        "generator": "faster-whisper FeatureExtractor",
        "sampling_rate": SAMPLING_RATE,
        "audio_samples": int(audio.shape[0]),
        "mel_shape": list(mel.shape),
        "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "mel_sha256": hashlib.sha256(mel.tobytes()).hexdigest(),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote fixture to {OUT_DIR}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
