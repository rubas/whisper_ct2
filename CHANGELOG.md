# Changelog

## 0.3.0

### Breaking

- Removed the bundled WAV decoder. `transcribe/3` and
  `transcribe_batch/3` now accept only `{:pcm_f32, binary}`; decoding,
  downmixing, and resampling are the caller's job. The `WhisperCt2.Wav`
  module is gone. Use `ffmpeg -ar 16000 -ac 1 -f f32le` (or any other
  audio stack) to produce the f32 PCM buffer upstream.

## 0.2.0

Major refactor onto `ct2rs::sys::Whisper` directly. The NIF now owns the
mel spectrogram, tokenizer, and prompt construction, which unlocks
structured segment data, prompt biasing, batched transcribe across
multiple audios, and word-level timestamps.

### Breaking

- `%WhisperCt2.Transcription{}` now exposes `:segments` as a list of
  `%WhisperCt2.Segment{}` (text, start, end, no_speech_prob, avg_logprob,
  tokens, words) instead of plain strings, plus new `:language` and
  `:duration_s` fields.
- `:timestamp`, `:return_scores`, `:return_logits_vocab`, and
  `:return_no_speech_prob` options were removed - the corresponding
  fields are always populated.
- Raw bare-binary audio is no longer accepted by `transcribe/3`. Use
  `{:pcm_f32, binary}`; non-`.wav` paths now return a clear
  `:invalid_request` error instead of silently feeding garbage PCM.
- `WhisperCt2.available_devices/0` now returns `{:ok, info} | {:error, _}`
  instead of crashing on the unhappy path.

### Added

- `WhisperCt2.transcribe_batch/3` batches every chunk of every input
  through one encoder forward pass - large speedup for
  diarization-driven workflows with many short turns.
- `:initial_prompt` and `:prefix` options bias decoding toward domain
  vocabulary or a forced opening (same role as in faster-whisper).
- `:word_timestamps` runs one batched DTW alignment pass and attaches
  `%WhisperCt2.Word{}` entries with per-word timing to each segment.
- `%WhisperCt2.Segment{}` and `%WhisperCt2.Word{}` modules.
- `WhisperCt2.Pcm.slice/4` helper for cheaply carving sub-windows out of
  an already-decoded f32 buffer (diarization, VAD-driven splices).

### Internal

- New Rust modules: `preprocessor` (own mel filterbank), `tokens`
  (special-token IDs, prompt construction, timestamp parsing), `align`
  (batched DTW + BPE word grouping), `transcribe` (single + batched
  flow).
- English-only checkpoints (`*.en`) now use the correct
  `[<|startoftranscript|>]` prompt instead of the multilingual
  `[sot, lang, transcribe]` shape.

## 0.1.0

- Initial release.
- `WhisperCt2.load_model/2` loads a CTranslate2-converted Whisper directory.
- `WhisperCt2.transcribe/3` runs inference on `.wav` paths or raw f32 PCM.
- Built-in WAV decoder for 16 kHz mono / stereo, 16-bit and 32-bit PCM and
  32-bit float.
- Rustler NIF over [`ct2rs`](https://crates.io/crates/ct2rs); no Python
  runtime required.
- Precompiled NIF artefacts via `rustler_precompiled` for
  `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu` (oneDNN, optional
  `mkl` variant), and `aarch64-unknown-linux-gnu`. CUDA loaded lazily via
  `cuda-dynamic` on all Linux artefacts.
