# Changelog

## 0.5.0 - 2026-05-20

Initial public release. Native Elixir Whisper speech-to-text backed by
CTranslate2 through a Rustler NIF over `ct2rs::sys::Whisper`. No Python.

### Features

- `WhisperCt2.load_model/2` loads a CTranslate2-converted Whisper model
  directory and returns a `%WhisperCt2.Model{}` with resolved `:device`
  and `:compute_type`.
- `WhisperCt2.transcribe/3` accepts `{:pcm_f32, binary}` (mono, 16 kHz,
  little-endian f32) and returns a `%WhisperCt2.Transcription{}` whose
  `:segments` carry absolute start/end times, `:no_speech_prob`,
  `:avg_logprob`, the underlying token IDs, and optional per-word timing.
- `WhisperCt2.transcribe_batch/3` stacks every chunk of every input into
  one encoder forward pass - a large speedup for diarization-driven
  workflows with many short turns.
- `:initial_prompt` and `:prefix` bias decoding; `:word_timestamps` adds a
  batched DTW alignment pass attaching `%WhisperCt2.Word{}` entries;
  `:with_timestamps` toggles `<|t_..|>` segment timestamps for plain-text
  fine-tunes.
- English-only checkpoints (`*.en`) use the `[<|startoftranscript|>]`
  prompt; multilingual checkpoints use `[sot, lang, transcribe]`.
- `WhisperCt2.Pcm.slice/4` carves sub-windows out of an already-decoded
  f32 buffer with loud bounds checking.
- `WhisperCt2.available_devices/0` reports CPU/CUDA device counts and the
  build's CUDA-support flag.
- Structured `%WhisperCt2.Error{}` taxonomy: `:invalid_request`,
  `:load_error`, `:inference_error`, `:runtime_error`, `:nif_panic`,
  `:native_error`.

### Backends

- Precompiled NIF artefacts via `rustler_precompiled` for
  `aarch64-apple-darwin` (Accelerate), `x86_64-unknown-linux-gnu`
  (oneDNN, optional `mkl` variant), and `aarch64-unknown-linux-gnu`
  (oneDNN). CUDA is loaded lazily via `cuda-dynamic` on every Linux
  artefact, so one binary runs on CPU-only and CUDA hosts alike.
- Opt into a source build with `WHISPER_CT2_BUILD=1`, or pick the MKL
  artefact on x86_64 Linux with `WHISPER_CT2_VARIANT=mkl`.
