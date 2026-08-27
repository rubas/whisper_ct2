# Changelog

## 0.6.2 - 2026-08-27

### Changed

- ct2rs 0.9.22 -> 0.10.0. The release only adds segment and word-level
  helpers to the high-level `ct2rs::Whisper` wrapper; `ct2rs::sys`, which
  the NIF drives, and the vendored CTranslate2 4.8.1 are unchanged.
- Rust lockfile dependencies refreshed to their latest compatible releases.
  `mel_spec`, `ndarray` and `tokenizers` stay on 0.3 / 0.16 / 0.22 because
  ct2rs 0.10.0 still requires those ranges.
- Development-only Hex dependency `ex_slop` moved 0.4.3 -> 0.4.4.
- The two `chunks_exact(4)` PCM decode loops now use `slice::as_chunks`,
  which clears the `clippy::chunks_exact_to_as_chunks` lint on newer Rust
  toolchains. Both call sites already reject a non-multiple-of-4 length,
  so the decoded samples are unchanged.

## 0.6.1 - 2026-06-11

### Changed

- Rustler 0.38.0 is now used for source builds, and the
  `rustler_precompiled` requirement now targets 0.9.0. This keeps the NIF
  packaging stack current; the public Elixir API is unchanged.
- Development-only Hex dependencies and compatible Rust lockfile dependencies
  were refreshed to their latest patch releases.

## 0.6.0 - 2026-06-10

Fixes every finding from the 2026-06 multi-agent Rust/STT NIF audit
(label `stt-rust-audit`): silent text loss, faster-whisper parity gaps
in the mel preprocessor and word alignment, and late or raising error
paths. Word timings now track the faster-whisper reference within one
encoder frame (20 ms).

### Changed

- ct2rs 0.9.18 → 0.9.19, bumping the vendored CTranslate2 from 4.7.1 to
  4.7.2. (#34)
- The daily security workflow audits the NIF crate's Rust dependency
  tree with cargo-audit, alongside the existing `mix deps.audit`. (#34)

### Fixed

- Word text from `:word_timestamps` is decoded through the tokenizer's
  byte-level BPE decoder. Non-ASCII words used to come back as mojibake
  ("schön" surfaced as "schÃ¶n"), and codepoints split across tokens
  glued into one giant word; both now match faster-whisper, including
  per-codepoint word splitting for spaceless languages (zh, ja, th, lo,
  my, yue). (#19)
- The last word of every 30 s chunk ends at the alignment's EOT
  boundary instead of a fabricated 20 ms duration. (#23)
- Fallback segment ends (unclosed timestamp pair, or
  `with_timestamps: false`) are bounded by the chunk's real audio
  length; a 3 s clip no longer reports a segment ending at 30 s. (#24)
- PCM containing NaN or infinity is rejected as `:invalid_request`
  instead of silently transcribing the corrupted region as silence.
  Amplitudes that overflow the mel power are rejected the same way.
  (#20)
- The 2 GiB mel-buffer cap is enforced from input sizes before the PCM
  copy and the mel chunks are allocated, not after. (#21)
- `WhisperCt2.available_devices/0` runs on a dirty scheduler. On CUDA
  builds its first call initialises the NVIDIA driver, which used to
  stall a normal BEAM scheduler for the whole driver init. (#22)
- `WhisperCt2.load_model/2` fails at load when the tokenizer lacks
  `<|startofprev|>`, instead of degrading at inference time once
  `:initial_prompt` is used. (#31)
- The word-timestamp alignment prompt no longer carries an explicit
  `<|notimestamps|>` — CTranslate2 appends it internally, so the decoder
  used to see the token doubled, perturbing the cross-attention word
  timings derive from relative to faster-whisper. (#25)
- Text generated without an opening timestamp — a `:prefix` echo, a
  fine-tune opening with text, or text between lone timestamps — is kept
  as its own segment instead of silently discarded. (#26)
- Log-mel normalisation floors against the whole audio's maximum, as
  faster-whisper does, instead of per 30 s window; a near-silent window
  of a longer audio is no longer normalised against its own max. (#27)
- Reflect padding for audio shorter than 200 samples reads the
  zero-padded region like the reference instead of duplicating the last
  sample into the entire leading pad. (#28)
- `WhisperCt2.load_model/2` validates `preprocessor_config.json`: a zero
  numeric field or a mis-shaped `mel_filters` matrix fails as
  `:load_error` naming the offending field, instead of an opaque
  `:nif_panic` at the first transcribe. (#29)
- Integer options that overflow the NIF's fixed-width types (`u32` /
  `i32`) are rejected as `:invalid_request` instead of raising
  `ArgumentError` at the NIF boundary. (#30)

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
