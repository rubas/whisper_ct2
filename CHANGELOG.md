# Changelog

## 0.7.0 - 2026-09-23

Fixes the audit findings #55 to #65, #67 and #74, and moves the
precompiled NIFs to CTranslate2 4.8.2. Two changes break callers, so this
is a minor release: `Transcription.text` joins segments without a
separator, and `Segment.avg_logprob` no longer scales with
`:length_penalty`.

### Changed

- **Breaking:** `Transcription.text` no longer inserts a space between
  segments. The NIF keeps the tokenizer's spacing, so CJK transcripts get
  no extra spaces, and a word or punctuation mark split at a segment
  boundary joins: the segments `" inter"` and `"national"` now give
  `international`, not `inter national`. Callers that split `text` on the
  old separator must use `:segments` instead. (#57)
- **Breaking:** `Segment.avg_logprob` uses the faster-whisper formula: the
  mean log probability per generated token, end-of-text included. It no
  longer depends on `:length_penalty`. At the default penalty, values move
  slightly toward zero (-0.600 becomes -0.594). Check any `avg_logprob`
  threshold that you tuned with a non-default `:length_penalty`. (#56)
- `:initial_prompt` keeps only its last `min(max_length, 448) / 2 - 1`
  tokens and `:prefix` only its first `min(max_length, 448) / 2 - 1` tokens
  (223 at the default `:max_length`), like faster-whisper. A long prompt no
  longer shortens or empties the transcript or returns `:inference_error`.
  (#55)
- ct2rs 0.10.0 -> 0.10.1. The precompiled NIFs now vendor CTranslate2
  4.8.2, which adds StorageView bounds and allocation size checks, and
  oneDNN 3.13.2.
- Source builds require Rust 1.98 or later (was 1.91). The crate declares
  `rust-version = "1.98"`. Precompiled installs do not change.
- The precompiled NIFs build with `codegen-units = 16` (was 1) and keep
  thin LTO. (#51)
- Mel preprocessing is about 9 to 14 times faster. The output is
  bit-identical. (#67)
- Transcription frees the mel chunks before the encoder runs, which lowers
  peak memory for long audio and large batches. (#58)
- CI uses Elixir 1.20.4, OTP 29.1.1, Rust 1.98.1 and zizmor 1.30.1.
  `mix.exs` keeps `elixir: "~> 1.17"` as the minimum version.
- The release workflow decides from the `v<version>` tag whether a version
  is released, and a manual dispatch builds from the tag it gets. See
  `docs/release.md`.

### Fixed

- `rustls` in `Cargo.lock` moves to 0.23.45 to fix RUSTSEC-2026-0285. Only
  the optional `mkl` and `openblas` source builds use it. (#53)
- `load_model/2` returns `:load_error` for a `preprocessor_config.json`
  with ragged `mel_filters` rows, even when the row lengths add up to the
  expected total. The error names the first bad row. (#64)
- `load_model/2` returns `:load_error` when `nb_max_frames` is not
  `n_samples / hop_length`, or when the pad buffer of one chunk or the mel
  filterbank build goes past the 2 GiB feature buffer cap. Before, such a
  config loaded, and the first transcribe then aborted the VM or silently
  dropped audio. (#65)
- `Segment.start` and `Segment.end` of a closed timestamp pair are never
  past the real audio length. (#59)
- Word timestamps use the true median word duration for chunks with an
  even word count, as faster-whisper does. (#63)
- `transcribe/3` and `transcribe_batch/3` return `:inference_error` instead
  of raising `ArgumentError` when an extreme `:length_penalty` makes the
  decoder score infinite or NaN. (#74)
- Options that are not a keyword list, improper lists in
  `:suppress_tokens`, `:device_indices`, or the audios of
  `transcribe_batch/3`, float options outside the `f32` range, and strings
  that are not valid UTF-8 return `:invalid_request` instead of raising.
  (#60)
- `transcribe_batch/3` validates the options for an empty audio list.
  Invalid options return `:invalid_request` instead of `{:ok, []}`. (#61)
- `WhisperCt2.Pcm.slice/4` returns `:invalid_request` for a start or
  duration so large that the sample count overflows a float, instead of
  raising `ArithmeticError`. (#62)

## 0.6.2 - 2026-08-28

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
