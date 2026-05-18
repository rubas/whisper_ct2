# Changelog

## Unreleased

### Added

- Precompiled NIF distribution via `rustler_precompiled`. Consumers
  install Hex artefacts matching their target triple instead of building
  CTranslate2 from source on first compile. Four artefacts ship: Apple
  Silicon (Accelerate), x86_64-linux (oneDNN + cuda-dynamic), x86_64-linux
  `mkl` variant (Intel MKL + cuda-dynamic), and aarch64-linux
  (oneDNN + cuda-dynamic).
- New cargo features in `whisper_ct2_native`: `mkl`, `dnnl`, `openblas`,
  `accelerate`, plus the existing `cuda` and `cuda-dynamic`. Selectable at
  source-build time via `WHISPER_CT2_FEATURES`.
- New `WHISPER_CT2_VARIANT=mkl` install-time switch to pick the Intel MKL
  artefact on x86_64-linux.
- New `WHISPER_CT2_BUILD=1` install-time switch (and
  `config :rustler_precompiled, :force_build, whisper_ct2: true`) to force
  a from-source build instead of downloading a precompiled artefact.
- `.github/workflows/release.yml` — matrix build of all four artefacts on
  tag push, uploads tarballs and `checksum-Elixir.WhisperCt2.Native.exs`
  to the GitHub release.
- `.github/workflows/ci.yml` and `integration.yml` — PR-time fast checks
  and weekly end-to-end transcription against the real tiny model.
- Strict Elixir-side option validation for both `load_model/2` and
  `transcribe/3`. Unknown keys and out-of-range values now return
  `{:error, %WhisperCt2.Error{reason: :invalid_request}}` before any NIF call.
- New `load_model/2` options: `:max_queued_batches`, `:cpu_core_offset`
  (forwarded to `ct2rs::Config`).
- New `transcribe/3` options: `:num_hypotheses`, `:return_scores`,
  `:return_logits_vocab`, `:return_no_speech_prob`,
  `:max_initial_timestamp_index`, `:suppress_tokens`.
- README "Audio contract" section documenting the underlying
  CTranslate2 contract (mono f32 PCM at the model's sample rate, normalized
  to `-1.0..1.0`).
- Cargo.lock is now part of the published Hex package for reproducible
  native builds.

### Changed

- Native error payloads are now plain `NifMap` structs with atom keys
  (`:type`, `:message`, `:details`) instead of JSON-encoded serde values.
- `nif_available_devices` and `nif_model_info` are now wrapped in
  `run_with_panic_protection`, so every NIF entry point catches Rust panics
  and returns `:nif_panic` instead of risking the BEAM.

### Removed

- Dropped unused Rust dependencies: `hound`, `serde`, `serde_json`, and the
  `serde` feature on `rustler`. WAV decoding remains in pure Elixir
  (`WhisperCt2.Wav`).

## 0.1.0

- Initial release.
- `WhisperCt2.load_model/1` loads a CTranslate2-converted Whisper directory.
- `WhisperCt2.transcribe/3` runs inference on `.wav` paths or raw f32 PCM.
- Built-in WAV decoder for 16 kHz mono / stereo, 16-bit and 32-bit PCM and
  32-bit float.
- Rustler NIF over [`ct2rs`](https://crates.io/crates/ct2rs); no Python
  runtime required.
