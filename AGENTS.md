# whisper_ct2 Agent Instructions

`whisper_ct2` is a native Elixir Whisper STT library backed by CTranslate2
via a Rustler NIF over the `ct2rs` crate. There is no Python in the loop.

## Development

- The Rust crate lives in `native/whisper_ct2_native/`. Bundled CTranslate2
  is compiled from source on first build (CMake + C++17 required).
- Keep the NIF surface narrow: load model, query info, transcribe one chunk.
  All chunking, validation, and audio decoding lives in Elixir.
- The `WhisperCt2.Wav` decoder handles only the formats Whisper accepts
  directly (16 kHz mono / stereo, 16/32-bit PCM, 32-bit float). For other
  formats, resample upstream with `ffmpeg -ar 16000 -ac 1`.

## Quality Gates

- `mix format --check-formatted --dry-run`
- `mix test` (unit, fast, no network)
- `mix test --include integration` (downloads model + WAV, runs real
  inference; ~75 MB first run)
- `cargo test --manifest-path native/whisper_ct2_native/Cargo.toml`
- `mix credo --strict` when Credo is available

## Errors

All public functions return `{:error, %WhisperCt2.Error{}}` on failure.
Never raise from happy-path code. The NIF protects against Rust panics and
maps them to `:nif_panic`.
