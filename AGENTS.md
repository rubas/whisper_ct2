# whisper_ct2 Agent Instructions

`whisper_ct2` is a native Elixir Whisper STT library backed by CTranslate2
via a Rustler NIF over the `ct2rs` crate. No Python is involved.

For the user-facing intro, options, audio contract, and error taxonomy,
read `README.md`. For consumer rules synced via `mix usage_rules.sync`,
read `usage-rules.md`. This file covers only repo-internal guidance.

## Workflow

Standard interface is `Taskfile.yml`:

- `task setup` - `mix deps.get`.
- `task compile` - `mix compile --warnings-as-errors`. First source build
  of CTranslate2 takes ~10 minutes.
- `task fmt` / `task fmt:check` - Elixir + Rust formatting.
- `task lint` - `mix credo --strict` and `cargo clippy -D warnings`.
- `task test` - fast Elixir unit tests.
- `task test:integration` - real model, real inference, ~75 MB download.
- `task test:rust` - Rust unit tests.
- `task check` - full local gate (fmt, compile, lint, tests, docs).

## Architectural invariants

These are the load-bearing decisions that aren't visible from the public
API surface and should not be silently undone:

- Drive `ct2rs::sys::Whisper` directly; the NIF owns the mel filterbank,
  prompt construction, and word alignment. The `ct2rs::Whisper`
  high-level wrapper does not expose structured per-segment data,
  `initial_prompt` / `prefix`, or batched multi-audio transcribe.
- `transcribe_batch/3` stacks all chunks of all audios into one storage
  view; the encoder runs once across the whole batch.
- English-only checkpoints (`*.en`) get the `[<|startoftranscript|>]`
  prompt only; multilingual checkpoints get `[sot, lang, transcribe]`.
  The branch lives in `transcribe.rs::transcribe_many`; do not collapse
  it.
- Whisper timestamp tokens are not in the tokenizer vocab; their base
  ID is `no_timestamps_id + 1` (matches faster-whisper). See
  `tokens::SpecialTokens::resolve`.

## Release

Precompiled artefacts are built on tag push by
`.github/workflows/release.yml` for four targets:

- `aarch64-apple-darwin` (Accelerate)
- `x86_64-unknown-linux-gnu` (oneDNN + `cuda-dynamic`)
- `x86_64-unknown-linux-gnu` `mkl` variant (Intel MKL + `cuda-dynamic`)
- `aarch64-unknown-linux-gnu` (oneDNN + `cuda-dynamic`)

Consumers fetch the artefact matching their triple through
`rustler_precompiled`. Opt into a source build with `WHISPER_CT2_BUILD=1`
or pick an MKL artefact on x86_64 Linux with `WHISPER_CT2_VARIANT=mkl`.

## Hex publish flow

1. Bump `@version` in `mix.exs` and add a `CHANGELOG.md` entry. Push to
   `main`.
2. The `release.yml` workflow detects the version bump, builds NIF
   tarballs for every target/variant, creates the tag, and uploads the
   tarballs plus `SHA256SUMS` to the GitHub release.
3. Locally, regenerate the checksum file from the published assets,
   then commit and push it:

   ```bash
   mix rustler_precompiled.download WhisperCt2.Native --all --ignore-unavailable --print
   jj describe -m "chore: update NIF checksum for v$(...)" && jj git push -b main
   ```

   `checksum-Elixir.WhisperCt2.Native.exs` is tracked in git so the
   checksum that matches each tagged release is reproducible from the
   repo.
4. Run `mix hex.publish` from a clean tree. The checksum is included in
   the Hex tarball via `files: ~w(... checksum-*.exs ...)` in `mix.exs`.
