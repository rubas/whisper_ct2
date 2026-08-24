# whisper_ct2

Elixir bindings for Whisper speech-to-text through CTranslate2, driven by a
Rustler NIF over the `ct2rs` crate. No Python at runtime.

`README.md` owns the user-facing intro, options, audio contract, and error
taxonomy. This file covers repo-internal work only.

## Gates

`task check` is the default gate: format check, compile, credo, Elixir tests,
Rust tests, and zizmor over the workflows. CI runs the same target on pushes
to `main` and on pull requests, then builds the Hex tarball.

- The first compile builds CTranslate2 from source and takes about ten
  minutes. Later compiles reuse the Cargo target dir.
- `task test:integration` downloads the tiny model and the JFK clip (about
  75 MB) and runs a real transcription. GitHub runs it on a weekly cron and
  on manual dispatch, never on a pull request. Run it locally after you
  change the NIF.
- `task fix` applies formatting and clippy fixes in place.

## Layout

- `native/whisper_ct2_native/src/` holds the NIF: `transcribe.rs` the
  transcription flow, `preprocessor.rs` the mel filterbank, `tokens.rs` the
  special-token resolution, `align.rs` the word alignment.
- `tools/*/generate.py` regenerate the golden fixtures from `faster-whisper`.
  Run them only when the reference implementation changes.
- `test/fixtures/` is downloaded on demand and gitignored, except the
  `*_golden/` directories, which are checked in so the parity tests run
  without network access.
- `checksum-Elixir.WhisperCt2.Native.exs` is tracked so the checksum matching
  each tagged release is reproducible from the repo.
- `docs/release.md` holds the artefact matrix and the publish procedure.

## House decisions

- The NIF drives `ct2rs::sys::Whisper` directly and owns the mel filterbank,
  prompt construction, and word alignment. The high-level `ct2rs::Whisper`
  wrapper exposes no structured per-segment data, no `initial_prompt` or
  `prefix`, and no batched multi-audio transcribe.
- `transcribe_batch/3` stacks every chunk of every audio into one storage
  view, so the encoder runs once across the whole batch.
- English-only checkpoints (`*.en`) get the `[<|startoftranscript|>]` prompt
  alone; multilingual checkpoints get `[sot, lang, transcribe]`. The branch
  lives in `transcribe.rs::transcribe_many`. Keep it split.
- Whisper timestamp tokens are absent from the tokenizer vocab. Their base ID
  is `no_timestamps_id + 1`, which matches faster-whisper. See
