# whisper_ct2

Elixir bindings for Whisper speech-to-text through CTranslate2: a Rustler NIF
over the `ct2rs` crate, with no Python at runtime. `README.md` describes the
public API. This file holds the rules for work in the repo.

## Builds and tests

- `config/config.exs` always builds the NIF from source in this repo. The
  first compile builds CTranslate2 and takes about 10 minutes. Later compiles
  reuse the Cargo target directory.
- `task test:integration` downloads the tiny model and the JFK clip (about
  75 MB) and runs a real transcription. CI runs it weekly and on manual
  dispatch, never on a pull request. Run it locally after you change the NIF.
- `tools/*/generate.py` write the golden fixtures from `faster-whisper`. Run
  them only when the reference implementation changes.
- `test/fixtures/` is gitignored and downloaded on demand, except the
  `*_golden/` directories. Git tracks those, so the parity tests run without
  network access.

## Release

[docs/release.md](docs/release.md) is the runbook. Git tracks
`checksum-Elixir.WhisperCt2.Native.exs`, so the checksum of each tagged
release stays in the repo.

## Design decisions

- The NIF calls `ct2rs::sys::Whisper` directly and owns the mel filterbank,
  the prompt, and the word alignment. The high-level `ct2rs::Whisper` wrapper
  gives no structured segment data, no `initial_prompt` or `prefix`, and no
  batch of several audios.
- `transcribe_batch/3` puts every chunk of every audio into one storage view,
  so the encoder runs once for the whole batch.
- English-only checkpoints (`*.en`) get only `<|startoftranscript|>` as the
  start of the prompt. Multilingual checkpoints also get the language token
  and `<|transcribe|>`. `PromptParts::build` in `tokens.rs` holds this branch.
  Keep it.
- The tokenizer vocabulary has no timestamp tokens. Their base ID is
  `no_timestamps_id + 1`, as in faster-whisper. See
  `SpecialTokens::resolve` in `tokens.rs`.
