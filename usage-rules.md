# whisper_ct2 usage rules

Rules for LLM coding agents using `whisper_ct2` in a consumer project.
Published per the [`usage_rules`](https://hex.pm/packages/usage_rules)
convention; sync into your project with `mix usage_rules.sync`.

## Load the model once, reuse the struct

`WhisperCt2.load_model/2` returns `{:ok, %WhisperCt2.Model{ref: ref}}` where
`ref` is a NIF resource pointing at the live CTranslate2 model. The model
stays in memory as long as some process holds the struct. Do **not** call
`load_model/2` per request - hold it in a long-lived process.

```elixir
defmodule MyApp.Whisper do
  use GenServer

  def start_link(path), do: GenServer.start_link(__MODULE__, path, name: __MODULE__)
  def transcribe(audio, opts \\ []),
    do: GenServer.call(__MODULE__, {:transcribe, audio, opts}, :infinity)

  @impl true
  def init(path) do
    {:ok, model} = WhisperCt2.load_model(path)
    {:ok, model}
  end

  @impl true
  def handle_call({:transcribe, audio, opts}, _from, model) do
    {:reply, WhisperCt2.transcribe(model, audio, opts), model}
  end
end
```

Put it under your supervision tree. When the process dies the NIF resource
is freed, so let the supervisor reload it.

## Parallelism: one Model serialises inside ct2rs

A single `%Model{}` processes calls serially through the NIF. For real
concurrency across multiple callers, load N replicas (one per process) and
pool them - e.g. with `:poolboy`, `nimble_pool`, or a `Registry`-keyed set
of GenServers. Increasing `:max_queued_batches` only deepens the queue, not
the worker count.

Do not share the same model across OS threads expecting parallel inference;
share it across BEAM processes for fan-in, not fan-out.

## Batched transcribe collapses per-call overhead

`WhisperCt2.transcribe_batch(model, [audio1, audio2, ...], opts)` stacks
every chunk of every input into one mel batch and runs the encoder once
across the whole thing. For diarization-driven workflows (one call per
turn, dozens to hundreds of turns) this is materially faster than
looping `transcribe/3` because CTranslate2 amortises the encoder
forward pass across the batch.

`:language` applies to every audio in the batch; pass `nil` to
auto-detect per-audio (only meaningful on multilingual checkpoints).

For carving sub-windows out of an already-decoded buffer, use
`WhisperCt2.Pcm.slice(samples, sample_rate, start_s, duration_s)` -
it does the f32 byte math and bounds checking for you.

## Word-level timestamps are opt-in

Pass `word_timestamps: true` to attach `%WhisperCt2.Word{text, start, end,
probability}` entries to each segment. Implementation reuses the encoder
output from `generate` and runs one extra batched `align` call (DTW over
decoder attention) across every chunk in the batch. Cost is on the order
of the alignment pass itself, not a second encoder forward. Use it for
caption alignment or diarization-aware splicing; skip it when you only
need segment timing.

## Initial prompt and prefix

- `:initial_prompt` - free-text conditioning prepended via
  `<|startofprev|>`. Bias the decoder toward domain vocabulary, names,
  or speaker style ("Discussion of CTranslate2 internals", "Dialogue
  between Alice and Bob"). Same role as in faster-whisper.
- `:prefix` - forced text the generation must start with. Useful when
  the first words are already known (caption corrections, fixed
  intro lines).

Both are tokenised inside the NIF without special-token expansion, so
control tokens in the strings are not interpreted.

## Pass `:language` when you know it

`:language` defaults to `nil`, which makes Whisper auto-detect from the first
chunk. Auto-detection adds latency and can misfire on short or noisy clips
(English-only fine-tunes still sometimes guess `:cy` or `:fr`). Always pass
`language: "en"` (or the relevant ISO code) when the source language is known.

`model.multilingual` tells you whether the loaded checkpoint can do anything
other than English - `faster-whisper-*.en` variants are monolingual and ignore
`:language`. Branch on `model.multilingual` if your code supports both.

## Result shape

`{:ok, %WhisperCt2.Transcription{text, segments, language, duration_s}}`:

- `text` - all segment texts joined by `" "` and `String.trim/1`'d. Use this
  for display or downstream NLP.
- `segments` - list of `%WhisperCt2.Segment{}`, each carrying absolute
  `:start` / `:end` seconds, `:no_speech_prob`, `:avg_logprob`, the
  underlying text-token IDs (`:tokens`), and `:words` (`nil` unless
  `:word_timestamps` was set).
- `language` - resolved ISO code (auto-detected when not pinned).
- `duration_s` - input audio length in seconds.

Segment timestamps are real fields, not embedded tokens - do **not** regex
the text for `<|t_..|>`. Boundaries are produced by Whisper's own timestamp
tokens, parsed inside the NIF.

`:no_speech_prob` and `:avg_logprob` are always populated; filter
hallucination with e.g. `seg.avg_logprob < -1.0` or
`seg.no_speech_prob > 0.6`.

## Model struct fields are part of the API

Illustrative shape (`:ref` and `:path` omitted for brevity; both are
also part of the struct):

```elixir
%WhisperCt2.Model{
  sampling_rate: 16_000,   # always 16 kHz for published Whisper
  n_samples: 480_000,      # samples in one Whisper window (30 s)
  multilingual: true,      # false for *.en variants
  device: :cpu,            # resolved (never :auto)
  compute_type: :int8,     # resolved (never :default / :auto)
  ...
}
```

Read these at runtime instead of hardcoding. `device` and `compute_type` are
the **resolved** values - `:auto` and `:default` are normalised at load time.

## Audio contract is strict

CTranslate2 wants **mono `f32` PCM at the model's sample rate** (always
16 kHz for published Whisper checkpoints), normalised to `-1.0..1.0`.
`transcribe/3` accepts:

- a `.wav` path (16 kHz mono or stereo, 16/32-bit PCM, or 32-bit float) -
  decoded by `WhisperCt2.Wav`;
- `{:pcm_f32, binary}` - little-endian f32 samples (canonical form).

Bare-binary input is **rejected** - a typo'd path used to silently become
garbage PCM. Non-`.wav` paths are also rejected at the boundary with a
clear `:invalid_request` error.

Anything else (mp3, opus, 44.1 kHz, 8 kHz, multichannel beyond stereo)
**must be resampled upstream**. There is no built-in decoder. Use ffmpeg:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

For microphone or streaming sources, build the f32 buffer yourself and pass
`{:pcm_f32, binary}`. The authoritative sample rate is `model.sampling_rate`
on the loaded struct, not a hard-coded `16_000`.

For in-memory WAV bytes (HTTP uploads, S3 fetches, anything you do not want
to spill to disk), use `WhisperCt2.Wav.decode/1` directly instead of writing
a temp file:

```elixir
{:ok, samples} = WhisperCt2.Wav.decode(uploaded_bytes)
WhisperCt2.transcribe(model, {:pcm_f32, samples}, language: "en")
```

Audio longer than 30 s is split into Whisper-window chunks automatically;
per-chunk text is in `transcription.segments`.

## Return shape: never raises on the happy path

Every public function returns `{:ok, _} | {:error, %WhisperCt2.Error{}}`.
The error struct implements `Exception`, so `raise/1` works if you want
let-it-crash behaviour, but do not write `case` clauses that assume an
`{:ok, _}` pattern only. `Error.reason` is one of:

- `:invalid_request` - bad options or audio shape; rejected before the NIF
- `:load_error` - model directory missing or unreadable
- `:inference_error` - CTranslate2 raised during transcription
- `:runtime_error` - other ct2rs-side failure
- `:nif_panic` - Rust panic caught by the panic boundary
- `:native_error` - fallback for unrecognised native errors

## Device and compute_type selection

Probe before deciding:

```elixir
WhisperCt2.available_devices()
#=> {:ok, %{cpu: 1, cuda: 1, cuda_supported: true}}
```

- `device: :auto` (default) picks CUDA when the artefact was built with it
  and at least one device is visible; otherwise CPU. Use this unless you
  have a reason not to.
- `device: :cuda` returns
  `{:error, %WhisperCt2.Error{reason: :invalid_request}}` if CUDA is
  unavailable - do not assume it succeeds.
- `compute_type: :default` keeps the stored quantisation of the model
  (recommended for `Systran/faster-whisper-*` int8 builds).
  `compute_type: :auto` lets ct2rs pick the fastest supported on-device.

Do not hardcode `:float16` / `:int8_float16` unless you know the target
hardware supports it - mismatches raise `:load_error`.

## Model files

`load_model/2` needs a directory containing:

```
model.bin
config.json
tokenizer.json
vocabulary.txt
preprocessor_config.json
```

`Systran/faster-whisper-*` ships the first four. `preprocessor_config.json`
must be copied from any `openai/whisper-*` repo (all sizes share the file).
A missing `preprocessor_config.json` is the most common `:load_error`
cause; check this first when load fails.

## Backend selection at install time

The published Hex package picks the right precompiled NIF from your target
triple automatically. Two consumer-facing knobs:

- `WHISPER_CT2_VARIANT=mkl` on `x86_64-unknown-linux-gnu` selects the Intel
  MKL artefact instead of oneDNN. Only set this on Intel-only fleets.
- `WHISPER_CT2_BUILD=1` (or
  `config :rustler_precompiled, :force_build, whisper_ct2: true`) forces a
  source build. First build of CTranslate2 takes ~10 minutes and needs
  Rust, CMake, and a C++17 toolchain. Do not enable this in CI unless you
  understand the cost.

x86_64 macOS and Windows are not shipped - source build only.

## Do not

- Do not call `load_model/2` per transcription.
- Do not pass mp3/opus/non-16 kHz audio expecting it to work.
- Do not assume `device: :cuda` succeeds; check `available_devices/0` or
  use `:auto`.
- Do not share a single `%Model{}` to get parallel inference; pool replicas.
- Do not catch `:nif_panic` and retry blindly - it indicates a bug worth
  reporting.
- Do not hardcode `16_000` as the sample rate - read `model.sampling_rate`.
- Do not pass `:language` to a `*.en` checkpoint and expect anything but
  English; check `model.multilingual` if the language is dynamic.
- Do not regex segment text for `<|t_..|>` tokens - segment timestamps
  are real fields (`:start`, `:end`) populated from the model output.
- Do not loop `transcribe/3` over a list of short clips when
  `transcribe_batch/3` would batch them through one encoder pass.
- Do not pass control tokens like `<|en|>` inside `:initial_prompt` or
  `:prefix`; they are tokenised as plain text and will not behave as
  special tokens.
