# whisper_ct2

`whisper_ct2` is an Elixir library for running OpenAI Whisper speech-to-text
models inside the BEAM. It loads CTranslate2-converted Whisper models through a
Rustler NIF, so Elixir code can transcribe WAV files or f32 PCM buffers without
starting Python or a separate inference service.


## Installation

```elixir
def deps do
  [{:whisper_ct2, "~> 0.1.0"}]
end
```

Installation downloads a precompiled NIF artefact matching your target
triple from the project's GitHub releases. No Rust toolchain or CMake is
needed on the consumer side.

### Source builds

Set `WHISPER_CT2_BUILD=1` in your environment (or
`config :rustler_precompiled, :force_build, whisper_ct2: true` in your
parent project) to compile from source instead. The first source build of
CTranslate2 takes ~10 minutes and requires:

- Rust toolchain (`rustup`, stable)
- `cmake`, a C++17 compiler, `make`
- Linux: `libstdc++`, `libgomp` available at link time
- CUDA toolkit 12+ if building with `cuda` or `cuda-dynamic` features

## Models

Point `WhisperCt2.load_model/2` at a directory containing a CTranslate2-converted
Whisper model. Required files:

```text
model.bin
config.json
tokenizer.json
vocabulary.txt
preprocessor_config.json
```

The [`Systran/faster-whisper-*`](https://huggingface.co/Systran) repositories
ship the first four directly. They do **not** include
`preprocessor_config.json`; copy the canonical one from `openai/whisper-tiny.en`
(or any other openai/whisper-* repo — they all share the same file):

```bash
uvx hf download Systran/faster-whisper-tiny.en \
  --local-dir models/faster-whisper-tiny.en

uvx hf download openai/whisper-tiny.en preprocessor_config.json \
  --local-dir models/faster-whisper-tiny.en
```

## Backends

The published Hex package ships four precompiled artefacts; install picks
the right one automatically based on your target triple:

| Target triple                  | CPU backend  | CUDA            | Notes                                              |
| ------------------------------ | ------------ | --------------- | -------------------------------------------------- |
| `aarch64-apple-darwin`         | Accelerate   | —               | Apple Silicon (M1+). Uses Accelerate / AMX paths.  |
| `x86_64-unknown-linux-gnu`     | oneDNN       | `cuda-dynamic`  | Default x86_64 binary; runs well on Intel & AMD.   |
| `x86_64-unknown-linux-gnu` (`mkl`) | Intel MKL | `cuda-dynamic` | Intel-tuned variant. Opt in via env var (below).   |
| `aarch64-unknown-linux-gnu`    | oneDNN       | `cuda-dynamic`  | Graviton/Grace, optional CUDA on GH200-class hosts.|

`cuda-dynamic` defers loading `libcudart` until first GPU use, so each
artefact still runs on hosts without CUDA installed — `:device` selection
picks CUDA when available, else CPU.

x86_64 macOS and Windows are not shipped.

### Selecting the MKL variant

For Intel-only fleets where you want maximum SGEMM throughput:

```bash
WHISPER_CT2_VARIANT=mkl mix deps.compile whisper_ct2
```

(`rustler_precompiled` reads this env var at install time and selects the
`--mkl` artefact instead of the default.)

### Build from source with a custom backend

For source builds you can pick any combination of `ct2rs` features:

```bash
WHISPER_CT2_BUILD=1 WHISPER_CT2_FEATURES="dnnl cuda-dynamic" mix compile
# other options: mkl, openblas, accelerate, cuda, cuda-dynamic
```

### Runtime device selection

```elixir
WhisperCt2.available_devices()
#=> %{cpu: 1, cuda: 1, cuda_supported: true}

{:ok, model} =
  WhisperCt2.load_model("models/faster-whisper-tiny.en",
    device: :auto,            # :cpu | :cuda | :auto (default)
    compute_type: :auto,      # :default | :auto | :float16 | :int8_float16 | ...
    device_indices: [0]
  )
```

`:auto` picks CUDA when the artefact supports it and at least one CUDA
device is visible; otherwise CPU. Explicit `:cuda` returns
`{:error, %WhisperCt2.Error{reason: :invalid_request}}` if either condition
fails.

## Usage

```elixir
{:ok, model} = WhisperCt2.load_model("models/faster-whisper-tiny.en")
{:ok, %WhisperCt2.Transcription{text: text}} =
  WhisperCt2.transcribe(model, "jfk.wav", language: "en")

IO.puts(text)
# => "And so my fellow Americans ask not what your country can do for you ..."
```

### Audio contract

CTranslate2 expects **mono `f32` PCM samples** at the model's sample rate
(16 kHz for every published Whisper checkpoint), normalized to the
`-1.0..1.0` range. `transcribe/3` accepts three forms that all bridge to
that contract:

- a `.wav` path (16 kHz mono, 16-bit / 32-bit PCM, or 32-bit float) —
  decoded by the built-in `WhisperCt2.Wav` module;
- `{:pcm_f32, binary}` with little-endian f32 samples — the canonical form;
- a raw binary already in f32 form (treated as `{:pcm_f32, binary}`).

If you build samples yourself (microphone, resampler, etc.), hand them in
via `{:pcm_f32, binary}`. The authoritative sample rate is
`model.sampling_rate` on the loaded `%WhisperCt2.Model{}`.

Audio longer than 30 s is split into Whisper-window chunks automatically;
per-chunk text is exposed via `Transcription.segments`.

## Options

`transcribe/3` accepts any subset of:

| Option                         | Type                | Notes                                                  |
| ------------------------------ | ------------------- | ------------------------------------------------------ |
| `:language`                    | `String.t \| nil`   | ISO code (`"en"`). `nil` auto-detects.                 |
| `:timestamp`                   | `boolean`           | Include `<\|t_..\|>` tokens in the output text.        |
| `:beam_size`                   | `pos_integer`       | Beam-search width.                                    |
| `:patience`                    | `float`             | Beam-search patience.                                 |
| `:length_penalty`              | `float`             | Decoding length penalty.                              |
| `:repetition_penalty`          | `float`             | Decoding repetition penalty.                          |
| `:no_repeat_ngram_size`        | `non_neg_integer`   | Disallow repeated n-grams of this size.                |
| `:sampling_temperature`        | `float`             | Sampling temperature.                                 |
| `:sampling_topk`               | `pos_integer`       | Top-k sampling.                                       |
| `:suppress_blank`              | `boolean`           | Suppress the initial blank token.                     |
| `:suppress_tokens`             | `[integer]`         | Suppress these token IDs.                              |
| `:max_length`                  | `pos_integer`       | Max tokens per chunk.                                 |
| `:num_hypotheses`              | `pos_integer`       | Number of decoded hypotheses.                          |
| `:return_scores`               | `boolean`           | Forwarded to CTranslate2; not exposed in the result.   |
| `:return_logits_vocab`         | `boolean`           | Forwarded to CTranslate2; not exposed in the result.   |
| `:return_no_speech_prob`       | `boolean`           | Forwarded to CTranslate2; not exposed in the result.   |
| `:max_initial_timestamp_index` | `non_neg_integer`   | Cap the first timestamp token.                         |

Unset values use the CTranslate2 defaults.

Unknown option keys and out-of-range values return
`{:error, %WhisperCt2.Error{reason: :invalid_request}}` before reaching the
NIF.

## Errors

All failures return `{:error, %WhisperCt2.Error{}}`. `reason` is one of
`:invalid_request`, `:load_error`, `:inference_error`, `:runtime_error`,
`:nif_panic`, or `:native_error`. The struct also implements `Exception`, so
`raise/1` works.

## Testing

Unit tests run with no external dependencies:

```bash
mix test
```

The end-to-end transcription test downloads the `faster-whisper-tiny.en`
model (~75 MB) and the `jfk.wav` clip from the whisper.cpp samples:

```bash
mix test --include integration
```

Cached under `test/fixtures/`. Set `WHISPER_CT2_REFRESH=1` to redownload.

## License

Apache-2.0. CTranslate2 itself is MIT-licensed. The bundled `ct2rs` crate
links CTranslate2 statically by default.
