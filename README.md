# whisper_ct2

`whisper_ct2` is an Elixir library for running OpenAI Whisper speech-to-text
models inside the BEAM. It loads CTranslate2-converted Whisper models through a
Rustler NIF, so Elixir code can transcribe f32 PCM buffers without starting
Python or a separate inference service.

CTranslate2 is the speed-optimised C++ inference engine that powers
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) — 4-8x faster
than vanilla `openai-whisper` on the same hardware, with int8 / int8-float16
quantisation and CUDA / oneDNN / MKL / Accelerate backends. 

## Installation

```elixir
def deps do
  [{:whisper_ct2, "~> 0.2.0"}]
end
```

Installation downloads a precompiled NIF artefact matching your target triple
from the project's GitHub releases. No Rust toolchain or CMake is needed on
the consumer side.

### Source builds

Set `WHISPER_CT2_BUILD=1` in your environment (or
`config :rustler_precompiled, :force_build, whisper_ct2: true` in your parent
project) to compile from source instead. The first source build of CTranslate2
takes ~10 minutes and requires:

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
(or any other `openai/whisper-*` repo - all Whisper sizes share the same file):

```bash
uvx hf download Systran/faster-whisper-tiny.en \
  --local-dir models/faster-whisper-tiny.en

uvx hf download openai/whisper-tiny.en preprocessor_config.json \
  --local-dir models/faster-whisper-tiny.en
```

## Backends

The published Hex package ships four precompiled artefacts; install picks the
right one automatically based on your target triple:

| Target triple                       | CPU backend | GPU            | Notes                                                |
| ----------------------------------- | ----------- | -------------- | ---------------------------------------------------- |
| `aarch64-apple-darwin`              | Accelerate  | none           | Apple Silicon (M1+). Uses Accelerate / AMX paths.    |
| `x86_64-unknown-linux-gnu`          | oneDNN      | `cuda-dynamic` | Default x86_64 binary; runs well on Intel and AMD.   |
| `x86_64-unknown-linux-gnu` (`mkl`)  | Intel MKL   | `cuda-dynamic` | Intel-tuned variant. Opt in via env var (below).     |
| `aarch64-unknown-linux-gnu`         | oneDNN      | `cuda-dynamic` | Graviton/Grace, optional CUDA on GH200-class hosts.  |

> **AMD ROCm/HIP** is supported as a source build only in 0.4.0
> (`WHISPER_CT2_BUILD=1 WHISPER_CT2_FEATURES="hip dnnl"`). A prebuilt
> `--rocm` artefact is planned for a follow-up release.

`cuda-dynamic` defers loading `libcudart` until first GPU use, so each artefact
still runs on hosts without CUDA installed. `:device` selection picks CUDA when
available, otherwise CPU.

x86_64 macOS and Windows are not shipped.

### Selecting the MKL variant

For Intel-only fleets where you want maximum SGEMM throughput:

```bash
WHISPER_CT2_VARIANT=mkl mix deps.compile whisper_ct2
```

`rustler_precompiled` reads this env var at install time and selects the `--mkl`
artefact instead of the default.

### Source-building for AMD GPUs (ROCm/HIP)

For hosts with AMD GPUs and the ROCm 6.2+ SDK installed at `/opt/rocm`
(override with `ROCM_PATH`):

```bash
WHISPER_CT2_BUILD=1 WHISPER_CT2_FEATURES="hip dnnl" mix compile
```

The build links the shared `libctranslate2.so` produced by CTranslate2's
HIP CMake branch and embeds an rpath to it; runtime requires
`libamdhip64.so` and `libhipblas.so` to be present (no dynamic-loading
fallback to CPU). `device: :cuda` and `device: :auto` resolve to the AMD
GPU — CTranslate2 reuses `Device::CUDA` internally for HIP; the Elixir
`available_devices/0` reports `:hip_supported: true` so callers can
distinguish.

Override the GFX target list with `CMAKE_HIP_ARCHITECTURES="gfx1100"`
(semicolon-separated). Common targets: gfx906 (Vega20/MI50), gfx908
(MI100), gfx90a (MI200/MI210/MI250), gfx942 (MI300), gfx1030 (RDNA2 /
RX 6000), gfx1100 (RDNA3 / RX 7900).

A prebuilt `--rocm` artefact is planned for a follow-up release.

### Build from source with a custom backend

For source builds you can pick any combination of `ct2rs` features:

```bash
WHISPER_CT2_BUILD=1 WHISPER_CT2_FEATURES="dnnl cuda-dynamic" mix compile
# other options: mkl, openblas, accelerate, cuda, cuda-dynamic, hip
```

`hip` is mutually exclusive with `cuda` / `cuda-dynamic` (CTranslate2's
HIP CMake arm refuses to build with WITH_CUDA=ON). Source builds with
`hip` need the ROCm SDK at `ROCM_PATH` (default `/opt/rocm`).

### Runtime device selection

```elixir
WhisperCt2.available_devices()
#=> {:ok, %{cpu: 1, cuda: 1, cuda_supported: true}}

{:ok, model} =
  WhisperCt2.load_model("models/faster-whisper-tiny.en",
    device: :auto,            # :cpu | :cuda | :auto (default)
    compute_type: :auto,      # :default | :auto | :float16 | :int8_float16 | ...
    device_indices: [0]
  )
```

`:auto` picks CUDA when the artefact supports it and at least one CUDA device
is visible; otherwise CPU. Explicit `:cuda` returns
`{:error, %WhisperCt2.Error{reason: :invalid_request}}` if either condition
fails.

## Usage

```elixir
{:ok, model} = WhisperCt2.load_model("models/faster-whisper-tiny.en")

# Decode/resample to 16 kHz mono f32 PCM upstream (ffmpeg, Membrane,
# anything that produces little-endian f32 bytes).
pcm = File.read!("jfk.pcm")

{:ok, %WhisperCt2.Transcription{text: text, segments: segs}} =
  WhisperCt2.transcribe(model, {:pcm_f32, pcm}, language: "en")

IO.puts(text)
# => "And so, my fellow Americans ask not what your country can do for you ..."

for s <- segs do
  IO.puts("[#{s.start}-#{s.end}] (no_speech=#{Float.round(s.no_speech_prob, 3)}) #{s.text}")
end
```

`%WhisperCt2.Segment{}` carries absolute `:start` / `:end` seconds,
`:no_speech_prob`, `:avg_logprob`, the underlying text token IDs, and
(when `:word_timestamps` is on) a list of `%WhisperCt2.Word{}` with
per-word timing.

### Audio contract

CTranslate2 expects **mono `f32` PCM samples** at the model's sample rate
(16 kHz for every published Whisper checkpoint), normalized to the
`-1.0..1.0` range. `transcribe/3` and `transcribe_batch/3` accept exactly
one shape:

- `{:pcm_f32, binary}` - little-endian f32 samples at the model's
  sample rate.

Anything else (paths, raw bare binaries, WAV bytes, MP3, 44.1 kHz, ...)
is rejected at the boundary with an `:invalid_request` error. There is
no bundled audio decoder; decode, downmix, and resample upstream using
your tool of choice. For a one-shot file conversion:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f f32le output.pcm
```

Audio longer than 30 s is chunked into Whisper windows automatically; the
encoder runs once across every chunk in the batch.

### Batched transcribe and word timestamps

```elixir
# Diarization-driven workflow: one master decode upstream, many short
# splices fed in as PCM byte ranges.
samples = File.read!("call.pcm")
turns =
  [
    WhisperCt2.Pcm.slice(samples, 16_000, 0.0, 3.2),
    WhisperCt2.Pcm.slice(samples, 16_000, 3.2, 4.5)
    # ...
  ]
  |> Enum.map(fn {:ok, bin} -> {:pcm_f32, bin} end)

{:ok, transcriptions} =
  WhisperCt2.transcribe_batch(model, turns, language: "en", word_timestamps: true)
```

`transcribe_batch/3` stacks every chunk of every input into one encoder
forward pass. `:word_timestamps` adds one batched DTW alignment pass and
attaches `%Word{}` entries to each segment.

### Decoding biases

```elixir
WhisperCt2.transcribe(model, {:pcm_f32, talk_pcm},
  language: "en",
  initial_prompt: "Discussion of CTranslate2, BEAM, and Whisper internals.",
  prefix: "Welcome back to the show."
)
```

`:initial_prompt` prepends free-text context (via `<|startofprev|>`) so the
decoder is biased toward your domain vocabulary or speaker style;
`:prefix` forces the start of the generated transcript.

## Options

`transcribe/3` and `transcribe_batch/3` accept any subset of:

| Option                         | Type                | Notes                                                  |
| ------------------------------ | ------------------- | ------------------------------------------------------ |
| `:language`                    | `String.t \| nil`   | ISO code (`"en"`). `nil` auto-detects on multilingual. |
| `:initial_prompt`              | `String.t \| nil`   | Free-text context prepended via `<\|startofprev\|>`.   |
| `:prefix`                      | `String.t \| nil`   | Forced text the generation must start with.            |
| `:word_timestamps`             | `boolean`           | Attach per-word timing via a batched DTW alignment.    |
| `:beam_size`                   | `pos_integer`       | Beam-search width.                                     |
| `:patience`                    | `float`             | Beam-search patience.                                  |
| `:length_penalty`              | `float`             | Decoding length penalty.                               |
| `:repetition_penalty`          | `float`             | Decoding repetition penalty.                           |
| `:no_repeat_ngram_size`        | `non_neg_integer`   | Disallow repeated n-grams of this size.                |
| `:sampling_temperature`        | `float`             | Sampling temperature.                                  |
| `:sampling_topk`               | `pos_integer`       | Top-k sampling.                                        |
| `:suppress_blank`              | `boolean`           | Suppress the initial blank token.                      |
| `:suppress_tokens`             | `[integer]`         | Suppress these token IDs.                              |
| `:max_length`                  | `pos_integer`       | Max tokens per chunk.                                  |
| `:num_hypotheses`              | `pos_integer`       | Number of decoded hypotheses.                          |
| `:max_initial_timestamp_index` | `non_neg_integer`   | Cap the first timestamp token.                         |

Unset values use the CTranslate2 defaults. `no_speech_prob` and
`avg_logprob` are always populated on each segment - there is no opt-in
return-knob.

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

The end-to-end transcription test downloads the `faster-whisper-tiny.en` model
(~75 MB) and the `jfk.wav` clip from the whisper.cpp samples:

```bash
mix test --include integration
```

Cached under `test/fixtures/`. Set `WHISPER_CT2_REFRESH=1` to redownload.

## License

MIT. CTranslate2 itself is MIT-licensed. The bundled `ct2rs` crate links
CTranslate2 statically by default.
