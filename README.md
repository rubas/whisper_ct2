# whisper_ct2

`whisper_ct2` is an Elixir library that runs OpenAI Whisper speech-to-text
models inside the BEAM. A Rustler NIF loads Whisper models in the CTranslate2
format, so Elixir code can transcribe f32 PCM audio without Python or a
separate inference service.

CTranslate2 is the C++ inference engine behind
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper). It supports
int8 and int8-float16 quantization, and CUDA, oneDNN, MKL, and Accelerate
backends.

## Installation

```elixir
def deps do
  [{:whisper_ct2, "~> 0.7.0"}]
end
```

The install downloads the precompiled NIF for your target triple from the
GitHub release of this version. You need no Rust toolchain and no CMake.

To build from source instead, set `WHISPER_CT2_BUILD=1`, or set
`config :rustler_precompiled, :force_build, whisper_ct2: true` in your
project. The first source build of CTranslate2 takes about 10 minutes and
needs:

- Rust 1.98 or later (`rustup`)
- `cmake`, `make`, and a C++17 compiler
- On Linux: `libstdc++` and `libgomp` at link time
- CUDA toolkit 12 or later for the `cuda` or `cuda-dynamic` features

## Models

`WhisperCt2.load_model/2` takes a directory with a Whisper model in the
CTranslate2 format. The directory must contain these files:

```text
model.bin
config.json
tokenizer.json
vocabulary.txt
preprocessor_config.json
```

The [`Systran/faster-whisper-*`](https://huggingface.co/Systran) repositories
have the first four files, but not `preprocessor_config.json`. Copy that file
from any `openai/whisper-*` repository; all Whisper sizes use the same one:

```bash
uvx hf download Systran/faster-whisper-tiny.en \
  --local-dir models/faster-whisper-tiny.en

uvx hf download openai/whisper-tiny.en preprocessor_config.json \
  --local-dir models/faster-whisper-tiny.en
```

## Usage

```elixir
{:ok, model} = WhisperCt2.load_model("models/faster-whisper-tiny.en")

# Decode and resample to 16 kHz mono f32 PCM before this call (ffmpeg,
# Membrane, or any tool that writes little-endian f32 bytes).
pcm = File.read!("jfk.pcm")

{:ok, %WhisperCt2.Transcription{text: text, segments: segs}} =
  WhisperCt2.transcribe(model, {:pcm_f32, pcm}, language: "en")

IO.puts(text)
# => "And so, my fellow Americans ask not what your country can do for you ..."

for s <- segs do
  IO.puts("[#{s.start}-#{s.end}] (no_speech=#{Float.round(s.no_speech_prob, 3)}) #{s.text}")
end
```

A `%WhisperCt2.Segment{}` has the absolute `:start` and `:end` in seconds,
`:no_speech_prob`, `:avg_logprob`, and the text token IDs. With
`:word_timestamps`, it also has a list of `%WhisperCt2.Word{}` with the timing
of each word.

### Audio

`transcribe/3` and `transcribe_batch/3` accept only `{:pcm_f32, binary}`:
little-endian, mono `f32` samples in the range `-1.0..1.0`, at the sample rate
of the model. Every published Whisper checkpoint uses 16 kHz.

The library returns an `:invalid_request` error for every other input, such as
a path, WAV or MP3 bytes, or 44.1 kHz audio. It also rejects NaN and infinite
samples, which usually come from a bug in the upstream decoder. The library
has no audio decoder. To convert a file once:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f f32le output.pcm
```

The library splits audio longer than 30 s into Whisper windows. The encoder
runs once for all chunks.

### Batches and word timestamps

```elixir
# Diarization: decode the call once, then transcribe each speaker turn.
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

`transcribe_batch/3` puts every chunk of every input into one encoder forward
pass. `:word_timestamps` adds one batched DTW alignment pass and adds
`%Word{}` entries to each segment.

### Prompt and prefix

```elixir
WhisperCt2.transcribe(model, {:pcm_f32, talk_pcm},
  language: "en",
  initial_prompt: "Discussion of CTranslate2, BEAM, and Whisper internals.",
  prefix: "Welcome back to the show."
)
```

`:initial_prompt` adds text before the audio (through `<|startofprev|>`), so
the decoder prefers the words and style of that text. `:prefix` sets the start
of the transcript. Like faster-whisper, the library keeps the last
`min(max_length, 448) / 2 - 1` tokens of the prompt and the first
`min(max_length, 448) / 2 - 1` tokens of the prefix. At the default
`:max_length` of 448, that is 223 tokens each. 448 is the position count of
the decoder, so a long prompt cannot use up the output tokens.

## Options

`transcribe/3` and `transcribe_batch/3` accept any subset of:

| Option                         | Type                | Notes                                                  |
| ------------------------------ | ------------------- | ------------------------------------------------------ |
| `:language`                    | `String.t \| nil`   | ISO code (`"en"`). `nil` auto-detects on multilingual. |
| `:initial_prompt`              | `String.t \| nil`   | Free-text context prepended via `<\|startofprev\|>`.   |
| `:prefix`                      | `String.t \| nil`   | Forced text the generation must start with.            |
| `:word_timestamps`             | `boolean`           | Attach per-word timing via a batched DTW alignment.    |
| `:with_timestamps`             | `boolean`           | Emit `<\|t_..\|>` segment timestamps (default `true`). `false` for fine-tunes that emit plain text. |
| `:beam_size`                   | `pos_integer`       | Beam-search width.                                     |
| `:patience`                    | `float`             | Beam-search patience.                                  |
| `:length_penalty`              | `float`             | Decoding length penalty. A value that puts the decoder score out of `f32` range returns `:inference_error`. |
| `:repetition_penalty`          | `float`             | Decoding repetition penalty.                           |
| `:no_repeat_ngram_size`        | `non_neg_integer`   | Disallow repeated n-grams of this size.                |
| `:sampling_temperature`        | `float`             | Sampling temperature.                                  |
| `:sampling_topk`               | `pos_integer`       | Top-k sampling.                                        |
| `:suppress_blank`              | `boolean`           | Suppress the initial blank token.                      |
| `:suppress_tokens`             | `[integer]`         | Suppress these token IDs.                              |
| `:max_length`                  | `pos_integer`       | Max tokens per chunk.                                  |
| `:num_hypotheses`              | `pos_integer`       | Number of decoded hypotheses.                          |
| `:max_initial_timestamp_index` | `non_neg_integer`   | Cap the first timestamp token.                         |

Unset options use the CTranslate2 defaults. Every segment has
`no_speech_prob` and `avg_logprob`; no option is needed for them.

An unknown option key or a value out of range returns
`{:error, %WhisperCt2.Error{reason: :invalid_request}}` before the call
reaches the NIF.

## Errors

Every failure returns `{:error, %WhisperCt2.Error{}}`. `reason` is one of
`:invalid_request`, `:load_error`, `:inference_error`, `:runtime_error`,
`:nif_panic`, or `:native_error`. The struct is also an exception, so
`raise/1` works.

## Backends and devices

Each release has four precompiled NIFs. The install picks the one for your
target triple:

| Target triple                      | CPU backend | GPU            | Notes                                                 |
| ---------------------------------- | ----------- | -------------- | ----------------------------------------------------- |
| `aarch64-apple-darwin`             | Accelerate  | none           | Apple Silicon (M1 and later).                         |
| `x86_64-unknown-linux-gnu`         | oneDNN      | `cuda-dynamic` | The default on x86_64, for Intel and AMD.             |
| `x86_64-unknown-linux-gnu` (`mkl`) | Intel MKL   | `cuda-dynamic` | Tuned for Intel. Opt in with `WHISPER_CT2_VARIANT`.   |
| `aarch64-unknown-linux-gnu`        | oneDNN      | `cuda-dynamic` | Graviton and Grace; CUDA on GH200-class hosts.        |

There is no build for x86_64 macOS or Windows.

`cuda-dynamic` loads `libcudart` only on the first GPU use, so each NIF also
runs on a host without CUDA.

To get the MKL build on an Intel-only fleet, set the variable when you
compile the dependency:

```bash
WHISPER_CT2_VARIANT=mkl mix deps.compile whisper_ct2
```

A source build can use any combination of the `ct2rs` features `dnnl`, `mkl`,
`openblas`, `accelerate`, `cuda`, and `cuda-dynamic`:

```bash
WHISPER_CT2_BUILD=1 WHISPER_CT2_FEATURES="dnnl cuda-dynamic" mix compile
```

Choose the device when you load the model:

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

`:auto` uses CUDA when the NIF supports it and the host has at least one CUDA
device. Otherwise it uses the CPU. `:cuda` returns
`{:error, %WhisperCt2.Error{reason: :invalid_request}}` when one of these
conditions is false.

## Testing

The unit tests need no network:

```bash
mix test
```

The end-to-end test downloads the `faster-whisper-tiny.en` model (about 75 MB)
and the `jfk.wav` clip from the whisper.cpp samples into `test/fixtures/`:

```bash
mix test --include integration
```

Set `WHISPER_CT2_REFRESH=1` to download them again.

## License

MIT. CTranslate2 is also MIT. The `ct2rs` crate links CTranslate2 statically
by default.
