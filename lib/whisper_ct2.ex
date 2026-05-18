defmodule WhisperCt2 do
  @moduledoc """
  Native Elixir bindings for Whisper speech-to-text via CTranslate2.

  This library calls CTranslate2 directly through a Rustler NIF wrapping the
  [`ct2rs`](https://crates.io/crates/ct2rs) crate. There is no Python runtime
  involved.

  ## Quickstart

      {:ok, model} = WhisperCt2.load_model("/path/to/faster-whisper-tiny")
      {:ok, %WhisperCt2.Transcription{text: text}} =
        WhisperCt2.transcribe(model, "audio.wav", language: "en")
      IO.puts(text)

  ## Model files

  Point `load_model/2` at a directory containing these CT2 Whisper files:

  - `model.bin`
  - `config.json`
  - `tokenizer.json`
  - `vocabulary.txt`
  - `preprocessor_config.json` (mel filter constants — copy from
    `openai/whisper-tiny.en` if your CT2 conversion is missing it; all
    Whisper sizes share the same file)

  The [`Systran/faster-whisper-*`](https://huggingface.co/Systran) repos
  ship the first four directly.

  ## Audio contract

  CTranslate2 expects **mono `f32` PCM samples** at the model's
  `:sampling_rate` (16 kHz for every published Whisper checkpoint), normalized
  to the `-1.0..1.0` range. `transcribe/3` accepts:

  - a path to a `.wav` file (16 kHz mono, 16-bit or 32-bit PCM, or 32-bit float)
    — decoded by the built-in `WhisperCt2.Wav` module;
  - a `{:pcm_f32, binary}` tuple containing little-endian `f32` samples — the
    canonical form CTranslate2 consumes;
  - a bare `binary` already in little-endian `f32` form (treated as
    `{:pcm_f32, binary}`).

  If you build samples yourself (microphone, resampler, etc.), hand them in
  via `{:pcm_f32, binary}`. The authoritative sample rate is
  `model.sampling_rate` on a loaded model.

  Audio longer than the Whisper 30 s window is split internally by
  `ct2rs::Whisper`; per-chunk text is returned in `Transcription.segments`.
  """

  alias WhisperCt2.{Error, Model, Native, Transcription, Wav}

  @typedoc "Audio sources accepted by `transcribe/3`."
  @type audio :: Path.t() | binary() | {:pcm_f32, binary()}

  @typedoc "Options accepted by `transcribe/3`."
  @type transcribe_opt ::
          {:language, String.t() | nil}
          | {:timestamp, boolean()}
          | {:beam_size, pos_integer()}
          | {:patience, float()}
          | {:length_penalty, float()}
          | {:repetition_penalty, float()}
          | {:no_repeat_ngram_size, non_neg_integer()}
          | {:sampling_temperature, float()}
          | {:sampling_topk, pos_integer()}
          | {:suppress_blank, boolean()}
          | {:max_length, pos_integer()}
          | {:num_hypotheses, pos_integer()}
          | {:return_scores, boolean()}
          | {:return_logits_vocab, boolean()}
          | {:return_no_speech_prob, boolean()}
          | {:max_initial_timestamp_index, non_neg_integer()}
          | {:suppress_tokens, [integer()]}

  @typedoc "Options accepted by `load_model/2`."
  @type load_opt ::
          {:device, :cpu | :cuda | :auto}
          | {:compute_type, Model.compute_type()}
          | {:device_indices, [non_neg_integer()]}
          | {:num_threads_per_replica, non_neg_integer()}
          | {:max_queued_batches, integer()}
          | {:cpu_core_offset, integer()}

  @load_options [
    :device,
    :compute_type,
    :device_indices,
    :num_threads_per_replica,
    :max_queued_batches,
    :cpu_core_offset
  ]

  @transcribe_options [
    :language,
    :timestamp,
    :beam_size,
    :patience,
    :length_penalty,
    :repetition_penalty,
    :no_repeat_ngram_size,
    :sampling_temperature,
    :sampling_topk,
    :suppress_blank,
    :max_length,
    :num_hypotheses,
    :return_scores,
    :return_logits_vocab,
    :return_no_speech_prob,
    :max_initial_timestamp_index,
    :suppress_tokens
  ]

  @devices [:cpu, :cuda, :auto]
  @compute_types [
    :default,
    :auto,
    :float32,
    :float16,
    :bfloat16,
    :int8,
    :int8_float32,
    :int8_float16,
    :int8_bfloat16,
    :int16
  ]

  @doc """
  Reports CTranslate2 device support for this build.

  Returns a map with `:cpu` and `:cuda` device counts, plus `:cuda_supported`
  indicating whether the NIF was built with CUDA enabled. Build with CUDA
  support via:

      WHISPER_CT2_FEATURES=cuda-dynamic mix compile

  The `cuda-dynamic` feature defers loading `libcudart` until the GPU is
  actually used, so the same artefact works on CPU-only machines.
  """
  @spec available_devices() :: %{
          cpu: non_neg_integer(),
          cuda: non_neg_integer(),
          cuda_supported: boolean()
        }
  def available_devices do
    {:ok, info} = Native.available_devices()
    info
  end

  @doc """
  Loads a CTranslate2 Whisper model from a directory.

  ## Options

  - `:device` - `:cpu`, `:cuda`, or `:auto` (default). `:auto` picks CUDA when
    the binary was built with CUDA support and at least one CUDA device is
    visible; otherwise CPU.
  - `:compute_type` - precision used at inference. `:default` keeps the
    model's stored quantisation; `:auto` picks the fastest supported on this
    device. Other choices: `:float32`, `:float16`, `:bfloat16`, `:int8`,
    `:int8_float32`, `:int8_float16`, `:int8_bfloat16`, `:int16`.
  - `:device_indices` - non-empty list of GPU indices (default `[0]`).
  - `:num_threads_per_replica` - intra-op threads. `0` (default) lets
    CTranslate2 pick.
  - `:max_queued_batches` - CTranslate2 batcher queue depth.
  - `:cpu_core_offset` - first CPU core to bind worker threads to.
  """
  @spec load_model(Path.t(), [load_opt()]) :: {:ok, Model.t()} | {:error, Error.t()}
  def load_model(path, opts \\ [])

  def load_model(path, opts) when is_binary(path) and is_list(opts) do
    with :ok <- validate_non_empty_string(path, :path),
         :ok <- validate_known_options(opts, @load_options),
         :ok <- validate_load_options(opts) do
      do_load_model(path, opts)
    end
  end

  def load_model(_path, _opts) do
    {:error, Error.new(:invalid_request, "path must be a string and opts a keyword list")}
  end

  defp do_load_model(path, opts) do
    case Native.load_model(path, build_load_opts(opts)) do
      {:ok, ref} ->
        {:ok, info} = Native.model_info(ref)

        {:ok,
         %Model{
           ref: ref,
           path: path,
           sampling_rate: info.sampling_rate,
           n_samples: info.n_samples,
           multilingual: info.multilingual,
           device: String.to_atom(info.device),
           compute_type: String.to_atom(info.compute_type)
         }}

      {:error, payload} ->
        {:error, Error.from_native(payload)}
    end
  end

  defp build_load_opts(opts) do
    %{
      device: opts |> Keyword.get(:device) |> atom_to_string(),
      compute_type: opts |> Keyword.get(:compute_type) |> atom_to_string(),
      device_indices: Keyword.get(opts, :device_indices),
      num_threads_per_replica: Keyword.get(opts, :num_threads_per_replica),
      max_queued_batches: Keyword.get(opts, :max_queued_batches),
      cpu_core_offset: Keyword.get(opts, :cpu_core_offset)
    }
  end

  defp atom_to_string(nil), do: nil
  defp atom_to_string(value) when is_atom(value), do: Atom.to_string(value)
  defp atom_to_string(value) when is_binary(value), do: value

  @doc """
  Transcribes `audio` using `model`.

  Returns `{:ok, %WhisperCt2.Transcription{}}` on success.

  ## Options

  All CTranslate2 `WhisperOptions` knobs are accepted. Unset values fall
  through to the CTranslate2 defaults.

  - `:language` - ISO code (`"en"`); `nil` auto-detects.
  - `:timestamp` - include `<|t_..|>` tokens in the output (default `false`).
  - `:beam_size` - beam-search width (`> 0`).
  - `:patience`, `:length_penalty`, `:repetition_penalty` - decoding penalties.
  - `:no_repeat_ngram_size` - disallow repeated n-grams of this size.
  - `:sampling_temperature`, `:sampling_topk` - sampling controls.
  - `:suppress_blank` - suppress the leading blank token.
  - `:suppress_tokens` - list of token IDs to suppress.
  - `:max_length` - max generated tokens per chunk.
  - `:num_hypotheses` - number of decoded hypotheses.
  - `:return_scores`, `:return_logits_vocab`, `:return_no_speech_prob` -
    forwarded to CTranslate2; the public surface still returns text only.
  - `:max_initial_timestamp_index` - cap the first timestamp token.
  """
  @spec transcribe(Model.t(), audio(), [transcribe_opt()]) ::
          {:ok, Transcription.t()} | {:error, Error.t()}
  def transcribe(model, audio, opts \\ [])

  def transcribe(%Model{} = model, audio, opts) when is_list(opts) do
    with :ok <- validate_known_options(opts, @transcribe_options),
         :ok <- validate_transcribe_options(opts) do
      dispatch_audio(model, audio, opts)
    end
  end

  def transcribe(_model, _audio, _opts) do
    {:error, Error.new(:invalid_request, "expected a %WhisperCt2.Model{} and a keyword list")}
  end

  defp dispatch_audio(%Model{} = model, path, opts) when is_binary(path) do
    if String.ends_with?(path, ".wav") and File.regular?(path) do
      with {:ok, samples} <- Wav.read_file(path) do
        do_transcribe(model, samples, opts)
      end
    else
      do_transcribe(model, path, opts)
    end
  end

  defp dispatch_audio(%Model{} = model, {:pcm_f32, samples}, opts) when is_binary(samples) do
    do_transcribe(model, samples, opts)
  end

  defp dispatch_audio(_model, _audio, _opts) do
    {:error, Error.new(:invalid_request, "unsupported audio input")}
  end

  defp do_transcribe(%Model{ref: ref}, samples, opts) do
    if rem(byte_size(samples), 4) != 0 do
      {:error, Error.new(:invalid_request, "PCM binary length must be a multiple of 4 (f32)")}
    else
      case Native.transcribe(ref, samples, build_transcribe_opts(opts)) do
        {:ok, segments} ->
          {:ok,
           %Transcription{
             text: segments |> Enum.join(" ") |> String.trim(),
             segments: segments
           }}

        {:error, payload} ->
          {:error, Error.from_native(payload)}
      end
    end
  end

  defp build_transcribe_opts(opts) do
    %{
      language: Keyword.get(opts, :language),
      timestamp: Keyword.get(opts, :timestamp, false),
      beam_size: Keyword.get(opts, :beam_size),
      patience: Keyword.get(opts, :patience),
      length_penalty: Keyword.get(opts, :length_penalty),
      repetition_penalty: Keyword.get(opts, :repetition_penalty),
      no_repeat_ngram_size: Keyword.get(opts, :no_repeat_ngram_size),
      sampling_temperature: Keyword.get(opts, :sampling_temperature),
      sampling_topk: Keyword.get(opts, :sampling_topk),
      suppress_blank: Keyword.get(opts, :suppress_blank),
      max_length: Keyword.get(opts, :max_length),
      num_hypotheses: Keyword.get(opts, :num_hypotheses),
      return_scores: Keyword.get(opts, :return_scores),
      return_logits_vocab: Keyword.get(opts, :return_logits_vocab),
      return_no_speech_prob: Keyword.get(opts, :return_no_speech_prob),
      max_initial_timestamp_index: Keyword.get(opts, :max_initial_timestamp_index),
      suppress_tokens: Keyword.get(opts, :suppress_tokens)
    }
  end

  @spec validate_non_empty_string(String.t(), atom()) :: :ok | {:error, Error.t()}
  defp validate_non_empty_string(value, name) do
    if String.trim(value) == "" do
      {:error, Error.new(:invalid_request, "#{name} must be a non-empty string")}
    else
      :ok
    end
  end

  @spec validate_known_options(keyword(), [atom()]) :: :ok | {:error, Error.t()}
  defp validate_known_options(opts, allowed) do
    case Keyword.keys(opts) -- allowed do
      [] ->
        :ok

      [unknown | _] ->
        {:error, Error.new(:invalid_request, "unknown option #{inspect(unknown)}")}
    end
  end

  @spec validate_load_options(keyword()) :: :ok | {:error, Error.t()}
  defp validate_load_options(opts) do
    validators = %{
      device: &(&1 in @devices),
      compute_type: &(&1 in @compute_types),
      device_indices: &non_empty_list_of_non_neg_integers?/1,
      num_threads_per_replica: &non_neg_integer?/1,
      max_queued_batches: &is_integer/1,
      cpu_core_offset: &is_integer/1
    }

    validate_option_values(opts, validators)
  end

  @spec validate_transcribe_options(keyword()) :: :ok | {:error, Error.t()}
  defp validate_transcribe_options(opts) do
    validators = %{
      language: &valid_language?/1,
      timestamp: &is_boolean/1,
      beam_size: &positive_integer?/1,
      patience: &number?/1,
      length_penalty: &number?/1,
      repetition_penalty: &number?/1,
      no_repeat_ngram_size: &non_neg_integer?/1,
      sampling_temperature: &number?/1,
      sampling_topk: &positive_integer?/1,
      suppress_blank: &is_boolean/1,
      max_length: &positive_integer?/1,
      num_hypotheses: &positive_integer?/1,
      return_scores: &is_boolean/1,
      return_logits_vocab: &is_boolean/1,
      return_no_speech_prob: &is_boolean/1,
      max_initial_timestamp_index: &non_neg_integer?/1,
      suppress_tokens: &list_of_integers?/1
    }

    validate_option_values(opts, validators)
  end

  @spec validate_option_values(keyword(), map()) :: :ok | {:error, Error.t()}
  defp validate_option_values(opts, validators) do
    Enum.reduce_while(opts, :ok, fn {key, value}, :ok ->
      validator = Map.fetch!(validators, key)

      if validator.(value) do
        {:cont, :ok}
      else
        {:halt,
         {:error,
          Error.new(
            :invalid_request,
            "invalid value for option #{inspect(key)}: #{inspect(value)}"
          )}}
      end
    end)
  end

  defp valid_language?(nil), do: true
  defp valid_language?(value) when is_binary(value), do: String.trim(value) != ""
  defp valid_language?(_), do: false

  defp positive_integer?(v), do: is_integer(v) and v > 0
  defp non_neg_integer?(v), do: is_integer(v) and v >= 0
  defp number?(v), do: is_integer(v) or is_float(v)

  defp list_of_integers?(v) when is_list(v), do: Enum.all?(v, &is_integer/1)
  defp list_of_integers?(_), do: false

  defp non_empty_list_of_non_neg_integers?([_ | _] = v),
    do: Enum.all?(v, &non_neg_integer?/1)

  defp non_empty_list_of_non_neg_integers?(_), do: false
end
