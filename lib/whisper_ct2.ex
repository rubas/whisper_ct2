defmodule WhisperCt2 do
  @moduledoc """
  Native Elixir bindings for Whisper speech-to-text via CTranslate2.

  Calls `ct2rs::sys::Whisper` directly through a Rustler NIF: no Python, no
  HTTP gateway. The NIF owns the mel spectrogram, tokenizer, and prompt
  construction, so structured per-segment results, `:initial_prompt` /
  `:prefix` biasing, word-level timestamps, and batched transcribe across
  multiple audios are all first-class.

  ## Quickstart

      {:ok, model} = WhisperCt2.load_model("/path/to/faster-whisper-tiny")

      {:ok, %WhisperCt2.Transcription{text: text, segments: segs}} =
        WhisperCt2.transcribe(model, "audio.wav", language: "en")

      IO.puts(text)
      for s <- segs, do: IO.puts("[\#{s.start}-\#{s.end}] \#{s.text}")

  ## Audio contract

  CTranslate2 expects **mono `f32` PCM samples** at the model's
  `:sampling_rate` (16 kHz for every published Whisper checkpoint),
  normalised to the `-1.0..1.0` range. `transcribe/3` accepts:

  - a `.wav` path (16 kHz mono or stereo, 16/32-bit PCM, or 32-bit float),
    decoded by the built-in `WhisperCt2.Wav` module;
  - a `{:pcm_f32, binary}` tuple containing little-endian f32 samples.

  Raw bare-binary input is rejected on purpose: a typo'd path used to
  silently turn into garbage PCM. Use `{:pcm_f32, binary}` for in-memory
  buffers.

  Audio longer than the Whisper 30 s window is chunked internally; the
  encoder runs once across every chunk in the batch. Diarization-driven
  workflows that need many short splices should use
  `transcribe_batch/3`.
  """

  alias WhisperCt2.{Error, Model, Native, Segment, Transcription, Wav, Word}

  @typedoc "Audio sources accepted by `transcribe/3` and `transcribe_batch/3`."
  @type audio :: Path.t() | {:pcm_f32, binary()}

  @typedoc "Options accepted by `transcribe/3` / `transcribe_batch/3`."
  @type transcribe_opt ::
          {:language, String.t() | nil}
          | {:initial_prompt, String.t() | nil}
          | {:prefix, String.t() | nil}
          | {:word_timestamps, boolean()}
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
          | {:max_initial_timestamp_index, non_neg_integer()}
          | {:suppress_tokens, [integer()]}

  @typedoc "Options accepted by `load_model/2`."
  @type load_opt ::
          {:device, :cpu | :cuda | :auto}
          | {:compute_type, Model.compute_type()}
          | {:device_indices, [non_neg_integer(), ...]}
          | {:num_threads_per_replica, non_neg_integer()}
          | {:max_queued_batches, integer()}
          | {:cpu_core_offset, integer()}

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

  Returns `{:ok, %{cpu: n, cuda: n, cuda_supported: bool}}` on success.
  `cuda_supported` reflects compile-time CUDA features (build with
  `WHISPER_CT2_FEATURES=cuda-dynamic mix compile` to enable). `cuda` is the
  count of CUDA devices visible at runtime, or `0` when CUDA is not built in.
  """
  @spec available_devices() ::
          {:ok, %{cpu: non_neg_integer(), cuda: non_neg_integer(), cuda_supported: boolean()}}
          | {:error, Error.t()}
  def available_devices do
    case Native.available_devices() do
      {:ok, info} -> {:ok, info}
      {:error, payload} -> {:error, Error.from_native(payload)}
    end
  end

  @doc """
  Loads a CTranslate2 Whisper model from a directory.

  See the `WhisperCt2` module doc for required model files.

  ## Options

  - `:device` - `:cpu`, `:cuda`, or `:auto` (default). `:auto` picks CUDA
    when the binary was built with CUDA support and a device is visible;
    otherwise CPU.
  - `:compute_type` - precision used at inference. `:default` keeps the
    model's stored quantisation; `:auto` picks the fastest supported on
    this device.
  - `:device_indices` - non-empty list of GPU indices (default `[0]`).
  - `:num_threads_per_replica` - intra-op threads. `0` lets CTranslate2 pick.
  - `:max_queued_batches`, `:cpu_core_offset` - passed through to
    CTranslate2.
  """
  @spec load_model(Path.t(), [load_opt()]) :: {:ok, Model.t()} | {:error, Error.t()}
  def load_model(path, opts \\ [])

  def load_model(path, opts) when is_binary(path) and is_list(opts) do
    with :ok <- validate_non_empty_string(path, :path),
         :ok <- validate_options(opts, load_validators()) do
      do_load_model(path, opts)
    end
  end

  def load_model(_path, _opts) do
    {:error, Error.new(:invalid_request, "path must be a string and opts a keyword list")}
  end

  defp do_load_model(path, opts) do
    with {:ok, ref} <- native_call(Native.load_model(path, build_load_opts(opts))),
         {:ok, info} <- native_call(Native.model_info(ref)),
         :ok <- assert_sampling_rate(info.sampling_rate, path),
         {:ok, device} <- decode_device(info.device),
         {:ok, compute_type} <- decode_compute_type(info.compute_type) do
      {:ok,
       %Model{
         ref: ref,
         path: path,
         sampling_rate: info.sampling_rate,
         n_samples: info.n_samples,
         multilingual: info.multilingual,
         device: device,
         compute_type: compute_type
       }}
    end
  end

  # Every published Whisper checkpoint preprocesses at 16 kHz, and the
  # in-tree WAV decoder hardcodes the same rate. Fail load loudly if a
  # model reports anything else, rather than silently feeding it WAV
  # samples at the wrong rate.
  defp assert_sampling_rate(rate, path) do
    expected = Wav.target_rate()

    if rate == expected do
      :ok
    else
      {:error,
       Error.new(
         :load_error,
         "model sampling_rate does not match the bundled WAV decoder rate; \
          resample upstream or load a 16 kHz checkpoint",
         %{path: path, model_sampling_rate: rate, expected: expected}
       )}
    end
  end

  @device_atoms Map.new(@devices, fn a -> {Atom.to_string(a), a} end)
  @compute_type_atoms Map.new(@compute_types, fn a -> {Atom.to_string(a), a} end)

  defp decode_device(label) when is_binary(label) do
    case Map.fetch(@device_atoms, label) do
      {:ok, atom} ->
        {:ok, atom}

      :error ->
        {:error, Error.new(:runtime_error, "NIF reported unknown device", %{device: label})}
    end
  end

  defp decode_compute_type(label) when is_binary(label) do
    case Map.fetch(@compute_type_atoms, label) do
      {:ok, atom} ->
        {:ok, atom}

      :error ->
        {:error, Error.new(:runtime_error, "NIF reported unknown compute_type", %{compute_type: label})}
    end
  end

  defp native_call({:ok, _} = ok), do: ok
  defp native_call({:error, payload}), do: {:error, Error.from_native(payload)}

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

  @doc """
  Transcribes `audio` using `model`.

  Returns `{:ok, %WhisperCt2.Transcription{}}` whose `:segments` carry
  absolute start/end times, `no_speech_prob`, `avg_logprob`, the
  underlying text tokens, and (when `:word_timestamps` is set) per-word
  timing. `no_speech_prob` and `avg_logprob` are always populated.

  ## Options

  - `:language` - ISO code (`"en"`). `nil` (default) auto-detects.
  - `:initial_prompt` - free-text context prepended via `<|startofprev|>`
    to bias decoding.
  - `:prefix` - forced text the generation must start with.
  - `:word_timestamps` - when `true`, attaches `:words` to each segment
    via one extra batched DTW alignment pass. Default `false`.
  - Decoding knobs forwarded to CTranslate2: `:beam_size`, `:patience`,
    `:length_penalty`, `:repetition_penalty`, `:no_repeat_ngram_size`,
    `:sampling_temperature`, `:sampling_topk`, `:suppress_blank`,
    `:max_length`, `:num_hypotheses`, `:max_initial_timestamp_index`,
    `:suppress_tokens`.
  """
  @spec transcribe(Model.t(), audio(), [transcribe_opt()]) ::
          {:ok, Transcription.t()} | {:error, Error.t()}
  def transcribe(model, audio, opts \\ [])

  def transcribe(%Model{} = model, audio, opts) when is_list(opts) do
    with :ok <- validate_options(opts, transcribe_validators()),
         {:ok, samples} <- resolve_audio(audio) do
      do_transcribe(model, samples, opts)
    end
  end

  def transcribe(_model, _audio, _opts) do
    {:error, Error.new(:invalid_request, "expected a %WhisperCt2.Model{} and a keyword list")}
  end

  defp do_transcribe(%Model{ref: ref}, samples, opts) do
    case Native.transcribe(ref, samples, build_transcribe_opts(opts)) do
      {:ok, payload} -> {:ok, build_transcription(payload)}
      {:error, payload} -> {:error, Error.from_native(payload)}
    end
  end

  @doc """
  Transcribes a list of audios in one batched `generate` call. Every
  chunk of every input shares a single encoder forward pass; output
  preserves input order.

  Options are the same as `transcribe/3`. `:language` applies to every
  audio in the batch; pass `nil` to auto-detect per-audio.
  """
  @spec transcribe_batch(Model.t(), [audio()], [transcribe_opt()]) ::
          {:ok, [Transcription.t()]} | {:error, Error.t()}
  def transcribe_batch(model, audios, opts \\ [])

  def transcribe_batch(%Model{} = _model, [], _opts), do: {:ok, []}

  def transcribe_batch(%Model{} = model, audios, opts)
      when is_list(audios) and is_list(opts) do
    with :ok <- validate_options(opts, transcribe_validators()),
         {:ok, samples_list} <- resolve_audios(audios) do
      do_transcribe_batch(model, samples_list, opts)
    end
  end

  def transcribe_batch(_model, _audios, _opts) do
    {:error,
     Error.new(
       :invalid_request,
       "expected a %WhisperCt2.Model{}, a list of audios, and a keyword list"
     )}
  end

  defp do_transcribe_batch(%Model{ref: ref}, samples_list, opts) do
    case Native.transcribe_batch(ref, samples_list, build_transcribe_opts(opts)) do
      {:ok, payloads} ->
        {:ok, Enum.map(payloads, &build_transcription/1)}

      {:error, payload} ->
        {:error, Error.from_native(payload)}
    end
  end

  defp resolve_audios(audios) do
    result =
      Enum.reduce_while(audios, {:ok, []}, fn audio, {:ok, acc} ->
        case resolve_audio(audio) do
          {:ok, samples} -> {:cont, {:ok, [samples | acc]}}
          {:error, _} = err -> {:halt, err}
        end
      end)

    case result do
      {:ok, reversed} -> {:ok, Enum.reverse(reversed)}
      err -> err
    end
  end

  defp resolve_audio({:pcm_f32, samples}) when is_binary(samples) do
    cond do
      byte_size(samples) == 0 ->
        {:error, Error.new(:invalid_request, "PCM binary is empty")}

      rem(byte_size(samples), 4) != 0 ->
        {:error,
         Error.new(:invalid_request, "PCM binary length must be a multiple of 4 (f32)", %{
           byte_size: byte_size(samples)
         })}

      true ->
        {:ok, samples}
    end
  end

  defp resolve_audio(path) when is_binary(path) do
    cond do
      not File.regular?(path) ->
        {:error,
         Error.new(
           :invalid_request,
           "audio path does not exist or is not a regular file",
           %{path: path}
         )}

      String.ends_with?(path, ".wav") ->
        Wav.read_file(path)

      true ->
        {:error,
         Error.new(
           :invalid_request,
           "only .wav paths are accepted; resample/decode upstream or pass {:pcm_f32, binary}",
           %{path: path}
         )}
    end
  end

  defp resolve_audio(_) do
    {:error, Error.new(:invalid_request, "audio must be a .wav path or {:pcm_f32, binary}")}
  end

  # The three `build_*` functions pattern-match the NIF map shape
  # strictly in the function head. That makes the shape a static
  # contract: the Elixir 1.18 typechecker proves any caller that passes
  # a map it can't show fits the head will not match, and surfaces it as
  # a type warning at compile time. Exposed as `@doc false def` so the
  # contract tests in `nif_contract_test.exs` can pin which atom keys
  # map to which struct fields without a loaded model; not part of the
  # public API.

  @doc false
  @spec build_transcription(map()) :: Transcription.t()
  def build_transcription(%{
        language: language,
        duration_s: duration_s,
        segments: raw_segments
      }) do
    segments = Enum.map(raw_segments, &build_segment/1)

    text =
      segments
      |> Enum.map_join(" ", & &1.text)
      |> String.trim()

    %Transcription{
      text: text,
      segments: segments,
      language: language,
      duration_s: duration_s
    }
  end

  @doc false
  @spec build_segment(map()) :: Segment.t()
  def build_segment(%{
        text: text,
        start: start,
        end: end_s,
        no_speech_prob: no_speech_prob,
        avg_logprob: avg_logprob,
        tokens: tokens,
        words: words
      }) do
    %Segment{
      text: text,
      start: start,
      end: end_s,
      no_speech_prob: no_speech_prob,
      avg_logprob: avg_logprob,
      tokens: tokens,
      words: words && Enum.map(words, &build_word/1)
    }
  end

  @doc false
  @spec build_word(map()) :: Word.t()
  def build_word(%{text: text, start: start, end: end_s, probability: probability}) do
    %Word{text: text, start: start, end: end_s, probability: probability}
  end

  defp build_transcribe_opts(opts) do
    %{
      language: Keyword.get(opts, :language),
      initial_prompt: Keyword.get(opts, :initial_prompt),
      prefix: Keyword.get(opts, :prefix),
      word_timestamps: Keyword.get(opts, :word_timestamps),
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

  defp load_validators do
    %{
      device: &(&1 in @devices),
      compute_type: &(&1 in @compute_types),
      device_indices: &non_empty_list_of_non_neg_integers?/1,
      num_threads_per_replica: &non_neg_integer?/1,
      max_queued_batches: &is_integer/1,
      cpu_core_offset: &is_integer/1
    }
  end

  defp transcribe_validators do
    %{
      language: &valid_optional_string?/1,
      initial_prompt: &valid_optional_string?/1,
      prefix: &valid_optional_string?/1,
      word_timestamps: &is_boolean/1,
      beam_size: &positive_integer?/1,
      # `patience` is faster-whisper's beam-search patience; values < 1.0
      # are documented as effectively disabling it.
      patience: &positive_number?/1,
      # CTranslate2 accepts any sign for `length_penalty`, including
      # negative values that bias toward shorter generations.
      length_penalty: &number?/1,
      # `repetition_penalty` < 1.0 amplifies repetition; documented values
      # are >= 1.0 (1.0 = neutral). Reject < 1.0 at the boundary.
      repetition_penalty: &repetition_penalty?/1,
      no_repeat_ngram_size: &non_neg_integer?/1,
      # Negative temperatures are nonsensical; 0.0 = greedy.
      sampling_temperature: &non_neg_number?/1,
      sampling_topk: &positive_integer?/1,
      suppress_blank: &is_boolean/1,
      max_length: &positive_integer?/1,
      num_hypotheses: &positive_integer?/1,
      max_initial_timestamp_index: &non_neg_integer?/1,
      suppress_tokens: &list_of_integers?/1
    }
  end

  @spec validate_options(keyword(), map()) :: :ok | {:error, Error.t()}
  defp validate_options(opts, validators) do
    Enum.reduce_while(opts, :ok, fn pair, :ok -> check_option(pair, validators) end)
  end

  defp check_option({key, value}, validators) do
    case Map.fetch(validators, key) do
      :error ->
        {:halt, {:error, Error.new(:invalid_request, "unknown option #{inspect(key)}")}}

      {:ok, validator} ->
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
    end
  end

  defp valid_optional_string?(nil), do: true
  defp valid_optional_string?(value) when is_binary(value), do: String.trim(value) != ""
  defp valid_optional_string?(_), do: false

  defp positive_integer?(v), do: is_integer(v) and v > 0
  defp non_neg_integer?(v), do: is_integer(v) and v >= 0
  defp number?(v), do: is_integer(v) or is_float(v)
  defp positive_number?(v), do: number?(v) and v > 0
  defp non_neg_number?(v), do: number?(v) and v >= 0
  defp repetition_penalty?(v), do: number?(v) and v >= 1

  defp list_of_integers?(v) when is_list(v), do: Enum.all?(v, &is_integer/1)
  defp list_of_integers?(_), do: false

  defp non_empty_list_of_non_neg_integers?([_ | _] = v),
    do: Enum.all?(v, &non_neg_integer?/1)

  defp non_empty_list_of_non_neg_integers?(_), do: false
end
