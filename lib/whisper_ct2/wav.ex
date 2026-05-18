defmodule WhisperCt2.Wav do
  @moduledoc """
  Minimal RIFF/WAVE decoder for the formats Whisper consumes directly.

  Accepts 16 kHz audio in any of: mono / stereo 16-bit PCM, mono / stereo
  32-bit PCM, mono / stereo 32-bit float. Stereo is downmixed by averaging
  channels. Sample rates other than 16 kHz are rejected; resample upstream
  (e.g. `ffmpeg -ar 16000 -ac 1`).

  Returns samples as a binary of little-endian `f32` values in `[-1.0, 1.0]`,
  ready to feed into `WhisperCt2.Native.transcribe/3`.
  """

  alias WhisperCt2.Error

  @target_rate 16_000

  @doc """
  The sample rate this decoder produces (always 16 kHz). Exposed so the
  rest of the library can assert model sampling_rate matches without
  reaching into `@target_rate`.
  """
  @spec target_rate() :: pos_integer()
  def target_rate, do: @target_rate
  # 256 MiB cap on `read_file/1` — refuses a typo'd path or huge file
  # before slurping it into the BEAM heap; split larger inputs across
  # `transcribe_batch/3`.
  @max_bytes 268_435_456

  @spec read_file(Path.t()) :: {:ok, binary()} | {:error, Error.t()}
  def read_file(path) do
    with {:ok, bytes} <- read_bytes(path) do
      decode(bytes)
    end
  end

  @doc """
  Decodes the bytes of a WAV file into little-endian f32 PCM at 16 kHz mono.
  """
  @spec decode(binary()) :: {:ok, binary()} | {:error, Error.t()}
  def decode(<<"RIFF", _riff_size::little-32, "WAVE", rest::binary>>) do
    with {:ok, fmt, data} <- find_chunks(rest, nil, nil),
         :ok <- check_format(fmt) do
      {:ok, to_mono_f32(data, fmt)}
    end
  end

  def decode(_), do: {:error, Error.new(:invalid_request, "not a RIFF/WAVE file")}

  defp read_bytes(path) do
    case File.stat(path) do
      {:ok, %File.Stat{size: size}} when size > @max_bytes ->
        {:error,
         Error.new(:invalid_request, "WAV file exceeds the in-memory size cap", %{
           path: path,
           size: size,
           max_bytes: @max_bytes
         })}

      {:ok, _stat} ->
        case File.read(path) do
          {:ok, bytes} ->
            {:ok, bytes}

          {:error, posix} ->
            {:error, Error.new(:invalid_request, "cannot read WAV", %{posix: posix, path: path})}
        end

      {:error, posix} ->
        {:error, Error.new(:invalid_request, "cannot stat WAV", %{posix: posix, path: path})}
    end
  end

  defp find_chunks(<<>>, fmt, data) when is_map(fmt) and is_binary(data), do: {:ok, fmt, data}

  defp find_chunks(<<"fmt ", size::little-32, body::binary-size(size), rest::binary>>, _fmt, data) do
    case parse_fmt(body) do
      {:ok, fmt} -> find_chunks(skip_padding(rest, size), fmt, data)
      err -> err
    end
  end

  defp find_chunks(<<"data", size::little-32, body::binary-size(size), rest::binary>>, fmt, _data) do
    find_chunks(skip_padding(rest, size), fmt, body)
  end

  defp find_chunks(<<_tag::binary-size(4), size::little-32, _body::binary-size(size), rest::binary>>, fmt, data) do
    find_chunks(skip_padding(rest, size), fmt, data)
  end

  defp find_chunks(_, fmt, data) do
    cond do
      is_nil(fmt) -> {:error, Error.new(:invalid_request, "missing fmt chunk")}
      is_nil(data) -> {:error, Error.new(:invalid_request, "missing data chunk")}
      true -> {:ok, fmt, data}
    end
  end

  defp skip_padding(rest, size) when rem(size, 2) == 1 do
    case rest do
      <<_::binary-size(1), tail::binary>> -> tail
      _ -> rest
    end
  end

  defp skip_padding(rest, _), do: rest

  defp parse_fmt(
         <<format_tag::little-16, channels::little-16, sample_rate::little-32, _byte_rate::little-32,
           _block_align::little-16, bits::little-16, _rest::binary>>
       ) do
    {:ok,
     %{
       format_tag: format_tag,
       channels: channels,
       sample_rate: sample_rate,
       bits_per_sample: bits
     }}
  end

  defp parse_fmt(_), do: {:error, Error.new(:invalid_request, "malformed fmt chunk")}

  defp check_format(%{format_tag: tag}) when tag not in [1, 3] do
    {:error, Error.new(:invalid_request, "unsupported WAV format tag", %{format_tag: tag})}
  end

  defp check_format(%{sample_rate: sr}) when sr != @target_rate do
    {:error, Error.new(:invalid_request, "WAV must be 16 kHz mono", %{sample_rate: sr})}
  end

  defp check_format(%{bits_per_sample: bits}) when bits not in [16, 32] do
    {:error, Error.new(:invalid_request, "unsupported bits per sample", %{bits_per_sample: bits})}
  end

  defp check_format(%{channels: ch}) when ch not in [1, 2] do
    {:error, Error.new(:invalid_request, "unsupported channel count", %{channels: ch})}
  end

  defp check_format(_), do: :ok

  defp to_mono_f32(data, %{format_tag: 1, bits_per_sample: 16, channels: 1}) do
    for <<sample::little-signed-16 <- data>>, into: <<>> do
      <<sample / 32_768.0::little-float-32>>
    end
  end

  defp to_mono_f32(data, %{format_tag: 1, bits_per_sample: 16, channels: 2}) do
    for <<left::little-signed-16, right::little-signed-16 <- data>>, into: <<>> do
      <<(left + right) / 65_536.0::little-float-32>>
    end
  end

  defp to_mono_f32(data, %{format_tag: 3, bits_per_sample: 32, channels: 1}) do
    data
  end

  defp to_mono_f32(data, %{format_tag: 3, bits_per_sample: 32, channels: 2}) do
    for <<left::little-float-32, right::little-float-32 <- data>>, into: <<>> do
      <<(left + right) / 2.0::little-float-32>>
    end
  end

  defp to_mono_f32(data, %{format_tag: 1, bits_per_sample: 32, channels: 1}) do
    for <<sample::little-signed-32 <- data>>, into: <<>> do
      <<sample / 2_147_483_648.0::little-float-32>>
    end
  end

  defp to_mono_f32(data, %{format_tag: 1, bits_per_sample: 32, channels: 2}) do
    for <<left::little-signed-32, right::little-signed-32 <- data>>, into: <<>> do
      <<(left + right) / 4_294_967_296.0::little-float-32>>
    end
  end
end
