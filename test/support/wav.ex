defmodule WhisperCt2.TestWav do
  @moduledoc """
  Minimal RIFF/WAVE decoder used by the test suite to turn a fixture WAV
  into the `f32` PCM the library now demands directly.

  Production code no longer ships a WAV decoder; resampling and decoding
  are the caller's job. The integration tests still need to feed the
  canonical `jfk.wav` clip in, so this helper lives next to the fixture
  setup and decodes once per `setup_all`.

  Accepts 16 kHz mono / stereo, 16-bit PCM, 32-bit PCM, or 32-bit float.
  Stereo is downmixed by averaging. Returns little-endian `f32` samples
  in `[-1.0, 1.0]`.
  """

  @target_rate 16_000

  @spec read_file!(Path.t()) :: binary()
  def read_file!(path) do
    path
    |> File.read!()
    |> decode!()
  end

  @spec decode!(binary()) :: binary()
  def decode!(<<"RIFF", _riff_size::little-32, "WAVE", rest::binary>>) do
    {fmt, data} = find_chunks(rest, nil, nil)
    check_format!(fmt)
    to_mono_f32(data, fmt)
  end

  def decode!(_), do: raise("not a RIFF/WAVE file")

  defp find_chunks(<<>>, fmt, data) when is_map(fmt) and is_binary(data), do: {fmt, data}

  defp find_chunks(<<"fmt ", size::little-32, body::binary-size(size), rest::binary>>, _fmt, data) do
    find_chunks(skip_padding(rest, size), parse_fmt!(body), data)
  end

  defp find_chunks(<<"data", size::little-32, body::binary-size(size), rest::binary>>, fmt, _data) do
    find_chunks(skip_padding(rest, size), fmt, body)
  end

  defp find_chunks(<<_tag::binary-size(4), size::little-32, _body::binary-size(size), rest::binary>>, fmt, data) do
    find_chunks(skip_padding(rest, size), fmt, data)
  end

  defp find_chunks(_, fmt, data) when is_map(fmt) and is_binary(data), do: {fmt, data}
  defp find_chunks(_, _, _), do: raise("WAV is missing fmt or data chunk")

  defp skip_padding(rest, size) when rem(size, 2) == 1 do
    case rest do
      <<_::binary-size(1), tail::binary>> -> tail
      _ -> rest
    end
  end

  defp skip_padding(rest, _), do: rest

  defp parse_fmt!(
         <<format_tag::little-16, channels::little-16, sample_rate::little-32, _byte_rate::little-32,
           _block_align::little-16, bits::little-16, _rest::binary>>
       ) do
    %{
      format_tag: format_tag,
      channels: channels,
      sample_rate: sample_rate,
      bits_per_sample: bits
    }
  end

  defp parse_fmt!(_), do: raise("malformed fmt chunk")

  defp check_format!(%{format_tag: tag}) when tag not in [1, 3],
    do: raise("unsupported WAV format tag #{tag}")

  defp check_format!(%{sample_rate: sr}) when sr != @target_rate,
    do: raise("WAV must be #{@target_rate} Hz, got #{sr}")

  defp check_format!(%{bits_per_sample: bits}) when bits not in [16, 32],
    do: raise("unsupported bits per sample #{bits}")

  defp check_format!(%{channels: ch}) when ch not in [1, 2],
    do: raise("unsupported channel count #{ch}")

  defp check_format!(_), do: :ok

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

  defp to_mono_f32(data, %{format_tag: 3, bits_per_sample: 32, channels: 1}), do: data

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
