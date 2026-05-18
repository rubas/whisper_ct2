defmodule WhisperCt2.WavTest do
  @moduledoc """
  Tests WAV byte decoding into mono little-endian f32 PCM.

  These tests build minimal WAV binaries in memory and do not cover resampling
  or external audio tools.
  """

  use ExUnit.Case, async: true

  alias WhisperCt2.Wav

  defp wav_header(channels, sample_rate, bits, format_tag, data_size) do
    block_align = div(channels * bits, 8)
    byte_rate = sample_rate * block_align
    fmt_chunk_size = 16
    riff_size = 4 + (8 + fmt_chunk_size) + (8 + data_size)

    <<"RIFF", riff_size::little-32, "WAVE", "fmt ", fmt_chunk_size::little-32, format_tag::little-16,
      channels::little-16, sample_rate::little-32, byte_rate::little-32, block_align::little-16, bits::little-16,
      "data", data_size::little-32>>
  end

  test "decodes mono 16-bit 16kHz PCM" do
    samples = for i <- 1..4, do: <<i * 1000::little-signed-16>>
    data = IO.iodata_to_binary(samples)
    wav = wav_header(1, 16_000, 16, 1, byte_size(data)) <> data

    assert {:ok, pcm} = Wav.decode(wav)
    assert byte_size(pcm) == 4 * 4

    floats =
      for <<f::little-float-32 <- pcm>>, do: Float.round(f, 6)

    assert floats == [
             Float.round(1_000 / 32_768.0, 6),
             Float.round(2_000 / 32_768.0, 6),
             Float.round(3_000 / 32_768.0, 6),
             Float.round(4_000 / 32_768.0, 6)
           ]
  end

  test "downmixes stereo 16-bit by averaging" do
    data = <<10_000::little-signed-16, -10_000::little-signed-16, 100::little-signed-16, 200::little-signed-16>>
    wav = wav_header(2, 16_000, 16, 1, byte_size(data)) <> data

    assert {:ok, pcm} = Wav.decode(wav)
    assert <<a::little-float-32, b::little-float-32>> = pcm
    assert_in_delta a, 0.0, 1.0e-6
    assert_in_delta b, 300 / 65_536.0, 1.0e-6
  end

  test "passes 32-bit float mono through unchanged" do
    data = <<0.5::little-float-32, -0.25::little-float-32>>
    wav = wav_header(1, 16_000, 32, 3, byte_size(data)) <> data

    assert {:ok, ^data} = Wav.decode(wav)
  end

  test "downmixes 32-bit float stereo by averaging" do
    data = <<0.5::little-float-32, -0.5::little-float-32, 0.25::little-float-32, 0.75::little-float-32>>
    wav = wav_header(2, 16_000, 32, 3, byte_size(data)) <> data

    assert {:ok, pcm} = Wav.decode(wav)
    assert <<a::little-float-32, b::little-float-32>> = pcm
    assert_in_delta a, 0.0, 1.0e-6
    assert_in_delta b, 0.5, 1.0e-6
  end

  test "decodes 32-bit signed int mono" do
    # Half-scale positive sample: 2^30 = 1_073_741_824 / 2^31 = 0.5.
    data = <<1_073_741_824::little-signed-32, -1_073_741_824::little-signed-32>>
    wav = wav_header(1, 16_000, 32, 1, byte_size(data)) <> data

    assert {:ok, pcm} = Wav.decode(wav)
    assert <<a::little-float-32, b::little-float-32>> = pcm
    assert_in_delta a, 0.5, 1.0e-6
    assert_in_delta b, -0.5, 1.0e-6
  end

  test "downmixes 32-bit signed int stereo by averaging" do
    data =
      <<1_073_741_824::little-signed-32, -1_073_741_824::little-signed-32, 1_073_741_824::little-signed-32,
        1_073_741_824::little-signed-32>>

    wav = wav_header(2, 16_000, 32, 1, byte_size(data)) <> data

    assert {:ok, pcm} = Wav.decode(wav)
    assert <<a::little-float-32, b::little-float-32>> = pcm
    assert_in_delta a, 0.0, 1.0e-6
    assert_in_delta b, 0.5, 1.0e-6
  end

  test "rejects non-16kHz audio" do
    data = <<0::little-signed-16>>
    wav = wav_header(1, 44_100, 16, 1, byte_size(data)) <> data

    assert {:error, %WhisperCt2.Error{reason: :invalid_request, message: msg}} = Wav.decode(wav)
    assert msg =~ "16 kHz"
  end

  test "rejects unsupported bit depth" do
    data = <<0::little-signed-32, 0::little-signed-32>>
    block = div(1 * 24, 8)
    fmt_size = 16
    data_size = byte_size(data)
    riff_size = 4 + (8 + fmt_size) + (8 + data_size)

    wav =
      <<"RIFF", riff_size::little-32, "WAVE", "fmt ", fmt_size::little-32, 1::little-16, 1::little-16,
        16_000::little-32, 16_000 * block::little-32, block::little-16, 24::little-16, "data", data_size::little-32,
        data::binary>>

    assert {:error, %WhisperCt2.Error{reason: :invalid_request}} = Wav.decode(wav)
  end

  test "rejects non-RIFF input" do
    assert {:error, %WhisperCt2.Error{reason: :invalid_request, message: "not a RIFF/WAVE file"}} =
             Wav.decode(<<"NOPE", 0::little-32>>)
  end
end
