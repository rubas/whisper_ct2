defmodule WhisperCt2.PcmTest do
  @moduledoc "Tests for WhisperCt2.Pcm.slice/4 - byte-math and bounds checks."

  use ExUnit.Case, async: true

  alias WhisperCt2.Error
  alias WhisperCt2.Pcm

  @sample_rate 16_000

  defp samples(seconds) do
    n = @sample_rate * seconds
    for i <- 0..(n - 1), into: <<>>, do: <<i / n::float-32-little>>
  end

  describe "slice/4" do
    test "returns the bytes for the requested window" do
      buffer = samples(2)

      assert {:ok, slice} = Pcm.slice(buffer, @sample_rate, 0.5, 1.0)
      # 1 s at 16 kHz, 4 bytes/sample => 64_000 bytes.
      assert byte_size(slice) == 64_000
    end

    test "rejects a window that extends past the end of the buffer" do
      buffer = samples(1)

      # 0.9..1.9 asks for 1 s starting at 0.9 s in a 1 s buffer.
      # The library does not clamp — caller must bound duration_s.
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               Pcm.slice(buffer, @sample_rate, 0.9, 1.0)

      assert msg =~ "extends past the end"
    end

    test "accepts a window that ends exactly at the buffer end" do
      buffer = samples(1)
      # 0.5..1.0 ends right at the buffer boundary.
      assert {:ok, slice} = Pcm.slice(buffer, @sample_rate, 0.5, 0.5)
      # 0.5 s at 16 kHz, 4 bytes/sample => 32_000 bytes.
      assert byte_size(slice) == 32_000
    end

    test "rejects start past the buffer end" do
      buffer = samples(1)

      assert {:error, %Error{reason: :invalid_request}} =
               Pcm.slice(buffer, @sample_rate, 5.0, 1.0)
    end

    test "rejects negative start" do
      buffer = samples(1)

      assert {:error, %Error{reason: :invalid_request}} =
               Pcm.slice(buffer, @sample_rate, -0.1, 1.0)
    end

    test "rejects non-positive duration" do
      buffer = samples(1)

      assert {:error, %Error{reason: :invalid_request}} =
               Pcm.slice(buffer, @sample_rate, 0.0, 0)

      assert {:error, %Error{reason: :invalid_request}} =
               Pcm.slice(buffer, @sample_rate, 0.0, -1)
    end

    test "rejects misaligned PCM length" do
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               Pcm.slice(<<1, 2, 3>>, @sample_rate, 0.0, 1.0)

      assert msg =~ "multiple of 4"
    end
  end
end
