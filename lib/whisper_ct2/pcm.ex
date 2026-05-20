defmodule WhisperCt2.Pcm do
  @moduledoc """
  Helpers for slicing little-endian f32 PCM buffers without re-decoding the
  source audio.

  A diarization-driven workflow typically decodes the master audio to
  f32 PCM once (upstream of this library) and then runs many short
  transcribe calls over per-turn slices. `slice/4` does the byte math
  (4 bytes per sample at `sample_rate` samples/second) and bounds-checks
  against the buffer size.
  """

  alias WhisperCt2.Error

  @bytes_per_sample 4

  @doc """
  Returns the f32-PCM bytes for `[start_s, start_s + duration_s)`.

  `samples` is a binary of little-endian `f32` samples at `sample_rate`
  samples/second. Both `start_s` and `duration_s` are seconds; either
  integer or float.

  Returns `{:error, %WhisperCt2.Error{reason: :invalid_request}}` when:

  - `start_s < 0`;
  - `duration_s` is zero or negative, or rounds to zero samples;
  - the requested window is not fully contained in the buffer
    (`start_sample + requested_samples > total_samples`).

  The range is **not** clamped — slicing past the end is a caller bug
  and we surface it loudly. Bound your duration before calling, or use
  `min(duration_s, buffer_duration - start_s)`.
  """
  @spec slice(binary(), pos_integer(), number(), number()) ::
          {:ok, binary()} | {:error, Error.t()}
  def slice(samples, sample_rate, start_s, duration_s)
      when is_binary(samples) and is_integer(sample_rate) and sample_rate > 0 and
             is_number(start_s) and is_number(duration_s) do
    cond do
      start_s < 0 ->
        {:error, Error.new(:invalid_request, "start_s must be >= 0", %{start_s: start_s})}

      duration_s <= 0 ->
        {:error, Error.new(:invalid_request, "duration_s must be > 0", %{duration_s: duration_s})}

      rem(byte_size(samples), @bytes_per_sample) != 0 ->
        {:error,
         Error.new(:invalid_request, "samples binary length must be a multiple of 4 (f32)", %{
           byte_size: byte_size(samples)
         })}

      true ->
        do_slice(samples, sample_rate, start_s, duration_s)
    end
  end

  def slice(_samples, _sample_rate, _start_s, _duration_s) do
    {:error, Error.new(:invalid_request, "invalid arguments to WhisperCt2.Pcm.slice/4")}
  end

  defp do_slice(samples, sample_rate, start_s, duration_s) do
    total_samples = div(byte_size(samples), @bytes_per_sample)
    start_sample = trunc(start_s * sample_rate)
    requested_samples = trunc(duration_s * sample_rate)
    end_sample = start_sample + requested_samples
    buffer_duration_s = total_samples / sample_rate

    cond do
      requested_samples <= 0 ->
        {:error,
         Error.new(:invalid_request, "duration_s rounds to zero samples", %{
           duration_s: duration_s,
           sample_rate: sample_rate
         })}

      start_sample >= total_samples ->
        {:error,
         Error.new(:invalid_request, "start_s is past the end of the buffer", %{
           start_s: start_s,
           buffer_duration_s: buffer_duration_s
         })}

      end_sample > total_samples ->
        {:error,
         Error.new(
           :invalid_request,
           "requested window extends past the end of the buffer; bound duration_s yourself",
           %{
             start_s: start_s,
             duration_s: duration_s,
             buffer_duration_s: buffer_duration_s
           }
         )}

      true ->
        byte_start = start_sample * @bytes_per_sample
        byte_len = requested_samples * @bytes_per_sample
        {:ok, binary_part(samples, byte_start, byte_len)}
    end
  end
end
