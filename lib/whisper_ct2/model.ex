defmodule WhisperCt2.Model do
  @moduledoc """
  Loaded Whisper model handle.

  Holds an opaque NIF reference plus cached metadata (`:sampling_rate`,
  `:n_samples`, `:multilingual`). The reference is garbage-collected by the
  BEAM when no longer reachable; CTranslate2 frees the model at that point.
  """

  @type device :: :cpu | :cuda
  @type compute_type ::
          :default
          | :auto
          | :float32
          | :float16
          | :bfloat16
          | :int8
          | :int8_float32
          | :int8_float16
          | :int8_bfloat16
          | :int16

  @type t :: %__MODULE__{
          ref: reference(),
          path: Path.t(),
          sampling_rate: pos_integer(),
          n_samples: pos_integer(),
          multilingual: boolean(),
          device: device(),
          compute_type: compute_type()
        }

  @enforce_keys [:ref, :path, :sampling_rate, :n_samples, :multilingual, :device, :compute_type]
  defstruct [:ref, :path, :sampling_rate, :n_samples, :multilingual, :device, :compute_type]
end
