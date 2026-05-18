defmodule WhisperCt2.Word do
  @moduledoc """
  One word produced by `:word_timestamps`.

  Times are absolute seconds within the input audio. `probability` is the
  mean of per-token acoustic probabilities from the DTW alignment pass.
  """

  @type t :: %__MODULE__{
          text: String.t(),
          start: float(),
          end: float(),
          probability: float()
        }

  @enforce_keys [:text, :start, :end, :probability]
  defstruct [:text, :start, :end, :probability]
end
