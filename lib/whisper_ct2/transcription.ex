defmodule WhisperCt2.Transcription do
  @moduledoc """
  Result of a `WhisperCt2.transcribe/3` call.

  `:text` is the concatenated, whitespace-trimmed transcript. `:segments` is
  the raw per-chunk string list returned by CTranslate2 — useful when callers
  passed audio longer than the 30 s Whisper window and need to align chunks
  with timestamps managed outside this library.
  """

  @type t :: %__MODULE__{
          text: String.t(),
          segments: [String.t()]
        }

  @enforce_keys [:text, :segments]
  defstruct [:text, :segments]
end
