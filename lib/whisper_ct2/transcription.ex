defmodule WhisperCt2.Transcription do
  @moduledoc """
  Result of a `WhisperCt2.transcribe/3` call.

  `text` is the concatenated, whitespace-trimmed transcript across every
  segment. `segments` is the structured per-`<|t_..|>` decomposition
  produced by CTranslate2, with absolute start/end times in seconds,
  `no_speech_prob`, sequence-level `avg_logprob`, and the underlying token
  IDs. `language` is the resolved ISO code (auto-detected when not pinned).
  `duration_s` is the input audio length, useful for VAD/diarization
  pipelines that hand short splices in.
  """

  alias WhisperCt2.Segment

  @type t :: %__MODULE__{
          text: String.t(),
          segments: [Segment.t()],
          language: String.t(),
          duration_s: float()
        }

  @enforce_keys [:text, :segments, :language, :duration_s]
  defstruct [:text, :segments, :language, :duration_s]
end
