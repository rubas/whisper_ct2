defmodule WhisperCt2.Segment do
  @moduledoc """
  One `<|t_start|> text <|t_end|>` segment of a transcription.

  Times are absolute seconds within the input audio. `tokens` is the raw
  text-token ID list (timestamp tokens stripped); useful for diarization or
  custom decoding. `no_speech_prob` is the no-speech probability of the
  parent 30 s chunk, repeated on every segment in that chunk. `avg_logprob`
  is the sequence-level average log probability returned by CTranslate2 -
  filter at e.g. `avg_logprob < -1.0` to reject low-confidence hallucination.
  `words` is `nil` unless `:word_timestamps` was set on the transcribe call;
  when present it carries one `%WhisperCt2.Word{}` per Whisper word with its
  own time span.
  """

  alias WhisperCt2.Word

  @type t :: %__MODULE__{
          text: String.t(),
          start: float(),
          end: float(),
          no_speech_prob: float(),
          avg_logprob: float(),
          tokens: [non_neg_integer()],
          words: [Word.t()] | nil
        }

  @enforce_keys [:text, :start, :end, :no_speech_prob, :avg_logprob, :tokens]
  defstruct [:text, :start, :end, :no_speech_prob, :avg_logprob, :tokens, words: nil]
end
