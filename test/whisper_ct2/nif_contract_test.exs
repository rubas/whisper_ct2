defmodule WhisperCt2.NifContractTest do
  @moduledoc """
  Pins the map shape `nif_transcribe` / `nif_transcribe_batch` are expected
  to produce.

  The Rust NIF derives `NifMap` on `NifTranscription`, `NifSegment`, and
  `NifWord`, which encode as atom-keyed Elixir maps. The Elixir wrapper
  pattern-matches strictly on those exact key sets to build structs.

  Negative shape-mismatch cases are not tested at runtime: the strict
  pattern match in each `build_*` head is a static contract the Elixir
  1.18 set-theoretic typechecker enforces at every call site. A field
  rename on the Rust side updates the `NifMap` derive output; the
  Elixir wrapper code consuming it would then be a known-bad call that
  the typechecker flags at compile time. The positive tests below pin
  which atom keys with which value types map to which struct fields.
  """

  use ExUnit.Case, async: true

  alias WhisperCt2.{Segment, Transcription, Word}

  @nif_word %{text: "hello", start: 0.0, end: 0.5, probability: 0.9}

  @nif_segment %{
    text: "hello world",
    start: 0.0,
    end: 1.2,
    no_speech_prob: 0.01,
    avg_logprob: -0.4,
    tokens: [50_257, 1, 2, 3],
    words: [@nif_word]
  }

  @nif_transcription %{
    language: "en",
    duration_s: 1.5,
    segments: [@nif_segment]
  }

  describe "build_word/1" do
    test "accepts the exact NIF map shape" do
      assert %Word{text: "hello", start: +0.0, end: 0.5, probability: 0.9} =
               WhisperCt2.build_word(@nif_word)
    end
  end

  describe "build_segment/1" do
    test "accepts the exact NIF map shape with words present" do
      assert %Segment{
               text: "hello world",
               start: +0.0,
               end: 1.2,
               no_speech_prob: 0.01,
               avg_logprob: -0.4,
               tokens: [50_257, 1, 2, 3],
               words: [%Word{text: "hello"}]
             } = WhisperCt2.build_segment(@nif_segment)
    end

    test "accepts nil words (word_timestamps disabled)" do
      assert %Segment{words: nil} =
               WhisperCt2.build_segment(%{@nif_segment | words: nil})
    end
  end

  describe "build_transcription/1" do
    test "accepts the exact NIF map shape" do
      assert %Transcription{
               text: "hello world",
               language: "en",
               duration_s: 1.5,
               segments: [%Segment{}]
             } = WhisperCt2.build_transcription(@nif_transcription)
    end

    test "joins segment texts with a single space and trims" do
      payload = %{
        @nif_transcription
        | segments: [
            %{@nif_segment | text: "  hello"},
            %{@nif_segment | text: "world  "}
          ]
      }

      assert %Transcription{text: "hello world"} = WhisperCt2.build_transcription(payload)
    end
  end
end
