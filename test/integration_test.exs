defmodule WhisperCt2.IntegrationTest do
  @moduledoc """
  End-to-end transcription test using the real `faster-whisper-tiny.en` model
  and the canonical JFK clip from the whisper.cpp samples.

  Excluded from the default `mix test` run because it downloads ~75 MB on
  first execution and takes ~10s of CPU per call. Run with:

      mix test --include integration

  Set `WHISPER_CT2_REFRESH=1` to force re-download of cached fixtures.
  """

  use ExUnit.Case, async: false

  alias WhisperCt2.TestFixtures

  @moduletag :integration
  @moduletag timeout: 600_000

  setup_all do
    model_dir = TestFixtures.ensure_model!()
    audio_path = TestFixtures.ensure_jfk!()
    {:ok, model} = WhisperCt2.load_model(model_dir)
    {:ok, model: model, audio_path: audio_path}
  end

  test "transcribes the JFK clip to the expected sentence", %{model: model, audio_path: audio_path} do
    assert {:ok, %WhisperCt2.Transcription{text: text}} =
             WhisperCt2.transcribe(model, audio_path, language: "en")

    normalised = text |> String.downcase() |> String.replace(~r/[^a-z ]/, " ") |> String.replace(~r/\s+/, " ")

    assert normalised =~ "ask not what your country can do for you"
  end

  test "model_info reports Whisper's standard 16kHz/30s window", %{model: model} do
    assert model.sampling_rate == 16_000
    assert model.n_samples == 480_000
  end
end
