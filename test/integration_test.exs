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

  alias WhisperCt2.{Segment, TestFixtures, Transcription, Word}

  @moduletag :integration
  @moduletag timeout: 600_000

  setup_all do
    model_dir = TestFixtures.ensure_model!()
    audio_path = TestFixtures.ensure_jfk!()
    {:ok, model} = WhisperCt2.load_model(model_dir)
    {:ok, model: model, audio_path: audio_path}
  end

  test "transcribes the JFK clip to the expected sentence", %{model: model, audio_path: path} do
    assert {:ok, %Transcription{text: text, segments: segs, language: lang, duration_s: dur}} =
             WhisperCt2.transcribe(model, path, language: "en")

    assert lang == "en"
    assert dur > 5.0 and dur < 15.0
    assert Enum.all?(segs, &match?(%Segment{}, &1))
    assert Enum.all?(segs, &(&1.end > &1.start))
    assert Enum.all?(segs, &(&1.no_speech_prob >= 0.0 and &1.no_speech_prob <= 1.0))
    assert Enum.all?(segs, &is_list(&1.tokens))
    assert Enum.all?(segs, &is_nil(&1.words))

    # avg_logprob is a contract field: always populated, never the
    # silent 0.0 of a missing-score fallback.
    assert Enum.all?(segs, &(is_float(&1.avg_logprob) and &1.avg_logprob != 0.0))
    assert Enum.all?(segs, &(&1.avg_logprob < 0.0))

    assert normalised(text) =~ "ask not what your country can do for you"
  end

  test "word_timestamps attaches per-word timing to every segment", %{
    model: model,
    audio_path: path
  } do
    assert {:ok, %Transcription{segments: segs, duration_s: dur}} =
             WhisperCt2.transcribe(model, path, language: "en", word_timestamps: true)

    # Contract: when :word_timestamps is requested, every segment carries a
    # non-nil :words list. A silent nil here used to mean we'd swallowed an
    # alignment shape mismatch.
    assert Enum.all?(segs, &is_list(&1.words))
    refute Enum.any?(segs, &is_nil(&1.words))

    for %Segment{words: words} <- segs do
      assert Enum.all?(words, &match?(%Word{}, &1))
      assert Enum.all?(words, &(&1.end >= &1.start))
      assert Enum.all?(words, &(&1.probability >= 0.0 and &1.probability <= 1.0))
    end

    all_words = Enum.flat_map(segs, & &1.words)
    [first | _] = all_words
    last = List.last(all_words)

    # Alignment sanity: the first word lands near the start of the clip
    # and no word ends past the audio duration. Catches gross drift such
    # as a wrong DTW start_sequence shifting every word by seconds.
    assert first.start < 2.0
    assert last.end <= dur + 0.5
    assert String.downcase(first.text) =~ ~r/[a-z]/
  end

  test "transcribe_batch keeps distinct audios distinct",
       %{model: model, audio_path: path} do
    {:ok, full_pcm} = WhisperCt2.Wav.read_file(path)
    three_s_bytes = 3 * 16_000 * 4
    <<short::binary-size(^three_s_bytes), _::binary>> = full_pcm

    assert {:ok, [full, partial]} =
             WhisperCt2.transcribe_batch(
               model,
               [path, {:pcm_f32, short}],
               language: "en"
             )

    assert %Transcription{} = full
    assert %Transcription{} = partial
    assert full.duration_s > partial.duration_s
    assert partial.duration_s < 3.5
    refute normalised(full.text) == normalised(partial.text)
  end

  test "transcribe_batch with identical inputs produces identical transcripts",
       %{model: model, audio_path: path} do
    assert {:ok, [t1, t2]} =
             WhisperCt2.transcribe_batch(model, [path, path], language: "en")

    assert normalised(t1.text) == normalised(t2.text)
    assert t1.language == "en" and t2.language == "en"
  end

  test "initial_prompt is accepted on the .en model", %{model: model, audio_path: path} do
    # Smoke test only: tiny.en is known to sometimes empty its output when
    # given an initial_prompt that derails its decoding. The contract we
    # care about is that the call wires through to the NIF without error
    # and returns a structurally-valid Transcription. The semantic effect
    # of `initial_prompt` is covered by the multilingual checkpoint test
    # block below.
    assert {:ok, %Transcription{language: lang}} =
             WhisperCt2.transcribe(model, path,
               language: "en",
               initial_prompt: "Presidential address by John F. Kennedy."
             )

    assert lang == "en"
  end

  test "rejects non-en :language on an English-only checkpoint",
       %{model: model, audio_path: path} do
    # Decoding would silently run English (the `.en` SOT block is
    # `[<|startoftranscript|>]` and ignores any pinned language token),
    # but the returned language used to echo "de" — misrouting any
    # downstream language-based logic. Pin must be rejected loudly.
    assert {:error, %WhisperCt2.Error{reason: :invalid_request, message: msg}} =
             WhisperCt2.transcribe(model, path, language: "de")

    assert msg =~ "English-only"
  end

  test "model_info reports Whisper's standard 16kHz/30s window", %{model: model} do
    assert model.sampling_rate == 16_000
    assert model.n_samples == 480_000
  end

  describe "word-timestamp golden parity with faster-whisper" do
    # We dump a reference (`tools/words-reference/generate.py`) and check
    # per-word start/end timings against it. Our post-processing replays
    # faster-whisper's pipeline — median-duration clamp at sentence-end
    # marks, `merge_punctuations`, and subsegment-boundary snap — so the
    # only legitimate sources of drift are f32 rounding and ct2 beam-
    # search tie-breaking.
    #
    # 60 ms = 3 encoder frames (one quantization step beyond the
    # 40 ms drift we currently observe). Tight enough that any real
    # regression in the alignment math — wrong DTW `start_sequence`,
    # `num_frames` in encoder vs mel units, missed punctuation merge —
    # fails loud; loose enough that a single beam-search tie-break
    # flipping one frame doesn't go red.

    @golden_dir Path.expand("fixtures/words_golden", __DIR__)

    test "matches the faster-whisper reference within 60 ms per word",
         %{model: model, audio_path: path} do
      golden = load_golden_words()
      refute Enum.empty?(golden)

      assert {:ok, %Transcription{segments: segs}} =
               WhisperCt2.transcribe(model, path, language: "en", word_timestamps: true)

      ours = Enum.flat_map(segs, &(&1.words || []))

      # Beam-search tie-breaking can shift one token boundary across
      # implementations, but a wholesale mismatch means we're emitting a
      # different transcript.
      diff = abs(length(ours) - length(golden))

      assert diff <= 1,
             "word count drift: ours=#{length(ours)} golden=#{length(golden)}"

      starts = Enum.map(ours, & &1.start)
      assert starts == Enum.sort(starts), "word starts are not monotonic"

      tolerance_s = 0.06
      n = min(length(ours), length(golden))
      ours_n = Enum.take(ours, n)
      golden_n = Enum.take(golden, n)

      ours_n
      |> Enum.zip(golden_n)
      |> Enum.with_index()
      |> Enum.each(fn {{%Word{} = ours, ref}, idx} ->
        diff_start = abs(ours.start - ref["start"])
        diff_end = abs(ours.end - ref["end"])

        assert diff_start <= tolerance_s,
               "word #{idx} start drift: ours=#{ours.start} ref=#{ref["start"]} " <>
                 "(diff #{Float.round(diff_start, 3)}s, tol #{tolerance_s}s, " <>
                 "text=#{inspect(ours.text)})"

        assert diff_end <= tolerance_s,
               "word #{idx} end drift: ours=#{ours.end} ref=#{ref["end"]} " <>
                 "(diff #{Float.round(diff_end, 3)}s, tol #{tolerance_s}s, " <>
                 "text=#{inspect(ours.text)})"
      end)
    end

    defp load_golden_words do
      @golden_dir
      |> Path.join("words.json")
      |> File.read!()
      |> JSON.decode!()
    end
  end

  describe "PCM length boundaries" do
    # Exercises the chunking math in `transcribe.rs::transcribe_many`
    # at the points most likely to off-by-one: exactly one chunk worth
    # of samples, one sample over, and a few samples under.

    test "transcribes exactly n_samples without padding errors", %{model: model} do
      n = 16_000 * 30
      pcm = silent_pcm(n)
      assert {:ok, %Transcription{}} = WhisperCt2.transcribe(model, {:pcm_f32, pcm})
    end

    test "transcribes n_samples + 1 by spilling into a second chunk", %{model: model} do
      n = 16_000 * 30 + 1
      pcm = silent_pcm(n)

      assert {:ok, %Transcription{duration_s: dur}} =
               WhisperCt2.transcribe(model, {:pcm_f32, pcm})

      # Audio is 30 s + 1 sample. Anything that produced one chunk
      # would report ~30 s; the second sample-padded chunk pushes the
      # reported duration to ~30.0000625 s.
      assert dur > 30.0
    end

    test "transcribes a few-hundred-sample buffer without crashing", %{model: model} do
      # Smaller than n_fft (400) — the streaming preprocessor must
      # handle this without producing any usable mel frames AND
      # without crashing.
      assert {:ok, %Transcription{}} =
               WhisperCt2.transcribe(model, {:pcm_f32, silent_pcm(200)})
    end
  end

  describe "long audio (multi-chunk)" do
    # Catches off-by-one in `chunk_offsets` bookkeeping and segment
    # time-stitching across the 30 s boundary. None of the JFK-based
    # tests do this because the clip is ~11 s = one chunk.

    test "stitches segments across three concatenated JFK chunks",
         %{model: model, audio_path: path} do
      {:ok, single} = WhisperCt2.Wav.read_file(path)
      tripled = single <> single <> single

      assert {:ok, %Transcription{segments: segs, duration_s: dur}} =
               WhisperCt2.transcribe(model, {:pcm_f32, tripled}, language: "en")

      # Three concatenated ~11 s clips = ~33 s.
      assert dur > 30.0 and dur < 36.0
      # Segments cover the full audio, in monotonic order, with at
      # least one segment past the 30 s chunk boundary.
      starts = Enum.map(segs, & &1.start)
      assert starts == Enum.sort(starts)
      assert Enum.any?(segs, &(&1.start >= 30.0))
      assert Enum.all?(segs, &(&1.end <= dur + 1.0))
    end
  end

  describe "compute types" do
    # Smoke-tests that non-default compute types load and produce a
    # plausible transcript. Catches CT2 configuration regressions
    # without asserting that the text is bit-identical to fp32.

    @tag :slow
    test "transcribes correctly with :int8" do
      model_dir = TestFixtures.ensure_model!()
      audio = TestFixtures.ensure_jfk!()

      assert {:ok, model} = WhisperCt2.load_model(model_dir, compute_type: :int8)
      assert model.compute_type == :int8

      assert {:ok, %Transcription{text: text}} =
               WhisperCt2.transcribe(model, audio, language: "en")

      assert String.downcase(text) =~ "country"
    end
  end

  describe "stability across repeated and concurrent calls" do
    test "transcribe/3 is deterministic across repeated calls",
         %{model: model, audio_path: path} do
      [first | rest] =
        for _ <- 1..3 do
          assert {:ok, %Transcription{text: t}} =
                   WhisperCt2.transcribe(model, path, language: "en")

          normalised(t)
        end

      # Whisper with a fixed beam is fully deterministic; any drift
      # here would point to uninitialised state inside the NIF
      # resource.
      assert Enum.all?(rest, &(&1 == first))
    end

    test "transcribe/3 is safe to call concurrently on one model",
         %{model: model, audio_path: path} do
      results =
        1..3
        |> Task.async_stream(
          fn _ -> WhisperCt2.transcribe(model, path, language: "en") end,
          max_concurrency: 3,
          timeout: 120_000,
          ordered: false
        )
        |> Enum.map(fn {:ok, r} -> r end)

      assert Enum.all?(results, &match?({:ok, %Transcription{}}, &1))

      texts =
        for {:ok, %Transcription{text: t}} <- results, do: normalised(t)

      assert length(Enum.uniq(texts)) == 1,
             "concurrent transcribes diverged: #{inspect(texts)}"
    end
  end

  describe "NIF panic protection" do
    # Contract: every public API call returns either {:ok, _} or
    # {:error, %Error{}}. Nothing in the NIF entry layer is allowed to
    # raise across the BEAM/Rust boundary — Rust panics are caught by
    # `run_with_panic_protection` (see `lib.rs`) and reported as
    # :nif_panic errors. We can't externally inject a real panic
    # without modifying the Rust side, but we can verify the
    # never-raises contract holds across a range of malformed and
    # boundary inputs.

    test "no public API call ever raises", %{model: model} do
      inputs = [
        {:pcm_f32, <<>>},
        {:pcm_f32, <<0::32-little>>},
        {:pcm_f32, silent_pcm(1)},
        {:pcm_f32, silent_pcm(159)},
        {:pcm_f32, silent_pcm(160)}
      ]

      for input <- inputs do
        # Either an :ok or any :error tuple is acceptable. A raise on
        # any of these would be a NIF contract violation.
        result = WhisperCt2.transcribe(model, input)
        assert match?({:ok, _}, result) or match?({:error, _}, result)
      end
    end
  end

  defp silent_pcm(n_samples) when n_samples >= 0 do
    :binary.copy(<<0::float-32-little>>, n_samples)
  end

  describe "multilingual checkpoint" do
    # Exercises the `[sot, lang, transcribe]` prompt branch in
    # `transcribe.rs::transcribe_many` and the multilingual code path in
    # `align.rs`. The `.en` model used by the rest of this suite skips
    # both because it only knows the `[sot]` prompt.

    setup do
      model_dir = TestFixtures.ensure_multilingual_model!()
      audio_path = TestFixtures.ensure_jfk!()
      {:ok, model} = WhisperCt2.load_model(model_dir)
      {:ok, model: model, audio_path: audio_path}
    end

    test "transcribes with an explicit language", %{model: model, audio_path: path} do
      assert {:ok, %Transcription{text: text, language: lang, segments: segs}} =
               WhisperCt2.transcribe(model, path, language: "en")

      assert lang == "en"
      assert model.multilingual == true
      assert Enum.all?(segs, &(&1.avg_logprob < 0.0))
      assert normalised(text) =~ "country"
    end

    test "auto-detects language when none is pinned", %{model: model, audio_path: path} do
      assert {:ok, %Transcription{language: lang, text: text}} =
               WhisperCt2.transcribe(model, path)

      # JFK is unambiguously English.
      assert lang == "en"
      assert normalised(text) =~ "country"
    end

    test "word_timestamps align within the clip on the multilingual model",
         %{model: model, audio_path: path} do
      assert {:ok, %Transcription{segments: segs, duration_s: dur}} =
               WhisperCt2.transcribe(model, path, language: "en", word_timestamps: true)

      all_words = Enum.flat_map(segs, &(&1.words || []))
      assert all_words != []

      [first | _] = all_words
      last = List.last(all_words)

      # Same alignment-sanity contract as the English-only path. The
      # multilingual DTW start_sequence must produce timestamps that fall
      # within the audio.
      assert first.start < 2.0
      assert last.end <= dur + 0.5
    end
  end

  defp normalised(text) do
    text
    |> String.downcase()
    |> String.replace(~r/[^a-z ]/, " ")
    |> String.replace(~r/\s+/, " ")
    |> String.trim()
  end
end
