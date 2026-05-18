defmodule WhisperCt2Test do
  @moduledoc """
  Tests the public `WhisperCt2` API validation and error contracts.

  These tests use synthetic inputs for pre-NIF validation paths and do not
  prove real model inference; integration coverage lives in
  `WhisperCt2.IntegrationTest`.
  """

  use ExUnit.Case, async: true

  alias WhisperCt2.{Error, Model}

  describe "load_model/1" do
    test "returns invalid_request for a non-existent directory" do
      assert {:error, %Error{reason: :invalid_request}} = WhisperCt2.load_model("/no/such/dir")
    end

    test "returns invalid_request for a file path" do
      path = Path.join(System.tmp_dir!(), "whisper_ct2_not_a_model.txt")
      File.write!(path, "hello")
      on_exit(fn -> File.rm(path) end)

      assert {:error, %Error{reason: :invalid_request}} = WhisperCt2.load_model(path)
    end

    test "returns load_error for a directory missing model files" do
      path =
        Path.join(System.tmp_dir!(), "whisper_ct2_empty_#{System.unique_integer([:positive])}")

      File.mkdir_p!(path)
      on_exit(fn -> File.rm_rf!(path) end)

      assert {:error, %Error{reason: :load_error}} = WhisperCt2.load_model(path)
    end
  end

  describe "transcribe/3 input validation" do
    test "rejects PCM binary with non-f32-aligned length" do
      model = fake_model()

      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCt2.transcribe(model, {:pcm_f32, <<1, 2, 3>>})

      assert msg =~ "multiple of 4"
    end

    test "rejects unsupported audio shapes" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.transcribe(fake_model(), 42)
    end

    test "rejects a bare binary that is not a .wav path" do
      # Used to silently become garbage PCM; now must fail with a clear
      # error so typo'd paths are caught at the boundary.
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCt2.transcribe(fake_model(), <<0, 1, 2, 3>>)

      assert msg =~ "does not exist" or msg =~ ".wav"
    end

    test "rejects a non-.wav path that exists on disk" do
      path = Path.join(System.tmp_dir!(), "whisper_ct2_not_wav.mp3")
      File.write!(path, "id3 garbage")
      on_exit(fn -> File.rm(path) end)

      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCt2.transcribe(fake_model(), path)

      assert msg =~ ".wav"
    end
  end

  describe "transcribe_batch/3 input validation" do
    test "rejects non-list audios" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.transcribe_batch(fake_model(), "not-a-list")
    end

    test "propagates a bad audio entry's error" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.transcribe_batch(fake_model(), [
                 {:pcm_f32, <<0, 0, 0, 0>>},
                 {:pcm_f32, <<1, 2, 3>>}
               ])
    end

    test "short-circuits an empty list without touching the NIF" do
      # Contract: empty batch is a no-op, not an error. Critically, this
      # must not depend on a real loaded model.
      assert {:ok, []} = WhisperCt2.transcribe_batch(fake_model(), [])
    end
  end

  describe "available_devices/0" do
    test "returns an {:ok, info} tuple" do
      assert {:ok, %{cpu: cpu, cuda: cuda, cuda_supported: cuda_supported}} =
               WhisperCt2.available_devices()

      assert is_integer(cpu) and cpu >= 0
      assert is_integer(cuda) and cuda >= 0
      assert is_boolean(cuda_supported)
    end
  end

  describe "load_model/2 option validation" do
    test "rejects unknown option keys" do
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCt2.load_model("/tmp", boost: true)

      assert msg =~ "unknown option"
      assert msg =~ ":boost"
    end

    test "rejects invalid device atom" do
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCt2.load_model("/tmp", device: :tpu)

      assert msg =~ ":device"
    end

    test "rejects invalid compute_type" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.load_model("/tmp", compute_type: :nibble)
    end

    test "rejects empty device_indices" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.load_model("/tmp", device_indices: [])
    end

    test "rejects negative num_threads_per_replica" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.load_model("/tmp", num_threads_per_replica: -1)
    end

    test "accepts max_queued_batches and cpu_core_offset" do
      assert {:error, %Error{reason: reason}} =
               WhisperCt2.load_model("/no/such/dir",
                 max_queued_batches: 2,
                 cpu_core_offset: 4
               )

      assert reason == :invalid_request
    end
  end

  describe "transcribe/3 option validation" do
    test "rejects unknown option keys" do
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCt2.transcribe(fake_model(), {:pcm_f32, <<>>}, foo: 1)

      assert msg =~ "unknown option"
    end

    test "rejects beam_size of zero" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.transcribe(fake_model(), {:pcm_f32, <<>>}, beam_size: 0)
    end

    test "rejects non-integer suppress_tokens entries" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.transcribe(fake_model(), {:pcm_f32, <<>>}, suppress_tokens: [1, "x"])
    end

    test "rejects empty language string" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.transcribe(fake_model(), {:pcm_f32, <<>>}, language: "  ")
    end

    test "rejects empty :initial_prompt and :prefix strings" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.transcribe(fake_model(), {:pcm_f32, <<>>}, initial_prompt: "")

      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.transcribe(fake_model(), {:pcm_f32, <<>>}, prefix: "")
    end

    test "rejects non-boolean :word_timestamps" do
      assert {:error, %Error{reason: :invalid_request}} =
               WhisperCt2.transcribe(fake_model(), {:pcm_f32, <<>>}, word_timestamps: "yes")
    end

    test "rejects removed return_* options" do
      assert {:error, %Error{reason: :invalid_request, message: msg}} =
               WhisperCt2.transcribe(fake_model(), {:pcm_f32, <<>>}, return_scores: true)

      assert msg =~ "unknown option"
    end
  end

  # Synthetic model for validation paths that fail before inference.
  defp fake_model do
    %Model{
      ref: make_ref(),
      path: "/dev/null",
      sampling_rate: 16_000,
      n_samples: 480_000,
      multilingual: false,
      device: :cpu,
      compute_type: :default
    }
  end
end
