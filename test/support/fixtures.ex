defmodule WhisperCt2.TestFixtures do
  @moduledoc """
  Downloads and caches the integration-test artefacts (Whisper tiny model and
  the canonical `jfk.wav` clip) under `test/fixtures/`.

  Files are fetched once and reused across runs. Set `WHISPER_CT2_REFRESH=1`
  to force re-download.
  """

  @model_files [
    {"config.json", "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/config.json"},
    {"model.bin", "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/model.bin"},
    {"tokenizer.json", "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/tokenizer.json"},
    {"vocabulary.txt", "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/vocabulary.txt"},
    {"preprocessor_config.json", "https://huggingface.co/openai/whisper-tiny.en/resolve/main/preprocessor_config.json"}
  ]

  @jfk_url "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"

  @fixtures_root Path.expand("../fixtures", __DIR__)
  @model_dir Path.join(@fixtures_root, "faster-whisper-tiny.en")
  @jfk_path Path.join(@fixtures_root, "jfk.wav")

  @spec model_dir() :: Path.t()
  def model_dir, do: @model_dir

  @spec jfk_wav() :: Path.t()
  def jfk_wav, do: @jfk_path

  @spec ensure_model!() :: Path.t()
  def ensure_model! do
    File.mkdir_p!(@model_dir)
    Enum.each(@model_files, fn {name, url} -> ensure_file!(Path.join(@model_dir, name), url) end)
    @model_dir
  end

  @spec ensure_jfk!() :: Path.t()
  def ensure_jfk! do
    File.mkdir_p!(@fixtures_root)
    ensure_file!(@jfk_path, @jfk_url)
    @jfk_path
  end

  defp ensure_file!(path, url) do
    if System.get_env("WHISPER_CT2_REFRESH") == "1" do
      File.rm_rf!(path)
    end

    if File.regular?(path) and File.stat!(path).size > 0 do
      :ok
    else
      download!(url, path)
    end
  end

  defp download!(url, dest) do
    tmp = dest <> ".part"
    File.rm_rf!(tmp)

    args = ["-fL", "--retry", "3", "--retry-delay", "2", "-o", tmp, url]

    case System.cmd("curl", args, stderr_to_stdout: true) do
      {_out, 0} ->
        File.rename!(tmp, dest)

      {out, code} ->
        File.rm_rf!(tmp)
        raise "failed to download #{url} (exit #{code}): #{out}"
    end
  end
end
