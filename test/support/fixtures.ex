defmodule WhisperCt2.TestFixtures do
  @moduledoc """
  Downloads and caches the integration-test artefacts (Whisper tiny model and
  the canonical `jfk.wav` clip) under `test/fixtures/`.

  Files are fetched once and reused across runs. Set `WHISPER_CT2_REFRESH=1`
  to force re-download.
  """

  @en_model_files [
    {"config.json", "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/config.json"},
    {"model.bin", "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/model.bin"},
    {"tokenizer.json", "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/tokenizer.json"},
    {"vocabulary.txt", "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/vocabulary.txt"},
    {"preprocessor_config.json", "https://huggingface.co/openai/whisper-tiny.en/resolve/main/preprocessor_config.json"}
  ]

  @multi_model_files [
    {"config.json", "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/config.json"},
    {"model.bin", "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/model.bin"},
    {"tokenizer.json", "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/tokenizer.json"},
    {"vocabulary.txt", "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/vocabulary.txt"},
    {"preprocessor_config.json", "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json"}
  ]

  @jfk_url "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
  # SHA-256 of the canonical jfk.wav clip. Mirrored from
  # `test/fixtures/words_golden/meta.json` so a poisoned/truncated download
  # never silently passes the cache check and lets the golden assertions
  # drift on garbage audio.
  @jfk_sha256 "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"

  @fixtures_root Path.expand("../fixtures", __DIR__)
  @en_model_dir Path.join(@fixtures_root, "faster-whisper-tiny.en")
  @multi_model_dir Path.join(@fixtures_root, "faster-whisper-tiny")
  @jfk_path Path.join(@fixtures_root, "jfk.wav")

  @spec model_dir() :: Path.t()
  def model_dir, do: @en_model_dir

  @spec multilingual_model_dir() :: Path.t()
  def multilingual_model_dir, do: @multi_model_dir

  @spec jfk_wav() :: Path.t()
  def jfk_wav, do: @jfk_path

  @spec ensure_model!() :: Path.t()
  def ensure_model! do
    File.mkdir_p!(@en_model_dir)
    Enum.each(@en_model_files, fn {name, url} -> ensure_file!(Path.join(@en_model_dir, name), url) end)
    @en_model_dir
  end

  @spec ensure_multilingual_model!() :: Path.t()
  def ensure_multilingual_model! do
    File.mkdir_p!(@multi_model_dir)
    Enum.each(@multi_model_files, fn {name, url} -> ensure_file!(Path.join(@multi_model_dir, name), url) end)
    @multi_model_dir
  end

  @spec ensure_jfk!() :: Path.t()
  def ensure_jfk! do
    File.mkdir_p!(@fixtures_root)
    ensure_file!(@jfk_path, @jfk_url, sha256: @jfk_sha256)
    @jfk_path
  end

  defp ensure_file!(path, url, opts \\ []) do
    if System.get_env("WHISPER_CT2_REFRESH") == "1" do
      File.rm_rf!(path)
    end

    if file_present_and_valid?(path, opts) do
      :ok
    else
      File.rm_rf!(path)
      download!(url, path)
      ensure_sha!(path, opts[:sha256])
    end
  end

  defp file_present_and_valid?(path, opts) do
    cond do
      not File.regular?(path) -> false
      File.stat!(path).size == 0 -> false
      true -> sha_matches?(path, opts[:sha256])
    end
  end

  defp sha_matches?(_path, nil), do: true

  defp sha_matches?(path, expected) when is_binary(expected) do
    actual_sha256(path) == expected
  end

  defp ensure_sha!(_path, nil), do: :ok

  defp ensure_sha!(path, expected) when is_binary(expected) do
    actual = actual_sha256(path)

    if actual != expected do
      File.rm_rf!(path)

      raise """
      fixture SHA256 mismatch for #{path}
        expected: #{expected}
        actual:   #{actual}
      """
    end

    :ok
  end

  defp actual_sha256(path) do
    # Switch to a streaming hash if we ever pin model.bin (~75 MB).
    :sha256
    |> :crypto.hash(File.read!(path))
    |> Base.encode16(case: :lower)
  end

  defp download!(url, dest) do
    tmp = dest <> ".part"
    File.rm_rf!(tmp)

    args = ["-fL", "--retry", "3", "--retry-delay", "2", "-o", tmp, url]
    # `env: []` keeps curl from inheriting secrets from the parent
    # process; only PATH (needed to resolve curl itself) is forwarded.
    env = [{"PATH", System.get_env("PATH") || ""}]

    case System.cmd("curl", args, stderr_to_stdout: true, env: env) do
      {_out, 0} ->
        File.rename!(tmp, dest)

      {out, code} ->
        File.rm_rf!(tmp)
        raise "failed to download #{url} (exit #{code}): #{out}"
    end
  end
end
