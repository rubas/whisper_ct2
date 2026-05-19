defmodule WhisperCt2.Native do
  @moduledoc """
  Low-level Rustler bindings to `ct2rs::Whisper`.

  This module is private to the library. Use `WhisperCt2` for the public API.
  Stub names must match the Rust NIF symbols verbatim (Rustler verifies arity
  at module load time); user-friendly wrappers live below them.
  """

  @cargo_features_env System.get_env("WHISPER_CT2_FEATURES", "")
  @cargo_features_raw Application.compile_env(:whisper_ct2, :cargo_features, @cargo_features_env)
  @cargo_features String.split(@cargo_features_raw, ~r/[,\s]+/, trim: true)

  @version Mix.Project.config()[:version]

  use RustlerPrecompiled,
    otp_app: :whisper_ct2,
    crate: "whisper_ct2_native",
    base_url: "https://github.com/rubas/whisper_ct2/releases/download/v#{@version}",
    version: @version,
    # Opt-in source build; unconditional in this repo via `config/config.exs`.
    force_build:
      System.get_env("WHISPER_CT2_BUILD") in ["1", "true"] or
        Application.compile_env(:rustler_precompiled, [:force_build, :whisper_ct2], false),
    nif_versions: ["2.17"],
    targets: ~w(
      aarch64-apple-darwin
      x86_64-unknown-linux-gnu
      aarch64-unknown-linux-gnu
    ),
    # Optional variants on x86_64 Linux. Opt in via `WHISPER_CT2_VARIANT`:
    #   - `mkl`  : Intel-tuned MKL build (CPU only; CUDA still loaded
    #              dynamically on CUDA hosts).
    #   - `rocm` : AMD ROCm/HIP GPU build. Requires ROCm 7.x at runtime
    #              (libamdhip64, libhipblas). Mutually exclusive with the
    #              CUDA path bundled in the default artefact.
    variants: %{
      "x86_64-unknown-linux-gnu" => [
        mkl: fn -> System.get_env("WHISPER_CT2_VARIANT") == "mkl" end,
        rocm: fn -> System.get_env("WHISPER_CT2_VARIANT") == "rocm" end
      ]
    },
    features: @cargo_features

  @doc "CPU/CUDA device counts and the build's CUDA-support flag."
  @spec available_devices() :: {:ok, map()} | {:error, map()}
  def available_devices, do: nif_available_devices()

  @doc "Loads a CT2 Whisper model directory."
  @spec load_model(String.t(), map()) :: {:ok, reference()} | {:error, map()}
  def load_model(path, opts), do: nif_load_model(path, opts)

  @doc "Returns model metadata."
  @spec model_info(reference()) :: {:ok, map()} | {:error, map()}
  def model_info(model), do: nif_model_info(model)

  @doc """
  Runs Whisper on a buffer of PCM samples.

  `samples_bin` is a binary of little-endian `f32` samples (mono, 16 kHz).
  Audio longer than the 30 s Whisper window is chunked and batched
  internally; the encoder runs once across all chunks. Returns a structured
  transcription map (`%{language, duration_s, segments: [...]}`).
  """
  @spec transcribe(reference(), binary(), map()) :: {:ok, map()} | {:error, map()}
  def transcribe(model, samples_bin, opts), do: nif_transcribe(model, samples_bin, opts)

  @doc """
  Runs Whisper on a list of PCM sample buffers in one batched `generate`
  call. Returns a list of structured transcription maps in input order.
  """
  @spec transcribe_batch(reference(), [binary()], map()) :: {:ok, [map()]} | {:error, map()}
  def transcribe_batch(model, samples_bins, opts),
    do: nif_transcribe_batch(model, samples_bins, opts)

  defp nif_available_devices, do: :erlang.nif_error(:nif_not_loaded)
  defp nif_load_model(_path, _opts), do: :erlang.nif_error(:nif_not_loaded)
  defp nif_model_info(_model), do: :erlang.nif_error(:nif_not_loaded)
  defp nif_transcribe(_model, _samples_bin, _opts), do: :erlang.nif_error(:nif_not_loaded)
  defp nif_transcribe_batch(_model, _samples_bins, _opts), do: :erlang.nif_error(:nif_not_loaded)
end
