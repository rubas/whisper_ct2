defmodule WhisperCt2.Native do
  @moduledoc """
  Low-level Rustler bindings to `ct2rs::Whisper`.

  This module is private to the library. Use `WhisperCt2` for the public API.
  Stub names must match the Rust NIF symbols verbatim (Rustler verifies arity
  at module load time); user-friendly wrappers live below them.
  """

  @cargo_features_env System.get_env("WHISPER_CT2_FEATURES", "")
  @cargo_features Application.compile_env(:whisper_ct2, :cargo_features, @cargo_features_env)
                  |> String.split(~r/[,\s]+/, trim: true)

  @version Mix.Project.config()[:version]

  use RustlerPrecompiled,
    otp_app: :whisper_ct2,
    crate: "whisper_ct2_native",
    base_url: "https://github.com/rubas/whisper_ct2/releases/download/v#{@version}",
    version: @version,
    # `force_build` is opt-in for consumers (precompiled is the default path)
    # but unconditional for this repo via `config/config.exs`. The env-var
    # path lets contributors of a consuming app rebuild on demand.
    force_build:
      System.get_env("WHISPER_CT2_BUILD") in ["1", "true"] or
        Application.compile_env(:rustler_precompiled, [:force_build, :whisper_ct2], false),
    nif_versions: ["2.17"],
    targets: ~w(
      aarch64-apple-darwin
      x86_64-unknown-linux-gnu
      aarch64-unknown-linux-gnu
    ),
    # On x86_64 Linux we ship two artefacts: the default oneDNN build
    # (Intel + AMD) and an Intel-tuned MKL build. Consumers opt into MKL
    # by setting `WHISPER_CT2_VARIANT=mkl` in their environment before
    # `mix deps.compile`.
    variants: %{
      "x86_64-unknown-linux-gnu" => [
        mkl: fn -> System.get_env("WHISPER_CT2_VARIANT") == "mkl" end
      ]
    },
    features: @cargo_features

  @doc "Reports CUDA support of this build plus visible CPU/CUDA device counts."
  @spec available_devices() :: {:ok, map()} | {:error, map()}
  def available_devices, do: nif_available_devices()

  @doc "Loads a CTranslate2-converted Whisper model from a directory."
  @spec load_model(String.t(), map()) :: {:ok, reference()} | {:error, map()}
  def load_model(path, opts), do: nif_load_model(path, opts)

  @doc "Returns model metadata (sampling rate, window length, multilingual flag, device, compute_type)."
  @spec model_info(reference()) :: {:ok, map()} | {:error, map()}
  def model_info(model), do: nif_model_info(model)

  @doc """
  Runs Whisper on a buffer of PCM samples.

  `samples_bin` is a binary of little-endian `f32` samples (mono, 16 kHz).
  `ct2rs::Whisper` splits anything longer than the 30 s window internally.
  """
  @spec transcribe(reference(), binary(), map()) :: {:ok, [String.t()]} | {:error, map()}
  def transcribe(model, samples_bin, opts), do: nif_transcribe(model, samples_bin, opts)

  defp nif_available_devices, do: :erlang.nif_error(:nif_not_loaded)
  defp nif_load_model(_path, _opts), do: :erlang.nif_error(:nif_not_loaded)
  defp nif_model_info(_model), do: :erlang.nif_error(:nif_not_loaded)
  defp nif_transcribe(_model, _samples_bin, _opts), do: :erlang.nif_error(:nif_not_loaded)
end
