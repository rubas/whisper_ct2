defmodule WhisperCt2.Error do
  @moduledoc """
  Structured error returned from `WhisperCt2` calls.

  `reason` is one of:

  - `:invalid_request` - bad arguments (bad path, malformed audio, oversized chunk).
  - `:load_error`      - CTranslate2 could not load the model directory.
  - `:inference_error` - the Whisper model returned an error during generation.
  - `:runtime_error`   - internal NIF runtime fault (e.g. poisoned mutex).
  - `:nif_panic`       - the Rust side panicked; should never happen in practice.
  - `:native_error`    - fallback for unrecognised native error types.
  """

  @type reason ::
          :invalid_request
          | :load_error
          | :inference_error
          | :runtime_error
          | :nif_panic
          | :native_error

  @type t :: %__MODULE__{
          reason: reason(),
          message: String.t(),
          details: map()
        }

  defexception [:reason, :message, details: %{}]

  @spec new(reason(), String.t(), map()) :: t()
  def new(reason, message, details \\ %{}) do
    %__MODULE__{reason: reason, message: message, details: details}
  end

  @spec from_native(map()) :: t()
  def from_native(%{type: type, message: message} = payload) do
    new(to_reason(type), message, Map.get(payload, :details, %{}))
  end

  def from_native(other) do
    new(:native_error, "unexpected native error payload", %{raw: inspect(other)})
  end

  defp to_reason("invalid_request"), do: :invalid_request
  defp to_reason("load_error"), do: :load_error
  defp to_reason("inference_error"), do: :inference_error
  defp to_reason("runtime_error"), do: :runtime_error
  defp to_reason("nif_panic"), do: :nif_panic
  defp to_reason(_), do: :native_error

  @impl Exception
  def message(%__MODULE__{reason: reason, message: msg}), do: "#{reason}: #{msg}"
end
