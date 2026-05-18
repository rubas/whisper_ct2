defmodule WhisperCt2.ErrorTest do
  @moduledoc """
  Tests conversion of native error payloads into `WhisperCt2.Error`.

  These tests cover local mapping and exception formatting, not native NIF
  error generation.
  """

  use ExUnit.Case, async: true

  alias WhisperCt2.Error

  test "from_native maps known types to atoms" do
    payload = %{type: "load_error", message: "boom", details: %{"path" => "/x"}}

    assert %Error{reason: :load_error, message: "boom", details: %{"path" => "/x"}} =
             Error.from_native(payload)
  end

  test "from_native works without details" do
    payload = %{type: "invalid_request", message: "bad"}

    assert %Error{reason: :invalid_request, message: "bad", details: %{}} =
             Error.from_native(payload)
  end

  test "from_native falls back to :native_error for unknown types" do
    payload = %{type: "wat", message: "?", details: %{}}
    assert %Error{reason: :native_error} = Error.from_native(payload)
  end

  test "from_native tolerates garbage" do
    assert %Error{reason: :native_error} = Error.from_native(:nope)
  end

  test "Exception protocol" do
    err = Error.new(:invalid_request, "bad input")
    assert Exception.message(err) == "invalid_request: bad input"
  end

  # The NIF produces map payloads using rustler's NifMap derive on
  # `NativeError { r#type, message, details }`, which encodes as atom-keyed
  # Elixir maps. Pin every type string the NIF can return so a future
  # rename on either side fails loudly here instead of silently degrading
  # every error to `:native_error`.
  describe "from_native NIF contract" do
    for {type_string, reason} <- [
          {"invalid_request", :invalid_request},
          {"load_error", :load_error},
          {"inference_error", :inference_error},
          {"runtime_error", :runtime_error},
          {"nif_panic", :nif_panic}
        ] do
      test "maps #{type_string} -> #{reason}" do
        payload = %{type: unquote(type_string), message: "x", details: %{}}
        assert %Error{reason: unquote(reason)} = Error.from_native(payload)
      end
    end

    test "string-keyed payloads fall through to :native_error" do
      payload = %{"type" => "load_error", "message" => "boom"}
      assert %Error{reason: :native_error} = Error.from_native(payload)
    end
  end
end
