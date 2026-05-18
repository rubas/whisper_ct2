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
end
