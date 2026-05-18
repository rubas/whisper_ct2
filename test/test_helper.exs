# Integration tests are opt-in; run them with `mix test --include integration`.
ExUnit.configure(exclude: [:integration])
ExUnit.start()
