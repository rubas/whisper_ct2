defmodule WhisperCt2.MixProject do
  use Mix.Project

  @version "0.2.0"
  @source_url "https://github.com/rubas/whisper_ct2"

  @spec project() :: keyword()
  def project do
    [
      app: :whisper_ct2,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      description: description(),
      package: package(),
      docs: docs(),
      deps: deps()
    ]
  end

  @spec application() :: keyword()
  def application do
    [extra_applications: [:logger]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  @spec docs() :: keyword()
  defp docs do
    [
      main: "readme",
      extras: ["README.md", "CHANGELOG.md", "usage-rules.md"],
      source_url: @source_url,
      source_ref: "v#{@version}",
      homepage_url: @source_url
    ]
  end

  @spec description() :: String.t()
  defp description do
    "Native Elixir bindings for Whisper inference via CTranslate2 (no Python)."
  end

  @spec package() :: keyword()
  defp package do
    [
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "CTranslate2" => "https://github.com/OpenNMT/CTranslate2",
        "ct2rs" => "https://github.com/jkawamoto/ctranslate2-rs"
      },
      files:
        ~w(lib native/whisper_ct2_native/src native/whisper_ct2_native/Cargo.toml native/whisper_ct2_native/Cargo.lock checksum-*.exs mix.exs README.md CHANGELOG.md LICENSE* usage-rules.md)
    ]
  end

  @spec deps() :: [tuple()]
  defp deps do
    [
      # `rustler_precompiled` selects a prebuilt NIF artefact at install time
      # from the GitHub release matching the package version. `rustler` is
      # only needed for source builds (`WHISPER_CT2_BUILD=1`) and during
      # release CI, so it is marked optional.
      {:rustler_precompiled, "~> 0.8"},
      {:rustler, "~> 0.37.3", optional: true},
      {:credo, "~> 1.7.18", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.40", only: :dev, runtime: false}
    ]
  end
end
