defmodule VSMVectorStore.MixProject do
  use Mix.Project

  def project do
    [
      app: :vsm_vector_store,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.detail": :test,
        "coveralls.post": :test,
        "coveralls.html": :test
      ],
      aliases: aliases()
    ]
  end

  def application do
    [
      extra_applications: [:logger, :runtime_tools],
      mod: {VSMVectorStore.Application, []}
    ]
  end

  defp deps do
    [
      # Core dependencies
      {:telemetry, "~> 1.0"},
      {:telemetry_metrics, "~> 1.0"},
      {:jason, "~> 1.0"},
      
      # VSM integration (assuming these are published packages)
      # {:vsm_core, "~> 0.1.0"},
      # {:vsm_telemetry, "~> 0.1.0"},
      
      # Development and testing
      {:ex_doc, "~> 0.27", only: :dev, runtime: false},
      {:excoveralls, "~> 0.10", only: :test},
      {:benchee, "~> 1.0", only: [:dev, :test]},
      {:stream_data, "~> 0.5", only: :test},
      {:dialyxir, "~> 1.0", only: [:dev], runtime: false},
      {:credo, "~> 1.6", only: [:dev, :test], runtime: false}
    ]
  end

  defp docs do
    [
      main: "VSMVectorStore",
      source_url: "https://github.com/viable-systems/vsm-vector-store",
      homepage_url: "https://github.com/viable-systems/vsm-vector-store",
      extras: ["README.md"]
    ]
  end

  defp aliases do
    [
      "test.all": ["test", "test --include slow"],
      "test.bench": ["run benchmarks/run_all.exs"],
      quality: ["format", "credo --strict", "dialyzer"]
    ]
  end
end