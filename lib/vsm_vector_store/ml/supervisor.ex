defmodule VSMVectorStore.ML.Supervisor do
  @moduledoc """
  ML subsystem supervisor - manages pattern recognition and anomaly detection.
  """
  
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    children = [
      # Anomaly detection service
      VSMVectorStore.ML.AnomalyDetection,
      
      # Pattern recognition service
      VSMVectorStore.ML.PatternRecognition
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end