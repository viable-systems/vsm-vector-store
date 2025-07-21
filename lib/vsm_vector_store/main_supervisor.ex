defmodule VSMVectorStore.MainSupervisor do
  @moduledoc """
  Main supervisor for VSM Vector Store following hierarchical VSM supervision pattern.
  
  Manages four key subsystems:
  - Storage (HNSW index, vector operations)
  - Indexing (K-means clustering, quantization)
  - ML (pattern recognition, anomaly detection)
  - Telemetry (monitoring and metrics)
  """
  
  use Supervisor
  require Logger
  
  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end
  
  @impl true
  def init(_init_arg) do
    Logger.info("Initializing VSM Vector Store Main Supervisor")
    
    children = [
      # Storage subsystem - manages vector storage and HNSW index
      {VSMVectorStore.Storage.Supervisor, []},
      
      # Indexing subsystem - manages clustering and quantization
      {VSMVectorStore.Indexing.Supervisor, []},
      
      # ML subsystem - manages pattern recognition and anomaly detection
      {VSMVectorStore.ML.Supervisor, []},
      
      # Core vector store interface
      {VSMVectorStore.Core, []}
    ]
    
    Supervisor.init(children, strategy: :one_for_one)
  end
end