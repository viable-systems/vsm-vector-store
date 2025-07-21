defmodule VSMVectorStore.Application do
  @moduledoc """
  The VSM Vector Store OTP Application.
  
  Follows Viable System Model architecture with hierarchical supervision
  and specialized subsystems for vector storage, indexing, and ML operations.
  """
  
  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    Logger.info("Starting VSM Vector Store Application...")
    
    children = [
      # Shared infrastructure (following VSM pattern)
      {Registry, keys: :unique, name: VSMVectorStore.Registry},
      {DynamicSupervisor, name: VSMVectorStore.DynamicSupervisor, strategy: :one_for_one},
      
      # Storage subsystem (System 1 - Operations)
      VSMVectorStore.Storage.Supervisor,
      
      # Indexing subsystem (System 2 - Coordination) 
      VSMVectorStore.Indexing.Supervisor,
      
      # ML subsystem (System 4 - Intelligence)
      VSMVectorStore.ML.Supervisor,
      
      # Telemetry and monitoring (System 3 - Control)
      VSMVectorStore.TelemetryReporter
    ]

    opts = [strategy: :one_for_one, name: VSMVectorStore.Supervisor]
    
    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        Logger.info("VSM Vector Store Application started successfully")
        setup_telemetry()
        {:ok, pid}
        
      {:error, reason} ->
        Logger.error("Failed to start VSM Vector Store Application: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @impl true
  def stop(_state) do
    Logger.info("Stopping VSM Vector Store Application...")
    :ok
  end
  
  # Private functions
  
  defp setup_telemetry do
    # Setup telemetry handlers for monitoring vector operations
    :telemetry.attach_many(
      "vsm-vector-store-telemetry",
      [
        [:vsm_vector_store, :vector, :insert],
        [:vsm_vector_store, :vector, :search],
        [:vsm_vector_store, :ml, :cluster],
        [:vsm_vector_store, :ml, :anomaly_detection],
        [:vsm_vector_store, :storage, :operation]
      ],
      &VSMVectorStore.TelemetryReporter.handle_event/4,
      nil
    )
    
    Logger.debug("Telemetry handlers attached successfully")
  end
end