defmodule VSMVectorStore.Core do
  @moduledoc """
  Core VSM Vector Store interface providing the main API for vector operations.
  
  This GenServer manages the coordination between storage, indexing, and ML subsystems
  while maintaining VSM patterns for state management and error handling.
  """
  
  use GenServer
  require Logger
  
  alias VSMVectorStore.Storage.{Manager, Space}
  alias VSMVectorStore.Storage.VectorOps
  alias VSMVectorStore.Indexing.KMeans
  alias VSMVectorStore.Indexing.Quantization
  alias VSMVectorStore.ML.PatternRecognition
  alias VSMVectorStore.ML.AnomalyDetection
  
  @type vector :: list(float())
  @type vector_id :: binary()
  @type metadata :: map()
  @type search_result :: {vector_id(), float(), metadata()}
  
  defstruct [
    :hnsw_pid,
    :vector_ops_pid,
    :kmeans_pid,
    :quantization_pid,
    :pattern_recognition_pid,
    :anomaly_detection_pid,
    dimension: 128,
    max_connections: 16,
    ef_construction: 200,
    ef_search: 50
  ]
  
  ## Public API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Inserts a vector with metadata into the vector store.
  """
  @spec insert(vector_id(), vector(), metadata()) :: :ok | {:error, term()}
  def insert(id, vector, metadata \\ %{}) do
    GenServer.call(__MODULE__, {:insert, id, vector, metadata})
  end
  
  @doc """
  Searches for similar vectors using HNSW algorithm.
  """
  @spec search(vector(), pos_integer()) :: {:ok, list(search_result())} | {:error, term()}
  def search(query_vector, k \\ 10) do
    GenServer.call(__MODULE__, {:search, query_vector, k})
  end
  
  @doc """
  Deletes a vector by ID.
  """
  @spec delete(vector_id()) :: :ok | {:error, term()}
  def delete(id) do
    GenServer.call(__MODULE__, {:delete, id})
  end
  
  @doc """
  Gets vector by ID.
  """
  @spec get(vector_id()) :: {:ok, {vector(), metadata()}} | {:error, :not_found}
  def get(id) do
    GenServer.call(__MODULE__, {:get, id})
  end
  
  @doc """
  Performs K-means clustering on stored vectors.
  """
  @spec cluster(pos_integer()) :: {:ok, list(list(vector_id()))} | {:error, term()}
  def cluster(k) do
    GenServer.call(__MODULE__, {:cluster, k}, 30_000)
  end
  
  @doc """
  Detects anomalies in stored vectors using isolation forest.
  """
  @spec detect_anomalies(float()) :: {:ok, list(vector_id())} | {:error, term()}
  def detect_anomalies(contamination \\ 0.1) do
    GenServer.call(__MODULE__, {:detect_anomalies, contamination}, 30_000)
  end
  
  ## GenServer Callbacks
  
  @impl true
  def init(opts) do
    dimension = Keyword.get(opts, :dimension, 128)
    max_connections = Keyword.get(opts, :max_connections, 16)
    ef_construction = Keyword.get(opts, :ef_construction, 200)
    ef_search = Keyword.get(opts, :ef_search, 50)
    
    state = %__MODULE__{
      dimension: dimension,
      max_connections: max_connections,
      ef_construction: ef_construction,
      ef_search: ef_search
    }
    
    {:ok, state, {:continue, :initialize_subsystems}}
  end
  
  @impl true
  def handle_continue(:initialize_subsystems, state) do
    Logger.info("Initializing VSM Vector Store Core subsystems")
    
    # Get PIDs of managed processes
    {:ok, hnsw_pid} = VSMVectorStore.Storage.HNSW.start_link([
      dimension: state.dimension,
      max_connections: state.max_connections,
      ef_construction: state.ef_construction
    ])
    
    {:ok, vector_ops_pid} = VectorOps.start_link([])
    {:ok, kmeans_pid} = KMeans.start_link([])
    {:ok, quantization_pid} = Quantization.start_link([])
    {:ok, pattern_recognition_pid} = PatternRecognition.start_link([])
    {:ok, anomaly_detection_pid} = AnomalyDetection.start_link([])
    
    updated_state = %{state |
      hnsw_pid: hnsw_pid,
      vector_ops_pid: vector_ops_pid,
      kmeans_pid: kmeans_pid,
      quantization_pid: quantization_pid,
      pattern_recognition_pid: pattern_recognition_pid,
      anomaly_detection_pid: anomaly_detection_pid
    }
    
    Logger.info("VSM Vector Store Core initialized successfully")
    {:noreply, updated_state}
  end
  
  @impl true
  def handle_call({:insert, id, vector, metadata}, _from, state) do
    start_time = System.monotonic_time()
    
    case VSMVectorStore.Storage.HNSW.insert(state.hnsw_pid, id, vector, metadata) do
      :ok ->
        duration = System.monotonic_time() - start_time
        :telemetry.execute([:vsm_vector_store, :storage, :vector, :operation], 
          %{duration: duration}, %{operation: :insert, vector_id: id})
        {:reply, :ok, state}
        
      {:error, reason} ->
        Logger.error("Failed to insert vector #{id}: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:search, query_vector, k}, _from, state) do
    start_time = System.monotonic_time()
    
    case VSMVectorStore.Storage.HNSW.search(state.hnsw_pid, query_vector, k, state.ef_search) do
      {:ok, results} ->
        duration = System.monotonic_time() - start_time
        :telemetry.execute([:vsm_vector_store, :storage, :hnsw, :search], 
          %{duration: duration, results_count: length(results)}, %{k: k})
        {:reply, {:ok, results}, state}
        
      {:error, reason} ->
        Logger.error("Search failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:delete, id}, _from, state) do
    case VSMVectorStore.Storage.HNSW.delete(state.hnsw_pid, id) do
      :ok -> {:reply, :ok, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:get, id}, _from, state) do
    case VSMVectorStore.Storage.HNSW.get(state.hnsw_pid, id) do
      {:ok, {vector, metadata}} -> {:reply, {:ok, {vector, metadata}}, state}
      {:error, :not_found} -> {:reply, {:error, :not_found}, state}
    end
  end
  
  @impl true
  def handle_call({:cluster, k}, _from, state) do
    start_time = System.monotonic_time()
    
    case VSMVectorStore.Storage.HNSW.get_all_vectors(state.hnsw_pid) do
      {:ok, vectors} ->
        case KMeans.cluster(state.kmeans_pid, vectors, k) do
          {:ok, clusters} ->
            duration = System.monotonic_time() - start_time
            :telemetry.execute([:vsm_vector_store, :indexing, :kmeans, :cluster], 
              %{duration: duration, clusters_count: length(clusters)}, %{k: k})
            {:reply, {:ok, clusters}, state}
            
          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
        
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:detect_anomalies, contamination}, _from, state) do
    start_time = System.monotonic_time()
    
    case VSMVectorStore.Storage.HNSW.get_all_vectors(state.hnsw_pid) do
      {:ok, vectors} ->
        case AnomalyDetection.detect_anomalies(state.anomaly_detection_pid, vectors, contamination) do
          {:ok, anomalies} ->
            duration = System.monotonic_time() - start_time
            :telemetry.execute([:vsm_vector_store, :ml, :anomaly, :detect], 
              %{duration: duration, anomalies_count: length(anomalies)}, 
              %{contamination: contamination})
            {:reply, {:ok, anomalies}, state}
            
          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
        
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
end