defmodule VSMVectorStore.Storage.Space do
  @moduledoc """
  Individual vector space process that manages its own HNSW index.
  """
  
  use GenServer
  require Logger
  
  alias VSMVectorStore.Storage.HNSW
  
  defstruct [:id, :name, :dimensions, :hnsw_pid, :distance_metric]
  
  def start_link(opts) do
    space_id = Keyword.fetch!(opts, :space_id)
    GenServer.start_link(__MODULE__, opts, name: via_tuple(space_id))
  end
  
  def search(space_id, query_vector, opts \\ []) do
    GenServer.call(via_tuple(space_id), {:search, query_vector, opts})
  end
  
  def insert_vectors(space_id, vectors, ids, metadata) do
    GenServer.call(via_tuple(space_id), {:insert_vectors, vectors, ids, metadata})
  end
  
  @impl true
  def init(opts) do
    space_id = Keyword.fetch!(opts, :space_id)
    space = Keyword.fetch!(opts, :space)
    
    # Start HNSW index for this space
    {:ok, hnsw_pid} = HNSW.start_link([
      space_id: space_id,
      dimensions: space.dimensions,
      distance_metric: space.distance_metric
    ])
    
    state = %__MODULE__{
      id: space_id,
      name: space.name,
      dimensions: space.dimensions,
      hnsw_pid: hnsw_pid,
      distance_metric: space.distance_metric
    }
    
    {:ok, state}
  end
  
  @impl true
  def handle_call({:search, query_vector, opts}, _from, state) do
    result = HNSW.search(state.hnsw_pid, query_vector, opts)
    {:reply, result, state}
  end
  
  @impl true
  def handle_call({:insert_vectors, vectors, ids, metadata}, _from, state) do
    result = HNSW.batch_insert(state.hnsw_pid, ids, vectors, metadata)
    {:reply, result, state}
  end
  
  defp via_tuple(space_id) do
    {:via, Registry, {VSMVectorStore.Registry, {:space, space_id}}}
  end
end