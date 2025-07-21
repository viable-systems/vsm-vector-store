defmodule VSMVectorStore.Storage.HNSW do
  @moduledoc """
  Hierarchical Navigable Small World (HNSW) implementation for approximate nearest neighbor search.
  
  This pure Elixir implementation provides O(log N) search complexity with VSM patterns.
  Features:
  - Multi-layer graph structure for efficient search
  - Dynamic insertion and deletion
  - Configurable parameters (M, efConstruction, efSearch)
  - Thread-safe operations through GenServer
  """
  
  use GenServer
  require Logger
  
  alias VSMVectorStore.Storage.VectorOps
  
  @type vector_id :: binary()
  @type vector :: list(float())
  @type metadata :: map()
  @type level :: non_neg_integer()
  @type hnsw_node :: %{
    id: vector_id(),
    vector: vector(),
    metadata: metadata(),
    connections: %{level() => MapSet.t(vector_id())}
  }
  
  defstruct [
    nodes: %{},
    entry_point: nil,
    dimension: 128,
    max_connections: 16,
    max_connections_level0: 32,
    level_multiplier: 1.0 / :math.log(2.0),
    ef_construction: 200,
    node_count: 0
  ]
  
  ## Public API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Insert a vector with metadata into the HNSW index.
  """
  @spec insert(pid(), vector_id(), vector(), metadata()) :: :ok | {:error, term()}
  def insert(pid \\ __MODULE__, id, vector, metadata \\ %{}) do
    GenServer.call(pid, {:insert, id, vector, metadata})
  end
  
  @doc """
  Search for k nearest neighbors of the query vector.
  """
  @spec search(pid(), vector(), pos_integer(), pos_integer()) :: 
    {:ok, list({vector_id(), float(), metadata()})} | {:error, term()}
  def search(pid \\ __MODULE__, query_vector, k, ef \\ 50) do
    GenServer.call(pid, {:search, query_vector, k, ef})
  end
  
  @doc """
  Delete a vector by ID from the index.
  """
  @spec delete(pid(), vector_id()) :: :ok | {:error, term()}
  def delete(pid \\ __MODULE__, id) do
    GenServer.call(pid, {:delete, id})
  end
  
  @doc """
  Get a vector and metadata by ID.
  """
  @spec get(pid(), vector_id()) :: {:ok, {vector(), metadata()}} | {:error, :not_found}
  def get(pid \\ __MODULE__, id) do
    GenServer.call(pid, {:get, id})
  end
  
  @doc """
  Get all vectors for clustering and ML operations.
  """
  @spec get_all_vectors(pid()) :: {:ok, list({vector_id(), vector(), metadata()})}
  def get_all_vectors(pid \\ __MODULE__) do
    GenServer.call(pid, :get_all_vectors)
  end
  
  ## GenServer Callbacks
  
  @impl true
  def init(opts) do
    dimension = Keyword.get(opts, :dimension, 128)
    max_connections = Keyword.get(opts, :max_connections, 16)
    ef_construction = Keyword.get(opts, :ef_construction, 200)
    
    state = %__MODULE__{
      dimension: dimension,
      max_connections: max_connections,
      max_connections_level0: max_connections * 2,
      ef_construction: ef_construction
    }
    
    Logger.info("HNSW index initialized with dimension: #{dimension}, M: #{max_connections}")
    {:ok, state}
  end
  
  @impl true
  def handle_call({:insert, id, vector, metadata}, _from, state) do
    if length(vector) != state.dimension do
      {:reply, {:error, :dimension_mismatch}, state}
    else
      case insert_node(state, id, vector, metadata) do
        {:ok, new_state} ->
          {:reply, :ok, new_state}
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end
  
  @impl true
  def handle_call({:search, query_vector, k, ef}, _from, state) do
    if length(query_vector) != state.dimension do
      {:reply, {:error, :dimension_mismatch}, state}
    else
      case search_knn(state, query_vector, k, ef) do
        {:ok, results} ->
          formatted_results = Enum.map(results, fn {id, distance} ->
            node = Map.get(state.nodes, id)
            {id, distance, node.metadata}
          end)
          {:reply, {:ok, formatted_results}, state}
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end
  
  @impl true
  def handle_call({:delete, id}, _from, state) do
    case Map.get(state.nodes, id) do
      nil ->
        {:reply, {:error, :not_found}, state}
      
      _node ->
        new_state = delete_node(state, id)
        {:reply, :ok, new_state}
    end
  end
  
  @impl true
  def handle_call({:get, id}, _from, state) do
    case Map.get(state.nodes, id) do
      nil ->
        {:reply, {:error, :not_found}, state}
      
      node ->
        {:reply, {:ok, {node.vector, node.metadata}}, state}
    end
  end
  
  @impl true
  def handle_call(:get_all_vectors, _from, state) do
    vectors = Enum.map(state.nodes, fn {id, node} ->
      {id, node.vector, node.metadata}
    end)
    {:reply, {:ok, vectors}, state}
  end
  
  ## Private Functions
  
  defp insert_node(state, id, vector, metadata) do
    level = get_random_level(state.level_multiplier)
    
    node = %{
      id: id,
      vector: vector,
      metadata: metadata,
      connections: initialize_connections(level)
    }
    
    new_state = %{state | 
      nodes: Map.put(state.nodes, id, node),
      node_count: state.node_count + 1
    }
    
    case state.entry_point do
      nil ->
        # First node becomes entry point
        {:ok, %{new_state | entry_point: id}}
      
      entry_id ->
        # Insert with search and connection
        insert_with_search(new_state, id, level, entry_id)
    end
  end
  
  defp insert_with_search(state, new_id, level, entry_id) do
    new_node = Map.get(state.nodes, new_id)
    
    # Search from top level down to level+1
    entry_level = get_node_level(Map.get(state.nodes, entry_id))
    
    current_closest = [entry_id]
    
    # Search from entry level down to target level + 1
    current_closest = if entry_level > level do
      search_layer(state, new_node.vector, current_closest, 1, entry_level, level + 1)
    else
      current_closest
    end
    
    # Search and connect at each level from level down to 0
    updated_state = Enum.reduce(level..0, state, fn lev, acc_state ->
      # Search layer for ef_construction candidates
      candidates = search_layer(acc_state, new_node.vector, current_closest, 
                               acc_state.ef_construction, lev, lev)
      
      # Select M best candidates for connections
      max_conn = if lev == 0, do: acc_state.max_connections_level0, else: acc_state.max_connections
      selected = select_neighbors_heuristic(acc_state, new_node.vector, candidates, max_conn)
      
      # Add bidirectional connections
      connect_nodes(acc_state, new_id, selected, lev)
    end)
    
    # Update entry point if necessary
    final_state = if level > entry_level do
      %{updated_state | entry_point: new_id}
    else
      updated_state
    end
    
    {:ok, final_state}
  end
  
  defp search_knn(state, query_vector, k, ef) do
    case state.entry_point do
      nil ->
        {:ok, []}
      
      entry_id ->
        entry_node = Map.get(state.nodes, entry_id)
        entry_level = get_node_level(entry_node)
        
        # Search from entry level down to level 1
        closest = [entry_id]
        closest = if entry_level > 0 do
          search_layer(state, query_vector, closest, 1, entry_level, 1)
        else
          closest
        end
        
        # Search level 0 with ef parameter
        candidates = search_layer(state, query_vector, closest, max(ef, k), 0, 0)
        
        # Return top k candidates with distances
        results = candidates
        |> Enum.take(k)
        |> Enum.map(fn id ->
          node = Map.get(state.nodes, id)
          distance = VectorOps.euclidean_distance(query_vector, node.vector)
          {id, distance}
        end)
        
        {:ok, results}
    end
  end
  
  # Hierarchical search from current_level down to target_level
  defp search_layer(state, query_vector, entry_points, num_closest, current_level, target_level) when current_level > target_level do
    # Search through multiple levels
    current_level..target_level
    |> Enum.reduce(entry_points, fn level, current_entry_points ->
      search_layer(state, query_vector, current_entry_points, num_closest, level, level)
    end)
  end
  
  # Single level search
  defp search_layer(state, query_vector, entry_points, num_closest, current_level, target_level) when current_level == target_level do
    # Priority queue for dynamic candidates (min-heap by distance)
    visited = MapSet.new()
    candidates = :gb_sets.new()
    w = :gb_sets.new()  # Dynamic list of found neighbors
    
    # Initialize with entry points
    {candidates, w, visited} = Enum.reduce(entry_points, {candidates, w, visited}, 
      fn ep, {cand_acc, w_acc, vis_acc} ->
        node = Map.get(state.nodes, ep)
        dist = VectorOps.euclidean_distance(query_vector, node.vector)
        new_cand = :gb_sets.add_element({dist, ep}, cand_acc)
        new_w = :gb_sets.add_element({-dist, ep}, w_acc)  # Negative for max-heap behavior
        new_vis = MapSet.put(vis_acc, ep)
        {new_cand, new_w, new_vis}
      end)
    
    search_layer_loop(state, query_vector, candidates, w, visited, num_closest, current_level)
  end
  
  defp search_layer_loop(state, query_vector, candidates, w, visited, num_closest, level) do
    case :gb_sets.is_empty(candidates) do
      true ->
        # Extract IDs from w (closest neighbors found)
        w 
        |> :gb_sets.to_list()
        |> Enum.map(fn {_neg_dist, id} -> id end)
        |> Enum.take(num_closest)
      
      false ->
        # Get closest unvisited candidate
        {{dist, c}, remaining_candidates} = :gb_sets.take_smallest(candidates)
        
        # Check if c is closer than the furthest in w
        w_size = :gb_sets.size(w)
        should_continue = if w_size < num_closest do
          true
        else
          {furthest_neg_dist, _} = :gb_sets.largest(w)
          dist < -furthest_neg_dist
        end
        
        if should_continue do
          # Explore connections of c
          node = Map.get(state.nodes, c)
          connections = Map.get(node.connections, level, MapSet.new())
          
          {new_candidates, new_w, new_visited} = Enum.reduce(connections, 
            {remaining_candidates, w, visited}, fn e, {cand_acc, w_acc, vis_acc} ->
              if MapSet.member?(vis_acc, e) do
                {cand_acc, w_acc, vis_acc}
              else
                e_node = Map.get(state.nodes, e)
                e_dist = VectorOps.euclidean_distance(query_vector, e_node.vector)
                new_vis = MapSet.put(vis_acc, e)
                
                # Check if e should be added to w
                w_size = :gb_sets.size(w_acc)
                {new_cand, new_w} = if w_size < num_closest do
                  # Add to both candidates and w
                  cand_with_e = :gb_sets.add_element({e_dist, e}, cand_acc)
                  w_with_e = :gb_sets.add_element({-e_dist, e}, w_acc)
                  {cand_with_e, w_with_e}
                else
                  # Check if e is better than worst in w
                  {worst_neg_dist, worst_id} = :gb_sets.largest(w_acc)
                  if e_dist < -worst_neg_dist do
                    cand_with_e = :gb_sets.add_element({e_dist, e}, cand_acc)
                    w_without_worst = :gb_sets.delete_any({worst_neg_dist, worst_id}, w_acc)
                    w_with_e = :gb_sets.add_element({-e_dist, e}, w_without_worst)
                    {cand_with_e, w_with_e}
                  else
                    {cand_acc, w_acc}
                  end
                end
                
                {new_cand, new_w, new_vis}
              end
            end)
          
          search_layer_loop(state, query_vector, new_candidates, new_w, new_visited, num_closest, level)
        else
          # No improvement possible, return current w
          w 
          |> :gb_sets.to_list()
          |> Enum.map(fn {_neg_dist, id} -> id end)
          |> Enum.take(num_closest)
        end
    end
  end
  
  defp delete_node(state, id) do
    node = Map.get(state.nodes, id)
    
    # Remove connections to this node from all connected nodes
    updated_nodes = Enum.reduce(node.connections, state.nodes, fn {level, connections}, acc_nodes ->
      Enum.reduce(connections, acc_nodes, fn connected_id, inner_acc ->
        case Map.get(inner_acc, connected_id) do
          nil -> inner_acc
          connected_node ->
            updated_connections = Map.update(connected_node.connections, level, MapSet.new(), 
              fn level_connections -> MapSet.delete(level_connections, id) end)
            updated_node = %{connected_node | connections: updated_connections}
            Map.put(inner_acc, connected_id, updated_node)
        end
      end)
    end)
    
    # Remove the node itself
    final_nodes = Map.delete(updated_nodes, id)
    
    # Update entry point if necessary
    new_entry_point = if state.entry_point == id do
      case Map.keys(final_nodes) do
        [] -> nil
        [first_id | _] -> first_id
      end
    else
      state.entry_point
    end
    
    %{state | 
      nodes: final_nodes,
      entry_point: new_entry_point,
      node_count: state.node_count - 1
    }
  end
  
  defp get_random_level(level_multiplier) do
    # Generate level using exponential decay probability
    level = 0
    while_condition = fn -> :rand.uniform() < 0.5 end
    
    get_random_level_loop(level, level_multiplier, while_condition)
  end
  
  defp get_random_level_loop(level, _level_multiplier, while_condition) do
    if while_condition.() do
      get_random_level_loop(level + 1, _level_multiplier, while_condition)
    else
      level
    end
  end
  
  defp get_node_level(node) do
    node.connections
    |> Map.keys()
    |> Enum.max(fn -> -1 end)
  end
  
  defp initialize_connections(max_level) do
    0..max_level
    |> Enum.map(fn level -> {level, MapSet.new()} end)
    |> Map.new()
  end
  
  defp select_neighbors_heuristic(state, query_vector, candidates, max_connections) do
    # Simple greedy selection - can be improved with more sophisticated heuristics
    candidates
    |> Enum.map(fn id ->
      node = Map.get(state.nodes, id)
      distance = VectorOps.euclidean_distance(query_vector, node.vector)
      {distance, id}
    end)
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.take(max_connections)
    |> Enum.map(&elem(&1, 1))
  end
  
  defp connect_nodes(state, new_id, selected_neighbors, level) do
    # Add bidirectional connections
    updated_nodes = Enum.reduce(selected_neighbors, state.nodes, fn neighbor_id, acc_nodes ->
      # Add new_id to neighbor's connections
      neighbor_node = Map.get(acc_nodes, neighbor_id)
      updated_neighbor_connections = Map.update(neighbor_node.connections, level, 
        MapSet.new(), fn level_connections -> MapSet.put(level_connections, new_id) end)
      updated_neighbor = %{neighbor_node | connections: updated_neighbor_connections}
      
      # Add neighbor_id to new node's connections
      new_node = Map.get(acc_nodes, new_id)
      updated_new_connections = Map.update(new_node.connections, level, 
        MapSet.new(), fn level_connections -> MapSet.put(level_connections, neighbor_id) end)
      updated_new_node = %{new_node | connections: updated_new_connections}
      
      acc_nodes
      |> Map.put(neighbor_id, updated_neighbor)
      |> Map.put(new_id, updated_new_node)
    end)
    
    %{state | nodes: updated_nodes}
  end
end