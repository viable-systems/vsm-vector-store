defmodule VSMVectorStore.Indexing.KMeans do
  @moduledoc """
  Pure Elixir K-means clustering implementation with K-means++ initialization.
  
  Features:
  - K-means++ initialization for better cluster quality
  - Configurable convergence criteria
  - Support for high-dimensional vectors
  - Telemetry integration for performance monitoring
  - VSM patterns for error handling and state management
  """
  
  use GenServer
  require Logger
  
  alias VSMVectorStore.Storage.VectorOps
  
  @type vector :: list(float())
  @type vector_id :: binary()
  @type cluster_id :: non_neg_integer()
  @type centroid :: vector()
  @type cluster :: %{
    id: cluster_id(),
    centroid: centroid(),
    members: list(vector_id())
  }
  
  defstruct [
    max_iterations: 100,
    tolerance: 1.0e-4,
    initialization: :kmeans_plus_plus
  ]
  
  ## Public API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Perform K-means clustering on vectors in a space.
  
  ## Parameters
  - space_id: The vector space ID
  - opts: Options including :k for number of clusters
  
  ## Returns
  - {:ok, clustering_result} with centroids and assignments
  - {:error, reason} on failure
  """
  def cluster(space_id, opts) when is_binary(space_id) do
    k = Keyword.get(opts, :k, 3)
    
    # Get all vectors from the space
    case VSMVectorStore.Storage.Manager.get_all_vectors(space_id) do
      {:ok, vectors} when length(vectors) > 0 ->
        # Call the clustering with vectors
        case cluster(__MODULE__, vectors, k, opts) do
          {:ok, clusters} ->
            # Convert to expected format with centroids, assignments, and inertia
            centroids = Enum.map(clusters, & &1.centroid)
            
            # Create assignment map from vector ID to cluster index
            assignments = clusters
            |> Enum.with_index()
            |> Enum.flat_map(fn {cluster, idx} ->
              Enum.map(cluster.members, fn member_id -> {member_id, idx} end)
            end)
            |> Map.new()
            
            # Calculate inertia (sum of squared distances to centroids)
            inertia = calculate_inertia(vectors, clusters)
            
            {:ok, %{
              centroids: centroids,
              assignments: assignments,
              inertia: inertia
            }}
            
          error ->
            error
        end
      
      {:ok, []} ->
        {:error, :no_vectors_in_space}
        
      error ->
        error
    end
  end
  
  @doc """
  Perform K-means clustering on vectors.
  
  ## Parameters
  - vectors: List of {id, vector, metadata} tuples
  - k: Number of clusters
  - opts: Additional options (max_iterations, tolerance, etc.)
  
  ## Returns
  - {:ok, clusters} where clusters is a list of cluster maps
  - {:error, reason} on failure
  """
  @spec cluster(pid(), list({vector_id(), vector(), map()}), pos_integer(), keyword()) ::
    {:ok, list(cluster())} | {:error, term()}
  def cluster(pid, vectors, k, opts) when is_pid(pid) or is_atom(pid) do
    GenServer.call(pid, {:cluster, vectors, k, opts}, 30_000)
  end
  
  @doc """
  Assign a single vector to the nearest cluster centroid.
  """
  @spec assign_to_cluster(pid(), vector(), list(centroid())) :: 
    {:ok, cluster_id()} | {:error, term()}
  def assign_to_cluster(pid \\ __MODULE__, vector, centroids) do
    GenServer.call(pid, {:assign_to_cluster, vector, centroids})
  end
  
  @doc """
  Calculate silhouette score for clustering quality assessment.
  """
  @spec silhouette_score(pid(), list({vector_id(), vector(), map()}), list(cluster())) ::
    {:ok, float()} | {:error, term()}
  def silhouette_score(pid \\ __MODULE__, vectors, clusters) do
    GenServer.call(pid, {:silhouette_score, vectors, clusters}, 30_000)
  end
  
  ## GenServer Callbacks
  
  @impl true
  def init(opts) do
    max_iterations = Keyword.get(opts, :max_iterations, 100)
    tolerance = Keyword.get(opts, :tolerance, 1.0e-4)
    initialization = Keyword.get(opts, :initialization, :kmeans_plus_plus)
    
    state = %__MODULE__{
      max_iterations: max_iterations,
      tolerance: tolerance,
      initialization: initialization
    }
    
    Logger.info("K-means clustering initialized with max_iterations: #{max_iterations}")
    {:ok, state}
  end
  
  @impl true
  def handle_call({:cluster, vectors, k, opts}, _from, state) do
    start_time = System.monotonic_time()
    
    case perform_clustering(vectors, k, state, opts) do
      {:ok, clusters} ->
        duration = System.monotonic_time() - start_time
        :telemetry.execute([:vsm_vector_store, :indexing, :kmeans, :cluster], 
          %{duration: duration, clusters_count: length(clusters), vectors_count: length(vectors)}, 
          %{k: k})
        {:reply, {:ok, clusters}, state}
        
      {:error, reason} ->
        Logger.error("K-means clustering failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:assign_to_cluster, vector, centroids}, _from, state) do
    case find_nearest_centroid(vector, centroids) do
      {:ok, cluster_id} ->
        {:reply, {:ok, cluster_id}, state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:silhouette_score, vectors, clusters}, _from, state) do
    case calculate_silhouette_score(vectors, clusters) do
      {:ok, score} ->
        {:reply, {:ok, score}, state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  ## Private Functions
  
  defp perform_clustering(vectors, k, state, opts) do
    if length(vectors) < k do
      {:error, :insufficient_data}
    else
      vector_data = Enum.map(vectors, fn {id, vector, _metadata} -> {id, vector} end)
      
      case initialize_centroids(vector_data, k, state.initialization) do
        {:ok, initial_centroids} ->
          run_kmeans_iterations(vector_data, initial_centroids, state)
        {:error, reason} ->
          {:error, reason}
      end
    end
  end
  
  defp initialize_centroids(vectors, k, :random) do
    # Simple random initialization
    random_vectors = vectors
    |> Enum.shuffle()
    |> Enum.take(k)
    |> Enum.map(fn {_id, vector} -> vector end)
    
    {:ok, random_vectors}
  end
  
  defp initialize_centroids(vectors, k, :kmeans_plus_plus) do
    # K-means++ initialization for better cluster quality
    case kmeans_plus_plus_init(vectors, k) do
      {:ok, centroids} -> {:ok, centroids}
      {:error, reason} -> {:error, reason}
    end
  end
  
  defp kmeans_plus_plus_init(vectors, k) do
    vector_list = Enum.map(vectors, fn {_id, vector} -> vector end)
    
    # Choose first centroid randomly
    first_centroid = vector_list |> Enum.random()
    
    # Choose remaining centroids based on distance-weighted probability
    centroids = kmeans_plus_plus_loop([first_centroid], vector_list, k - 1)
    
    {:ok, centroids}
  end
  
  defp kmeans_plus_plus_loop(centroids, _vectors, 0) do
    centroids
  end
  
  defp kmeans_plus_plus_loop(centroids, vectors, remaining) do
    # Calculate squared distances from each vector to nearest centroid
    distances_squared = Enum.map(vectors, fn vector ->
      min_distance = centroids
      |> Enum.map(fn centroid -> VectorOps.euclidean_distance(vector, centroid) end)
      |> Enum.min()
      
      {vector, min_distance * min_distance}
    end)
    
    # Calculate total weighted distance
    total_weight = distances_squared
    |> Enum.map(fn {_vector, dist_sq} -> dist_sq end)
    |> Enum.sum()
    
    # Choose next centroid with probability proportional to squared distance
    random_weight = :rand.uniform() * total_weight
    
    next_centroid = select_weighted_random(distances_squared, random_weight, 0.0)
    
    kmeans_plus_plus_loop([next_centroid | centroids], vectors, remaining - 1)
  end
  
  defp select_weighted_random([{vector, weight} | _rest], target_weight, accumulated_weight) 
    when accumulated_weight + weight >= target_weight do
    vector
  end
  
  defp select_weighted_random([{_vector, weight} | rest], target_weight, accumulated_weight) do
    select_weighted_random(rest, target_weight, accumulated_weight + weight)
  end
  
  defp select_weighted_random([], _target_weight, _accumulated_weight) do
    # Fallback - shouldn't happen with proper implementation
    raise "K-means++ selection failed"
  end
  
  defp run_kmeans_iterations(vectors, initial_centroids, state) do
    run_kmeans_loop(vectors, initial_centroids, state, 0)
  end
  
  defp run_kmeans_loop(vectors, centroids, state, iteration) do
    if iteration >= state.max_iterations do
      {:ok, build_final_clusters(vectors, centroids)}
    else
      # Assign vectors to nearest centroids
      assignments = assign_vectors_to_centroids(vectors, centroids)
      
      # Calculate new centroids
      new_centroids = calculate_new_centroids(assignments, centroids)
      
      # Check for convergence
      if converged?(centroids, new_centroids, state.tolerance) do
        {:ok, build_final_clusters(vectors, new_centroids)}
      else
        run_kmeans_loop(vectors, new_centroids, state, iteration + 1)
      end
    end
  end
  
  defp assign_vectors_to_centroids(vectors, centroids) do
    Enum.map(vectors, fn {id, vector} ->
      case find_nearest_centroid(vector, centroids) do
        {:ok, cluster_id} ->
          {id, vector, cluster_id}
        {:error, _} ->
          # Fallback to first cluster
          {id, vector, 0}
      end
    end)
  end
  
  defp find_nearest_centroid(vector, centroids) do
    if Enum.empty?(centroids) do
      {:error, :no_centroids}
    else
      {nearest_idx, _distance} = centroids
      |> Enum.with_index()
      |> Enum.map(fn {centroid, idx} -> 
        {idx, VectorOps.euclidean_distance(vector, centroid)}
      end)
      |> Enum.min_by(fn {_idx, distance} -> distance end)
      
      {:ok, nearest_idx}
    end
  end
  
  defp calculate_new_centroids(assignments, old_centroids) do
    # Group assignments by cluster
    clusters_map = Enum.group_by(assignments, fn {_id, _vector, cluster_id} -> cluster_id end)
    
    # Calculate new centroid for each cluster
    0..(length(old_centroids) - 1)
    |> Enum.map(fn cluster_id ->
      case Map.get(clusters_map, cluster_id) do
        nil ->
          # No vectors assigned to this cluster, keep old centroid
          Enum.at(old_centroids, cluster_id)
        
        cluster_vectors ->
          # Calculate centroid of assigned vectors
          vectors = Enum.map(cluster_vectors, fn {_id, vector, _cluster} -> vector end)
          VectorOps.centroid(vectors)
      end
    end)
  end
  
  defp converged?(old_centroids, new_centroids, tolerance) do
    old_centroids
    |> Enum.zip(new_centroids)
    |> Enum.all?(fn {old, new} ->
      VectorOps.euclidean_distance(old, new) < tolerance
    end)
  end
  
  defp calculate_inertia(vectors, clusters) do
    # Create a map of vector ID to vector
    vector_map = Map.new(vectors, fn {id, vector, _metadata} -> {id, vector} end)
    
    # Calculate sum of squared distances to centroids
    clusters
    |> Enum.map(fn cluster ->
      cluster.members
      |> Enum.map(fn member_id ->
        vector = Map.get(vector_map, member_id)
        distance = VectorOps.euclidean_distance(vector, cluster.centroid)
        distance * distance
      end)
      |> Enum.sum()
    end)
    |> Enum.sum()
  end
  
  defp build_final_clusters(vectors, centroids) do
    assignments = assign_vectors_to_centroids(vectors, centroids)
    clusters_map = Enum.group_by(assignments, fn {_id, _vector, cluster_id} -> cluster_id end)
    
    0..(length(centroids) - 1)
    |> Enum.map(fn cluster_id ->
      centroid = Enum.at(centroids, cluster_id)
      members = case Map.get(clusters_map, cluster_id) do
        nil -> []
        cluster_vectors -> Enum.map(cluster_vectors, fn {id, _vector, _cluster} -> id end)
      end
      
      %{
        id: cluster_id,
        centroid: centroid,
        members: members
      }
    end)
  end
  
  defp calculate_silhouette_score(vectors, clusters) do
    if length(clusters) < 2 do
      {:ok, 0.0}  # Silhouette score is undefined for single cluster
    else
      vector_map = Map.new(vectors, fn {id, vector, _metadata} -> {id, vector} end)
      
      # Create cluster membership map
      cluster_membership = clusters
      |> Enum.with_index()
      |> Enum.flat_map(fn {cluster, cluster_idx} ->
        Enum.map(cluster.members, fn member_id -> {member_id, cluster_idx} end)
      end)
      |> Map.new()
      
      silhouette_scores = Enum.map(vectors, fn {id, vector, _metadata} ->
        calculate_vector_silhouette(id, vector, cluster_membership, vector_map, clusters)
      end)
      
      case silhouette_scores do
        [] -> {:ok, 0.0}
        scores -> {:ok, Enum.sum(scores) / length(scores)}
      end
    end
  end
  
  defp calculate_vector_silhouette(vector_id, vector, cluster_membership, vector_map, clusters) do
    own_cluster_id = Map.get(cluster_membership, vector_id)
    own_cluster = Enum.at(clusters, own_cluster_id)
    
    # Calculate average distance to vectors in same cluster (a)
    a = if length(own_cluster.members) <= 1 do
      0.0
    else
      same_cluster_distances = own_cluster.members
      |> Enum.filter(fn id -> id != vector_id end)
      |> Enum.map(fn other_id ->
        other_vector = Map.get(vector_map, other_id)
        VectorOps.euclidean_distance(vector, other_vector)
      end)
      
      Enum.sum(same_cluster_distances) / length(same_cluster_distances)
    end
    
    # Calculate minimum average distance to vectors in other clusters (b)
    b = clusters
    |> Enum.with_index()
    |> Enum.filter(fn {_cluster, idx} -> idx != own_cluster_id end)
    |> Enum.map(fn {other_cluster, _idx} ->
      if Enum.empty?(other_cluster.members) do
        :infinity
      else
        other_cluster_distances = Enum.map(other_cluster.members, fn other_id ->
          other_vector = Map.get(vector_map, other_id)
          VectorOps.euclidean_distance(vector, other_vector)
        end)
        
        Enum.sum(other_cluster_distances) / length(other_cluster_distances)
      end
    end)
    |> Enum.filter(fn dist -> dist != :infinity end)
    |> case do
      [] -> 0.0
      distances -> Enum.min(distances)
    end
    
    # Calculate silhouette score for this vector
    max_ab = max(a, b)
    if max_ab == 0.0 do
      0.0
    else
      (b - a) / max_ab
    end
  end
end