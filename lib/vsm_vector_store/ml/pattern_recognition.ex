defmodule VSMVectorStore.ML.PatternRecognition do
  @moduledoc """
  Pattern recognition system for VSM Vector Store using cosine similarity and clustering analysis.
  
  Features:
  - Semantic similarity detection using cosine similarity
  - Pattern clustering and classification
  - Temporal pattern analysis
  - Statistical pattern validation
  - Multi-dimensional pattern matching
  - VSM-compatible error handling and telemetry
  """
  
  use GenServer
  require Logger
  
  alias VSMVectorStore.Storage.VectorOps
  alias VSMVectorStore.Indexing.KMeans
  
  @type vector :: list(float())
  @type vector_id :: binary()
  @type pattern :: %{
    id: binary(),
    centroid: vector(),
    members: list(vector_id()),
    similarity_threshold: float(),
    confidence: float(),
    created_at: integer(),
    updated_at: integer()
  }
  @type pattern_match :: %{
    pattern_id: binary(),
    similarity: float(),
    confidence: float()
  }
  
  defstruct [
    patterns: %{},
    similarity_threshold: 0.8,
    min_pattern_size: 3,
    max_patterns: 1000,
    temporal_window_ms: 3_600_000  # 1 hour
  ]
  
  ## Public API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Analyze patterns in vectors from a space.
  """
  def analyze(space_id, opts \\ []) when is_binary(space_id) do
    # Get all vectors from the space
    case VSMVectorStore.Storage.Manager.get_all_vectors(space_id) do
      {:ok, vectors} when length(vectors) > 0 ->
        # Learn patterns
        case learn_patterns(__MODULE__, vectors, opts) do
          {:ok, patterns} ->
            # Analyze the patterns for summary statistics
            cluster_count = length(patterns)
            density_peaks = patterns
            |> Enum.filter(& length(&1.members) > 5)
            |> length()
            
            outlier_regions = patterns
            |> Enum.filter(& length(&1.members) <= 2)
            |> length()
            
            {:ok, %{
              clusters: cluster_count,
              density_peaks: density_peaks,
              outlier_regions: outlier_regions,
              patterns: patterns
            }}
            
          error ->
            error
        end
      
      {:ok, []} ->
        {:ok, %{
          clusters: 0,
          density_peaks: 0,
          outlier_regions: 0,
          patterns: []
        }}
        
      error ->
        error
    end
  end
  
  @doc """
  Learn patterns from a set of vectors.
  Groups similar vectors into patterns using clustering analysis.
  """
  @spec learn_patterns(pid(), list({vector_id(), vector(), map()}), keyword()) ::
    {:ok, list(pattern())} | {:error, term()}
  def learn_patterns(pid \\ __MODULE__, vectors, opts \\ []) do
    GenServer.call(pid, {:learn_patterns, vectors, opts}, 30_000)
  end
  
  @doc """
  Recognize patterns in a query vector.
  Returns list of matching patterns with similarity scores.
  """
  @spec recognize(pid(), vector()) :: {:ok, list(pattern_match())} | {:error, term()}
  def recognize(pid \\ __MODULE__, query_vector) do
    GenServer.call(pid, {:recognize, query_vector})
  end
  
  @doc """
  Add a new vector to existing pattern recognition system.
  Updates patterns incrementally without full retraining.
  """
  @spec add_vector(pid(), vector_id(), vector(), map()) :: :ok | {:error, term()}
  def add_vector(pid \\ __MODULE__, id, vector, metadata \\ %{}) do
    GenServer.call(pid, {:add_vector, id, vector, metadata})
  end
  
  @doc """
  Get all learned patterns.
  """
  @spec get_patterns(pid()) :: {:ok, list(pattern())}
  def get_patterns(pid \\ __MODULE__) do
    GenServer.call(pid, :get_patterns)
  end
  
  @doc """
  Validate pattern quality using statistical measures.
  """
  @spec validate_patterns(pid()) :: {:ok, map()} | {:error, term()}
  def validate_patterns(pid \\ __MODULE__) do
    GenServer.call(pid, :validate_patterns, 30_000)
  end
  
  @doc """
  Find similar patterns based on centroid similarity.
  """
  @spec find_similar_patterns(pid(), pattern(), float()) ::
    {:ok, list({pattern(), float()})} | {:error, term()}
  def find_similar_patterns(pid \\ __MODULE__, query_pattern, threshold \\ 0.7) do
    GenServer.call(pid, {:find_similar_patterns, query_pattern, threshold})
  end
  
  @doc """
  Analyze temporal patterns in vector sequences.
  """
  @spec analyze_temporal_patterns(pid(), list({vector_id(), vector(), integer()})) ::
    {:ok, map()} | {:error, term()}
  def analyze_temporal_patterns(pid \\ __MODULE__, timestamped_vectors) do
    GenServer.call(pid, {:analyze_temporal_patterns, timestamped_vectors}, 30_000)
  end
  
  ## GenServer Callbacks
  
  @impl true
  def init(opts) do
    similarity_threshold = Keyword.get(opts, :similarity_threshold, 0.8)
    min_pattern_size = Keyword.get(opts, :min_pattern_size, 3)
    max_patterns = Keyword.get(opts, :max_patterns, 1000)
    temporal_window_ms = Keyword.get(opts, :temporal_window_ms, 3_600_000)
    
    state = %__MODULE__{
      similarity_threshold: similarity_threshold,
      min_pattern_size: min_pattern_size,
      max_patterns: max_patterns,
      temporal_window_ms: temporal_window_ms
    }
    
    Logger.info("Pattern Recognition initialized with similarity threshold: #{similarity_threshold}")
    {:ok, state}
  end
  
  @impl true
  def handle_call({:learn_patterns, vectors, opts}, _from, state) do
    start_time = System.monotonic_time()
    
    case learn_patterns_impl(vectors, state, opts) do
      {:ok, patterns, new_state} ->
        duration = System.monotonic_time() - start_time
        :telemetry.execute([:vsm_vector_store, :ml, :pattern_recognition, :learn], 
          %{duration: duration, patterns_count: length(patterns), vectors_count: length(vectors)}, %{})
        {:reply, {:ok, patterns}, new_state}
        
      {:error, reason} ->
        Logger.error("Pattern learning failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:recognize, query_vector}, _from, state) do
    start_time = System.monotonic_time()
    
    case recognize_patterns_impl(query_vector, state) do
      {:ok, matches} ->
        duration = System.monotonic_time() - start_time
        :telemetry.execute([:vsm_vector_store, :ml, :pattern_recognition, :recognize], 
          %{duration: duration, matches_count: length(matches)}, %{})
        {:reply, {:ok, matches}, state}
        
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:add_vector, id, vector, metadata}, _from, state) do
    case add_vector_impl(id, vector, metadata, state) do
      {:ok, new_state} ->
        {:reply, :ok, new_state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call(:get_patterns, _from, state) do
    patterns = Map.values(state.patterns)
    {:reply, {:ok, patterns}, state}
  end
  
  @impl true
  def handle_call(:validate_patterns, _from, state) do
    case validate_patterns_impl(state) do
      {:ok, validation_results} ->
        {:reply, {:ok, validation_results}, state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:find_similar_patterns, query_pattern, threshold}, _from, state) do
    case find_similar_patterns_impl(query_pattern, threshold, state) do
      {:ok, similar_patterns} ->
        {:reply, {:ok, similar_patterns}, state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:analyze_temporal_patterns, timestamped_vectors}, _from, state) do
    case analyze_temporal_patterns_impl(timestamped_vectors, state) do
      {:ok, temporal_analysis} ->
        {:reply, {:ok, temporal_analysis}, state}
      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
  
  ## Private Functions
  
  defp learn_patterns_impl(vectors, state, opts) do
    if length(vectors) < state.min_pattern_size do
      {:error, :insufficient_data}
    else
      # Determine optimal number of clusters using elbow method
      max_k = min(length(vectors) |> div(state.min_pattern_size), 20)
      
      optimal_k = case determine_optimal_clusters(vectors, max_k) do
        {:ok, k} -> k
        {:error, _} -> max(2, length(vectors) |> div(5))  # Fallback
      end
      
      # Perform clustering to identify patterns
      vectors_for_clustering = Enum.map(vectors, fn {id, vector, metadata} ->
        {id, vector, metadata}
      end)
      
      case KMeans.cluster(VSMVectorStore.Indexing.KMeans, vectors_for_clustering, optimal_k) do
        {:ok, clusters} ->
          patterns = create_patterns_from_clusters(clusters, state)
          patterns_map = Map.new(patterns, fn pattern -> {pattern.id, pattern} end)
          
          new_state = %{state | patterns: patterns_map}
          {:ok, patterns, new_state}
          
        {:error, reason} ->
          {:error, reason}
      end
    end
  end
  
  defp determine_optimal_clusters(vectors, max_k) do
    # Use elbow method with within-cluster sum of squares (WCSS)
    wcss_scores = 2..max_k
    |> Enum.map(fn k ->
      vectors_for_clustering = Enum.map(vectors, fn {id, vector, metadata} ->
        {id, vector, metadata}
      end)
      
      case KMeans.cluster(VSMVectorStore.Indexing.KMeans, vectors_for_clustering, k) do
        {:ok, clusters} ->
          wcss = calculate_wcss(clusters, vectors_for_clustering)
          {k, wcss}
        {:error, _} ->
          {k, :infinity}
      end
    end)
    |> Enum.filter(fn {_k, wcss} -> wcss != :infinity end)
    
    case find_elbow_point(wcss_scores) do
      {:ok, optimal_k} -> {:ok, optimal_k}
      {:error, _} -> 
        # Fallback to middle value
        {:ok, max_k |> div(2)}
    end
  end
  
  defp calculate_wcss(clusters, vectors) do
    vector_map = Map.new(vectors, fn {id, vector, _metadata} -> {id, vector} end)
    
    Enum.reduce(clusters, 0.0, fn cluster, acc ->
      cluster_wcss = Enum.reduce(cluster.members, 0.0, fn member_id, inner_acc ->
        member_vector = Map.get(vector_map, member_id)
        distance = VectorOps.euclidean_distance(member_vector, cluster.centroid)
        inner_acc + distance * distance
      end)
      acc + cluster_wcss
    end)
  end
  
  defp find_elbow_point(wcss_scores) do
    if length(wcss_scores) < 3 do
      {:error, :insufficient_points}
    else
      # Calculate rate of change in WCSS
      rate_changes = wcss_scores
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [{k1, wcss1}, {k2, wcss2}] ->
        {k2, wcss1 - wcss2}  # Decrease in WCSS
      end)
      
      # Find point where rate of decrease drops significantly
      case find_significant_drop(rate_changes) do
        {:ok, k} -> {:ok, k}
        {:error, _} ->
          # Fallback to maximum rate of change
          {k, _change} = Enum.max_by(rate_changes, fn {_k, change} -> change end)
          {:ok, k}
      end
    end
  end
  
  defp find_significant_drop(rate_changes) do
    if length(rate_changes) < 2 do
      {:error, :insufficient_points}
    else
      # Look for point where next change is < 50% of current change
      rate_changes
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.find(fn [{k1, change1}, {_k2, change2}] ->
        change2 < change1 * 0.5
      end)
      |> case do
        [{k, _change}, _] -> {:ok, k}
        nil -> {:error, :no_significant_drop}
      end
    end
  end
  
  defp create_patterns_from_clusters(clusters, state) do
    now = System.system_time(:millisecond)
    
    clusters
    |> Enum.filter(fn cluster -> length(cluster.members) >= state.min_pattern_size end)
    |> Enum.map(fn cluster ->
      pattern_id = generate_pattern_id()
      
      %{
        id: pattern_id,
        centroid: cluster.centroid,
        members: cluster.members,
        similarity_threshold: state.similarity_threshold,
        confidence: calculate_pattern_confidence(cluster),
        created_at: now,
        updated_at: now
      }
    end)
  end
  
  defp calculate_pattern_confidence(cluster) do
    # Calculate confidence based on cluster cohesion
    # Higher confidence for tighter clusters
    member_count = length(cluster.members)
    
    cond do
      member_count < 3 -> 0.3
      member_count < 5 -> 0.5
      member_count < 10 -> 0.7
      true -> 0.9
    end
  end
  
  defp generate_pattern_id do
    "pattern_" <> (:crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower))
  end
  
  defp recognize_patterns_impl(query_vector, state) do
    if map_size(state.patterns) == 0 do
      {:ok, []}
    else
      matches = state.patterns
      |> Map.values()
      |> Enum.map(fn pattern ->
        similarity = VectorOps.cosine_similarity(query_vector, pattern.centroid)
        
        if similarity >= pattern.similarity_threshold do
          %{
            pattern_id: pattern.id,
            similarity: similarity,
            confidence: pattern.confidence * similarity
          }
        else
          nil
        end
      end)
      |> Enum.filter(fn match -> not is_nil(match) end)
      |> Enum.sort_by(fn match -> match.similarity end, :desc)
      
      {:ok, matches}
    end
  end
  
  defp add_vector_impl(id, vector, metadata, state) do
    # Find best matching pattern
    case recognize_patterns_impl(vector, state) do
      {:ok, []} ->
        # No matching pattern, could create new pattern if we have enough similar vectors
        {:ok, state}
        
      {:ok, [best_match | _]} ->
        # Add to best matching pattern
        pattern = Map.get(state.patterns, best_match.pattern_id)
        updated_members = [id | pattern.members]
        
        # Recalculate centroid incrementally
        current_centroid = pattern.centroid
        member_count = length(pattern.members)
        new_centroid = VectorOps.lerp(current_centroid, vector, 1.0 / (member_count + 1))
        
        updated_pattern = %{pattern |
          members: updated_members,
          centroid: new_centroid,
          updated_at: System.system_time(:millisecond),
          confidence: calculate_incremental_confidence(pattern, 1)
        }
        
        updated_patterns = Map.put(state.patterns, pattern.id, updated_pattern)
        new_state = %{state | patterns: updated_patterns}
        
        {:ok, new_state}
        
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  defp calculate_incremental_confidence(pattern, additional_members) do
    new_member_count = length(pattern.members) + additional_members
    
    cond do
      new_member_count < 3 -> 0.3
      new_member_count < 5 -> 0.5
      new_member_count < 10 -> 0.7
      true -> min(0.95, pattern.confidence + 0.05)
    end
  end
  
  defp validate_patterns_impl(state) do
    if map_size(state.patterns) == 0 do
      {:ok, %{
        total_patterns: 0,
        average_confidence: 0.0,
        coverage: 0.0,
        quality_score: 0.0
      }}
    else
      patterns = Map.values(state.patterns)
      
      total_patterns = length(patterns)
      average_confidence = patterns
      |> Enum.map(fn pattern -> pattern.confidence end)
      |> Enum.sum()
      |> Kernel./(total_patterns)
      
      total_members = patterns
      |> Enum.map(fn pattern -> length(pattern.members) end)
      |> Enum.sum()
      
      # Calculate pattern overlap (ideally should be low)
      overlap_score = calculate_pattern_overlap(patterns)
      
      # Calculate overall quality score
      quality_score = (average_confidence * 0.5) + ((1.0 - overlap_score) * 0.3) + 
                     (min(1.0, total_members / 100.0) * 0.2)
      
      validation_results = %{
        total_patterns: total_patterns,
        average_confidence: average_confidence,
        total_members: total_members,
        overlap_score: overlap_score,
        quality_score: quality_score,
        patterns_by_size: group_patterns_by_size(patterns)
      }
      
      {:ok, validation_results}
    end
  end
  
  defp calculate_pattern_overlap(patterns) do
    # Calculate centroid similarity between patterns
    if length(patterns) < 2 do
      0.0
    else
      similarities = patterns
      |> Enum.with_index()
      |> Enum.flat_map(fn {pattern1, idx1} ->
        patterns
        |> Enum.with_index()
        |> Enum.filter(fn {_pattern2, idx2} -> idx2 > idx1 end)
        |> Enum.map(fn {pattern2, _idx2} ->
          VectorOps.cosine_similarity(pattern1.centroid, pattern2.centroid)
        end)
      end)
      
      case similarities do
        [] -> 0.0
        _ -> Enum.sum(similarities) / length(similarities)
      end
    end
  end
  
  defp group_patterns_by_size(patterns) do
    patterns
    |> Enum.group_by(fn pattern ->
      member_count = length(pattern.members)
      cond do
        member_count < 5 -> :small
        member_count < 10 -> :medium
        member_count < 20 -> :large
        true -> :xlarge
      end
    end)
    |> Enum.map(fn {size, pattern_list} -> {size, length(pattern_list)} end)
    |> Map.new()
  end
  
  defp find_similar_patterns_impl(query_pattern, threshold, state) do
    similar_patterns = state.patterns
    |> Map.values()
    |> Enum.filter(fn pattern -> pattern.id != query_pattern.id end)
    |> Enum.map(fn pattern ->
      similarity = VectorOps.cosine_similarity(query_pattern.centroid, pattern.centroid)
      if similarity >= threshold do
        {pattern, similarity}
      else
        nil
      end
    end)
    |> Enum.filter(fn result -> not is_nil(result) end)
    |> Enum.sort_by(fn {_pattern, similarity} -> similarity end, :desc)
    
    {:ok, similar_patterns}
  end
  
  defp analyze_temporal_patterns_impl(timestamped_vectors, state) do
    if length(timestamped_vectors) < 2 do
      {:ok, %{
        temporal_clusters: [],
        trend_analysis: %{},
        periodicity: %{}
      }}
    else
      # Sort by timestamp
      sorted_vectors = Enum.sort_by(timestamped_vectors, fn {_id, _vector, timestamp} -> timestamp end)
      
      # Analyze temporal clusters
      temporal_clusters = analyze_temporal_clusters(sorted_vectors, state.temporal_window_ms)
      
      # Analyze trends over time
      trend_analysis = analyze_vector_trends(sorted_vectors)
      
      # Detect periodicity
      periodicity = detect_periodicity(sorted_vectors)
      
      temporal_analysis = %{
        temporal_clusters: temporal_clusters,
        trend_analysis: trend_analysis,
        periodicity: periodicity,
        total_vectors: length(timestamped_vectors),
        time_span_ms: get_time_span(sorted_vectors)
      }
      
      {:ok, temporal_analysis}
    end
  end
  
  defp analyze_temporal_clusters(sorted_vectors, window_ms) do
    # Group vectors into time windows
    time_windows = sorted_vectors
    |> Enum.chunk_by(fn {_id, _vector, timestamp} ->
      div(timestamp, window_ms)
    end)
    
    # Analyze each time window
    time_windows
    |> Enum.with_index()
    |> Enum.map(fn {window_vectors, window_idx} ->
      vectors_for_analysis = Enum.map(window_vectors, fn {id, vector, timestamp} ->
        {id, vector, %{timestamp: timestamp}}
      end)
      
      %{
        window_index: window_idx,
        start_time: window_vectors |> List.first() |> elem(2),
        end_time: window_vectors |> List.last() |> elem(2),
        vector_count: length(window_vectors),
        centroid: calculate_window_centroid(vectors_for_analysis)
      }
    end)
  end
  
  defp calculate_window_centroid(vectors) do
    vector_data = Enum.map(vectors, fn {_id, vector, _metadata} -> vector end)
    
    case VectorOps.centroid(vector_data) do
      {:error, _} -> nil
      centroid -> centroid
    end
  end
  
  defp analyze_vector_trends(sorted_vectors) do
    if length(sorted_vectors) < 3 do
      %{trend: :insufficient_data}
    else
      # Calculate moving averages of vector components
      first_vector = sorted_vectors |> List.first() |> elem(1)
      dimension = length(first_vector)
      
      # Calculate trend for each dimension
      dimension_trends = 0..(dimension - 1)
      |> Enum.map(fn dim_idx ->
        values = sorted_vectors
        |> Enum.map(fn {_id, vector, _timestamp} -> Enum.at(vector, dim_idx) end)
        
        trend = calculate_trend(values)
        {dim_idx, trend}
      end)
      |> Map.new()
      
      %{
        dimension_trends: dimension_trends,
        overall_trend: calculate_overall_trend(dimension_trends)
      }
    end
  end
  
  defp calculate_trend(values) do
    if length(values) < 2 do
      :stable
    else
      # Simple linear trend calculation
      n = length(values)
      indices = Enum.to_list(0..(n - 1))
      
      mean_x = Enum.sum(indices) / n
      mean_y = Enum.sum(values) / n
      
      numerator = indices
      |> Enum.zip(values)
      |> Enum.reduce(0.0, fn {x, y}, acc -> acc + (x - mean_x) * (y - mean_y) end)
      
      denominator = indices
      |> Enum.reduce(0.0, fn x, acc -> acc + (x - mean_x) * (x - mean_x) end)
      
      slope = if denominator == 0.0, do: 0.0, else: numerator / denominator
      
      cond do
        slope > 0.01 -> :increasing
        slope < -0.01 -> :decreasing
        true -> :stable
      end
    end
  end
  
  defp calculate_overall_trend(dimension_trends) do
    trends = Map.values(dimension_trends)
    
    increasing_count = Enum.count(trends, fn trend -> trend == :increasing end)
    decreasing_count = Enum.count(trends, fn trend -> trend == :decreasing end)
    
    cond do
      increasing_count > decreasing_count -> :increasing
      decreasing_count > increasing_count -> :decreasing
      true -> :stable
    end
  end
  
  defp detect_periodicity(sorted_vectors) do
    # Simple periodicity detection using autocorrelation
    if length(sorted_vectors) < 10 do
      %{periodic: false}
    else
      # Extract time intervals
      time_intervals = sorted_vectors
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [{_id1, _v1, t1}, {_id2, _v2, t2}] -> t2 - t1 end)
      
      # Look for repeated intervals
      interval_frequency = time_intervals
      |> Enum.frequencies()
      |> Enum.sort_by(fn {_interval, freq} -> freq end, :desc)
      
      case interval_frequency do
        [{most_common_interval, frequency} | _] when frequency >= 3 ->
          %{
            periodic: true,
            period_ms: most_common_interval,
            confidence: frequency / length(time_intervals)
          }
        _ ->
          %{periodic: false}
      end
    end
  end
  
  defp get_time_span(sorted_vectors) do
    case sorted_vectors do
      [] -> 0
      [single] -> 0
      vectors ->
        first_timestamp = vectors |> List.first() |> elem(2)
        last_timestamp = vectors |> List.last() |> elem(2)
        last_timestamp - first_timestamp
    end
  end
end