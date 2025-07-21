defmodule VSMVectorStore.ML.AnomalyDetection do
  @moduledoc """
  Isolation Forest implementation for anomaly detection in high-dimensional vector spaces.
  
  The Isolation Forest algorithm isolates anomalies instead of profiling normal data points.
  It works on the principle that anomalies are few and different, making them easier to isolate.
  
  Features:
  - Pure Elixir implementation of Isolation Forest
  - Configurable contamination rate
  - Support for high-dimensional vectors
  - Statistical anomaly scoring
  - Batch and streaming anomaly detection
  - VSM patterns for state management and telemetry
  """
  
  use GenServer
  require Logger
  
  alias VSMVectorStore.Storage.VectorOps
  
  @type vector :: list(float())
  @type vector_id :: binary()
  @type anomaly_score :: float()
  @type isolation_tree :: %{
    feature_index: non_neg_integer(),
    split_value: float(),
    left: isolation_tree() | :leaf,
    right: isolation_tree() | :leaf,
    size: pos_integer()
  }
  @type forest :: list(isolation_tree())
  @type anomaly_result :: %{
    vector_id: vector_id(),
    anomaly_score: anomaly_score(),
    is_anomaly: boolean(),
    confidence: float()
  }
  
  defstruct [
    forest: [],
    tree_count: 100,
    subsample_size: 256,
    contamination: 0.1,
    trained: false,
    feature_count: 0
  ]
  
  ## Public API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Train the isolation forest on a set of vectors.
  """
  @spec train(pid(), list({vector_id(), vector(), map()}), keyword()) :: :ok | {:error, term()}
  def train(pid \\ __MODULE__, vectors, opts \\ []) do
    GenServer.call(pid, {:train, vectors, opts}, 60_000)
  end
  
  @doc """
  Detect anomalies in vectors from a space.
  """
  def detect(space_id, opts \\ []) when is_binary(space_id) do
    contamination = Keyword.get(opts, :contamination)
    
    # Get all vectors from the space
    case VSMVectorStore.Storage.Manager.get_all_vectors(space_id) do
      {:ok, vectors} when length(vectors) > 0 ->
        # Train and detect
        case train(__MODULE__, vectors, opts) do
          :ok ->
            detect_anomalies(__MODULE__, vectors, contamination)
          error ->
            error
        end
      
      {:ok, []} ->
        {:ok, []}
        
      error ->
        error
    end
  end
  
  @doc """
  Detect anomalies in a set of vectors.
  Returns list of vector IDs identified as anomalies.
  """
  @spec detect_anomalies(pid(), list({vector_id(), vector(), map()}), float()) ::
    {:ok, list(anomaly_result())} | {:error, term()}
  def detect_anomalies(pid \\ __MODULE__, vectors, contamination \\ nil) do
    GenServer.call(pid, {:detect_anomalies, vectors, contamination}, 30_000)
  end
  
  @doc """
  Calculate anomaly score for a single vector.
  Returns score between 0 and 1, where values closer to 1 indicate anomalies.
  """
  @spec anomaly_score(pid(), vector()) :: {:ok, float()} | {:error, term()}
  def anomaly_score(pid \\ __MODULE__, vector) do
    GenServer.call(pid, {:anomaly_score, vector})
  end
  
  @doc """
  Detect anomalies in streaming fashion without retraining.
  """
  @spec stream_detect(pid(), vector()) :: {:ok, anomaly_result()} | {:error, term()}
  def stream_detect(pid \\ __MODULE__, vector) do
    GenServer.call(pid, {:stream_detect, vector})
  end
  
  @doc """
  Get forest statistics for analysis.
  """
  @spec get_forest_stats(pid()) :: {:ok, map()} | {:error, term()}
  def get_forest_stats(pid \\ __MODULE__) do
    GenServer.call(pid, :get_forest_stats)
  end
  
  ## GenServer Callbacks
  
  @impl true
  def init(opts) do
    tree_count = Keyword.get(opts, :tree_count, 100)
    subsample_size = Keyword.get(opts, :subsample_size, 256)
    contamination = Keyword.get(opts, :contamination, 0.1)
    
    state = %__MODULE__{
      tree_count: tree_count,
      subsample_size: subsample_size,
      contamination: contamination
    }
    
    Logger.info("Anomaly Detection initialized with #{tree_count} trees, subsample_size: #{subsample_size}")
    {:ok, state}
  end
  
  @impl true
  def handle_call({:train, vectors, opts}, _from, state) do
    start_time = System.monotonic_time()
    
    case train_forest(vectors, state, opts) do
      {:ok, new_state} ->
        duration = System.monotonic_time() - start_time
        :telemetry.execute([:vsm_vector_store, :ml, :anomaly_detection, :train], 
          %{duration: duration, vectors_count: length(vectors), trees_count: state.tree_count}, %{})
        {:reply, :ok, new_state}
        
      {:error, reason} ->
        Logger.error("Anomaly detection training failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:detect_anomalies, vectors, contamination}, _from, state) do
    if not state.trained do
      {:reply, {:error, :not_trained}, state}
    else
      start_time = System.monotonic_time()
      
      actual_contamination = contamination || state.contamination
      
      case detect_anomalies_impl(vectors, actual_contamination, state) do
        {:ok, anomalies} ->
          duration = System.monotonic_time() - start_time
          :telemetry.execute([:vsm_vector_store, :ml, :anomaly_detection, :detect], 
            %{duration: duration, vectors_count: length(vectors), anomalies_count: length(anomalies)}, 
            %{contamination: actual_contamination})
          {:reply, {:ok, anomalies}, state}
          
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end
  
  @impl true
  def handle_call({:anomaly_score, vector}, _from, state) do
    if not state.trained do
      {:reply, {:error, :not_trained}, state}
    else
      case calculate_anomaly_score(vector, state) do
        {:ok, score} ->
          {:reply, {:ok, score}, state}
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end
  
  @impl true
  def handle_call({:stream_detect, vector}, _from, state) do
    if not state.trained do
      {:reply, {:error, :not_trained}, state}
    else
      case calculate_anomaly_score(vector, state) do
        {:ok, score} ->
          is_anomaly = score > get_anomaly_threshold(state.contamination)
          confidence = min(1.0, abs(score - 0.5) * 2.0)
          
          result = %{
            vector_id: "stream_" <> (:crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)),
            anomaly_score: score,
            is_anomaly: is_anomaly,
            confidence: confidence
          }
          
          {:reply, {:ok, result}, state}
          
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end
  
  @impl true
  def handle_call(:get_forest_stats, _from, state) do
    if not state.trained do
      {:reply, {:error, :not_trained}, state}
    else
      stats = calculate_forest_statistics(state)
      {:reply, {:ok, stats}, state}
    end
  end
  
  ## Private Functions
  
  defp train_forest(vectors, state, _opts) do
    if length(vectors) < state.subsample_size do
      {:error, :insufficient_training_data}
    else
      # Extract just the vectors for training
      vector_data = Enum.map(vectors, fn {_id, vector, _metadata} -> vector end)
      
      case vector_data do
        [] ->
          {:error, :no_vectors}
          
        [first_vector | _] ->
          feature_count = length(first_vector)
          
          # Build isolation forest
          forest = build_isolation_forest(vector_data, state.tree_count, state.subsample_size)
          
          new_state = %{state |
            forest: forest,
            trained: true,
            feature_count: feature_count
          }
          
          {:ok, new_state}
      end
    end
  end
  
  defp build_isolation_forest(vectors, tree_count, subsample_size) do
    1..tree_count
    |> Enum.map(fn _tree_index ->
      # Randomly sample vectors for this tree
      sampled_vectors = Enum.take_random(vectors, min(subsample_size, length(vectors)))
      
      # Build isolation tree
      build_isolation_tree(sampled_vectors, 0, calculate_max_height(subsample_size))
    end)
  end
  
  defp build_isolation_tree(vectors, current_height, max_height) 
    when current_height >= max_height or length(vectors) <= 1 do
    # Create leaf node
    %{
      type: :leaf,
      size: length(vectors)
    }
  end
  
  defp build_isolation_tree(vectors, current_height, max_height) do
    # Randomly select feature and split value
    [first_vector | _] = vectors
    feature_count = length(first_vector)
    
    feature_index = :rand.uniform(feature_count) - 1
    
    # Get min and max values for this feature
    feature_values = Enum.map(vectors, fn vector -> Enum.at(vector, feature_index) end)
    min_val = Enum.min(feature_values)
    max_val = Enum.max(feature_values)
    
    if min_val == max_val do
      # All values are the same, create leaf
      %{
        type: :leaf,
        size: length(vectors)
      }
    else
      # Random split value between min and max
      split_value = min_val + :rand.uniform() * (max_val - min_val)
      
      # Split vectors
      {left_vectors, right_vectors} = Enum.split_with(vectors, fn vector ->
        Enum.at(vector, feature_index) < split_value
      end)
      
      # Handle edge case where all vectors go to one side
      {left_vectors, right_vectors} = if Enum.empty?(left_vectors) or Enum.empty?(right_vectors) do
        # Split roughly in half
        split_point = div(length(vectors), 2)
        Enum.split(vectors, split_point)
      else
        {left_vectors, right_vectors}
      end
      
      # Recursively build subtrees
      left_subtree = build_isolation_tree(left_vectors, current_height + 1, max_height)
      right_subtree = build_isolation_tree(right_vectors, current_height + 1, max_height)
      
      %{
        type: :internal,
        feature_index: feature_index,
        split_value: split_value,
        left: left_subtree,
        right: right_subtree,
        size: length(vectors)
      }
    end
  end
  
  defp calculate_max_height(n) when n <= 1, do: 0
  defp calculate_max_height(n) do
    # Height limit: ceiling(log2(n))
    :math.log2(n) |> :math.ceil() |> trunc()
  end
  
  defp detect_anomalies_impl(vectors, contamination, state) do
    # Calculate anomaly scores for all vectors
    scored_vectors = vectors
    |> Enum.map(fn {id, vector, metadata} ->
      case calculate_anomaly_score(vector, state) do
        {:ok, score} ->
          {id, vector, metadata, score}
        {:error, _} ->
          {id, vector, metadata, 0.5}  # Default neutral score
      end
    end)
    
    # Determine threshold based on contamination rate
    threshold = determine_anomaly_threshold(scored_vectors, contamination)
    
    # Classify anomalies
    anomalies = scored_vectors
    |> Enum.filter(fn {_id, _vector, _metadata, score} -> score > threshold end)
    |> Enum.map(fn {id, _vector, _metadata, score} ->
      %{
        vector_id: id,
        anomaly_score: score,
        is_anomaly: true,
        confidence: min(1.0, (score - threshold) * 2.0)
      }
    end)
    |> Enum.sort_by(fn result -> result.anomaly_score end, :desc)
    
    {:ok, anomalies}
  end
  
  defp calculate_anomaly_score(vector, state) do
    if length(vector) != state.feature_count do
      {:error, :dimension_mismatch}
    else
      # Calculate average path length across all trees
      path_lengths = Enum.map(state.forest, fn tree ->
        calculate_path_length(vector, tree, 0)
      end)
      
      average_path_length = Enum.sum(path_lengths) / length(path_lengths)
      
      # Normalize using expected path length for random tree
      expected_path_length = expected_path_length_bst(state.subsample_size)
      
      # Anomaly score: s(x,n) = 2^(-E(h(x))/c(n))
      # where E(h(x)) is average path length and c(n) is expected path length
      anomaly_score = :math.pow(2, -average_path_length / expected_path_length)
      
      {:ok, anomaly_score}
    end
  end
  
  defp calculate_path_length(vector, tree, current_depth) do
    case tree.type do
      :leaf ->
        # For leaf nodes, add expected path length for remaining instances
        current_depth + expected_path_length_bst(tree.size)
        
      :internal ->
        feature_value = Enum.at(vector, tree.feature_index)
        
        if feature_value < tree.split_value do
          calculate_path_length(vector, tree.left, current_depth + 1)
        else
          calculate_path_length(vector, tree.right, current_depth + 1)
        end
    end
  end
  
  defp expected_path_length_bst(n) when n <= 1, do: 0.0
  defp expected_path_length_bst(2), do: 1.0
  defp expected_path_length_bst(n) do
    # c(n) = 2 * H(n-1) - (2 * (n-1) / n)
    # where H(n) is the harmonic number
    harmonic_n_minus_1 = harmonic_number(n - 1)
    2.0 * harmonic_n_minus_1 - (2.0 * (n - 1) / n)
  end
  
  defp harmonic_number(n) when n <= 0, do: 0.0
  defp harmonic_number(n) do
    1..n |> Enum.reduce(0.0, fn i, acc -> acc + 1.0 / i end)
  end
  
  defp determine_anomaly_threshold(scored_vectors, contamination) do
    scores = Enum.map(scored_vectors, fn {_id, _vector, _metadata, score} -> score end)
    sorted_scores = Enum.sort(scores, :desc)
    
    # Find threshold that captures approximately 'contamination' percentage as anomalies
    anomaly_count = max(1, round(length(sorted_scores) * contamination))
    
    case Enum.at(sorted_scores, anomaly_count - 1) do
      nil -> 0.6  # Default threshold
      threshold -> threshold
    end
  end
  
  defp get_anomaly_threshold(contamination) do
    # Simple threshold based on contamination rate
    # In isolation forest, scores > 0.6 are typically considered anomalous
    base_threshold = 0.6
    adjustment = (1.0 - contamination) * 0.1
    base_threshold - adjustment
  end
  
  defp calculate_forest_statistics(state) do
    tree_depths = Enum.map(state.forest, fn tree ->
      calculate_tree_depth(tree)
    end)
    
    tree_sizes = Enum.map(state.forest, fn tree ->
      tree.size
    end)
    
    %{
      total_trees: length(state.forest),
      average_depth: Enum.sum(tree_depths) / length(tree_depths),
      max_depth: Enum.max(tree_depths),
      min_depth: Enum.min(tree_depths),
      average_tree_size: Enum.sum(tree_sizes) / length(tree_sizes),
      feature_count: state.feature_count,
      subsample_size: state.subsample_size,
      contamination: state.contamination
    }
  end
  
  defp calculate_tree_depth(tree) do
    case tree.type do
      :leaf -> 1
      :internal ->
        left_depth = calculate_tree_depth(tree.left)
        right_depth = calculate_tree_depth(tree.right)
        1 + max(left_depth, right_depth)
    end
  end
end