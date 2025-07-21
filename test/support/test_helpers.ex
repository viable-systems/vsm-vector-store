defmodule VsmVectorStore.TestHelpers do
  @moduledoc """
  Test helpers for VSM Vector Store testing.
  
  Provides utilities for generating test data, validating ML algorithms,
  and performance testing.
  """
  
  use ExUnit.CaseTemplate
  import ExUnit.Assertions
  import StreamData
  
  # Vector generation utilities
  
  @doc """
  Generates random vectors with specified dimensions and properties.
  """
  def generate_random_vectors(count, dimensions, opts \\ []) do
    distribution = Keyword.get(opts, :distribution, :normal)
    scale = Keyword.get(opts, :scale, 1.0)
    seed = Keyword.get(opts, :seed)
    
    if seed, do: :rand.seed(:exsss, {seed, seed, seed})
    
    case distribution do
      :normal ->
        1..count
        |> Enum.map(fn _ ->
          generate_normal_vector(dimensions, scale)
        end)
      
      :uniform ->
        1..count
        |> Enum.map(fn _ ->
          generate_uniform_vector(dimensions, scale)
        end)
      
      :clustered ->
        cluster_count = Keyword.get(opts, :clusters, 3)
        generate_clustered_vectors(count, dimensions, cluster_count, scale)
    end
  end
  
  @doc """
  Generates vectors with known clusters for testing clustering algorithms.
  """
  def generate_clustered_vectors(count, dimensions, cluster_count, scale \\ 1.0) do
    vectors_per_cluster = div(count, cluster_count)
    
    # Generate cluster centers
    centers = generate_random_vectors(cluster_count, dimensions, scale: scale * 3)
    
    centers
    |> Enum.with_index()
    |> Enum.flat_map(fn {center, idx} ->
      actual_count = if idx == cluster_count - 1 do
        count - (vectors_per_cluster * (cluster_count - 1))
      else
        vectors_per_cluster
      end
      
      generate_vectors_around_center(center, actual_count, scale * 0.3)
    end)
  end
  
  defp generate_vectors_around_center(center, count, spread) do
    1..count
    |> Enum.map(fn _ ->
      noise = generate_normal_vector(length(center), spread)
      Enum.zip_with(center, noise, &+/2)
    end)
  end
  
  defp generate_normal_vector(dimensions, scale) do
    1..dimensions
    |> Enum.map(fn _ ->
      # Box-Muller transform for normal distribution
      u1 = :rand.uniform()
      u2 = :rand.uniform()
      z0 = :math.sqrt(-2 * :math.log(u1)) * :math.cos(2 * :math.pi() * u2)
      z0 * scale
    end)
  end
  
  defp generate_uniform_vector(dimensions, scale) do
    1..dimensions
    |> Enum.map(fn _ ->
      (:rand.uniform() - 0.5) * 2 * scale
    end)
  end
  
  # Distance calculation utilities
  
  @doc """
  Calculates Euclidean distance between two vectors.
  """
  def euclidean_distance(v1, v2) when length(v1) == length(v2) do
    v1
    |> Enum.zip(v2)
    |> Enum.map(fn {a, b} -> (a - b) * (a - b) end)
    |> Enum.sum()
    |> :math.sqrt()
  end
  
  @doc """
  Calculates cosine distance between two vectors.
  """
  def cosine_distance(v1, v2) when length(v1) == length(v2) do
    dot_product = Enum.zip_with(v1, v2, &*/2) |> Enum.sum()
    norm1 = vector_norm(v1)
    norm2 = vector_norm(v2)
    
    if norm1 == 0 or norm2 == 0 do
      1.0  # Maximum distance for zero vectors
    else
      1.0 - (dot_product / (norm1 * norm2))
    end
  end
  
  @doc """
  Calculates the L2 norm of a vector.
  """
  def vector_norm(vector) do
    vector
    |> Enum.map(&(&1 * &1))
    |> Enum.sum()
    |> :math.sqrt()
  end
  
  # Validation utilities for ML algorithms
  
  @doc """
  Validates K-means clustering results.
  """
  def validate_kmeans_result(vectors, result, k) do
    assert %{centroids: centroids, assignments: assignments} = result
    assert length(centroids) == k
    assert length(assignments) == length(vectors)
    
    # Check all assignments are valid cluster IDs
    assert Enum.all?(assignments, fn assignment ->
      assignment >= 0 and assignment < k
    end)
    
    # Check centroids have correct dimensionality
    expected_dims = length(hd(vectors))
    assert Enum.all?(centroids, fn centroid ->
      length(centroid) == expected_dims
    end)
    
    # Calculate and return inertia (within-cluster sum of squares)
    calculate_inertia(vectors, centroids, assignments)
  end
  
  defp calculate_inertia(vectors, centroids, assignments) do
    vectors
    |> Enum.zip(assignments)
    |> Enum.map(fn {vector, cluster_id} ->
      centroid = Enum.at(centroids, cluster_id)
      euclidean_distance(vector, centroid) |> :math.pow(2)
    end)
    |> Enum.sum()
  end
  
  @doc """
  Validates HNSW search results.
  """
  def validate_hnsw_search(query, results, vectors, k) do
    assert length(results) <= k
    
    # Check result format
    Enum.each(results, fn result ->
      assert %{id: id, distance: distance} = result
      assert is_integer(id)
      assert is_float(distance)
      assert distance >= 0.0
    end)
    
    # Check distances are sorted (ascending)
    distances = Enum.map(results, & &1.distance)
    assert distances == Enum.sort(distances)
    
    # Validate distances by recalculating
    Enum.each(results, fn %{id: id, distance: reported_distance} ->
      vector = Enum.at(vectors, id)
      actual_distance = euclidean_distance(query, vector)
      assert_in_delta(reported_distance, actual_distance, 0.001)
    end)
  end
  
  @doc """
  Validates anomaly detection results.
  """
  def validate_anomaly_detection(vectors, anomalies, expected_rate \\ 0.1) do
    total_vectors = length(vectors)
    anomaly_count = length(anomalies)
    actual_rate = anomaly_count / total_vectors
    
    # Check anomaly rate is reasonable (within 50% of expected)
    assert actual_rate <= expected_rate * 1.5
    assert actual_rate >= expected_rate * 0.5
    
    # Check anomaly format
    Enum.each(anomalies, fn anomaly ->
      assert %{vector_id: id, anomaly_score: score} = anomaly
      assert is_integer(id)
      assert is_float(score)
      assert id >= 0 and id < total_vectors
      assert score >= 0.0 and score <= 1.0
    end)
  end
  
  # Performance testing utilities
  
  @doc """
  Measures execution time of a function.
  """
  def measure_time(fun) when is_function(fun, 0) do
    start_time = System.monotonic_time(:microsecond)
    result = fun.()
    end_time = System.monotonic_time(:microsecond)
    duration_ms = (end_time - start_time) / 1000
    
    {result, duration_ms}
  end
  
  @doc """
  Runs performance benchmarks with multiple iterations.
  """
  def benchmark(name, fun, iterations \\ 10) do
    times = 
      1..iterations
      |> Enum.map(fn _ ->
        {_result, time} = measure_time(fun)
        time
      end)
    
    %{
      name: name,
      iterations: iterations,
      min_time: Enum.min(times),
      max_time: Enum.max(times),
      avg_time: Enum.sum(times) / iterations,
      median_time: median(times)
    }
  end
  
  defp median(list) do
    sorted = Enum.sort(list)
    len = length(sorted)
    
    if rem(len, 2) == 0 do
      (Enum.at(sorted, div(len, 2) - 1) + Enum.at(sorted, div(len, 2))) / 2
    else
      Enum.at(sorted, div(len, 2))
    end
  end
  
  # Property-based testing generators
  
  @doc """
  StreamData generator for vectors.
  """
  def vector_generator(dimensions, value_range \\ -10.0..10.0) do
    list_of(float(value_range), length: dimensions)
  end
  
  @doc """
  StreamData generator for vector batches.
  """
  def vector_batch_generator(min_count, max_count, dimensions) do
    bind(integer(min_count..max_count), fn count ->
      list_of(vector_generator(dimensions), length: count)
    end)
  end
  
  @doc """
  StreamData generator for clustering parameters.
  """
  def clustering_params_generator do
    gen all k <- integer(2..10),
            max_iterations <- integer(10..100),
            tolerance <- float(min: 1.0e-6, max: 1.0e-3) do
      %{k: k, max_iterations: max_iterations, tolerance: tolerance}
    end
  end
  
  # Memory testing utilities
  
  @doc """
  Measures memory usage before and after executing a function.
  """
  def measure_memory(fun) when is_function(fun, 0) do
    :erlang.garbage_collect()
    {memory_before, _} = :erlang.process_info(self(), :memory)
    
    result = fun.()
    
    :erlang.garbage_collect()
    {memory_after, _} = :erlang.process_info(self(), :memory)
    
    memory_used = memory_after - memory_before
    {result, memory_used}
  end
  
  # Test data fixtures
  
  @doc """
  Creates test vectors with known properties for algorithm validation.
  """
  def create_test_fixtures() do
    %{
      # Simple 2D vectors for visualization
      simple_2d: [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.1, 0.1]
      ],
      
      # 3D vectors with clear clusters
      clustered_3d: generate_clustered_vectors(300, 3, 3, 2.0),
      
      # High-dimensional sparse vectors
      sparse_vectors: generate_sparse_vectors(100, 50, 0.1),
      
      # Vectors with outliers for anomaly detection
      with_outliers: generate_vectors_with_outliers(200, 5, 10),
      
      # Normalized unit vectors
      unit_vectors: generate_unit_vectors(50, 4)
    }
  end
  
  defp generate_sparse_vectors(count, dimensions, sparsity) do
    non_zero_count = round(dimensions * sparsity)
    
    1..count
    |> Enum.map(fn _ ->
      vector = List.duplicate(0.0, dimensions)
      indices = Enum.take_random(0..(dimensions-1), non_zero_count)
      
      Enum.reduce(indices, vector, fn idx, acc ->
        List.replace_at(acc, idx, :rand.normal() * 2.0)
      end)
    end)
  end
  
  defp generate_vectors_with_outliers(normal_count, dimensions, outlier_count) do
    normal_vectors = generate_random_vectors(normal_count, dimensions, scale: 1.0)
    outlier_vectors = generate_random_vectors(outlier_count, dimensions, scale: 5.0)
    
    normal_vectors ++ outlier_vectors
  end
  
  defp generate_unit_vectors(count, dimensions) do
    1..count
    |> Enum.map(fn _ ->
      vector = generate_normal_vector(dimensions, 1.0)
      norm = vector_norm(vector)
      if norm > 0, do: Enum.map(vector, &(&1 / norm)), else: vector
    end)
  end
  
  # Assertion helpers
  
  @doc """
  Asserts that two vectors are approximately equal within tolerance.
  """
  def assert_vectors_equal(v1, v2, tolerance \\ 1.0e-6) do
    assert length(v1) == length(v2)
    
    Enum.zip(v1, v2)
    |> Enum.with_index()
    |> Enum.each(fn {{a, b}, idx} ->
      assert_in_delta(a, b, tolerance, "Vectors differ at index #{idx}")
    end)
  end
  
  @doc """
  Asserts that algorithm performance is within acceptable bounds.
  """
  def assert_performance_bounds(duration_ms, max_expected_ms) do
    assert duration_ms <= max_expected_ms,
      "Performance test failed: #{duration_ms}ms > #{max_expected_ms}ms"
  end
end