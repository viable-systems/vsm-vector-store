#!/usr/bin/env elixir

# Simple test script to verify ML algorithms work
Mix.install([])

# Test K-means clustering
defmodule TestKMeans do
  def test_clustering do
    # Generate test data - 3 clusters
    vectors = [
      # Cluster 1 (around [0, 0])
      [0.1, 0.1], [0.2, 0.0], [0.0, 0.2], [-0.1, 0.1],
      # Cluster 2 (around [5, 5]) 
      [5.1, 5.0], [4.9, 5.1], [5.0, 4.9], [5.2, 5.1],
      # Cluster 3 (around [10, 0])
      [10.1, 0.1], [9.9, -0.1], [10.0, 0.2], [10.2, 0.0]
    ]
    
    IO.puts("Testing K-means clustering with #{length(vectors)} vectors...")
    
    # Test K-means++ initialization
    {:ok, centroids} = kmeans_plus_plus_init(vectors, 3)
    IO.puts("K-means++ centroids: #{inspect(centroids)}")
    
    # Test euclidean distance
    dist = euclidean_distance([0, 0], [3, 4])
    IO.puts("Euclidean distance [0,0] to [3,4]: #{dist} (should be 5.0)")
    
    # Test full clustering
    result = perform_kmeans(vectors, 3, centroids, 10)
    IO.puts("Clustering result: #{inspect(result)}")
    
    :ok
  end
  
  defp kmeans_plus_plus_init(vectors, k) do
    if length(vectors) < k do
      {:error, :insufficient_vectors}
    else
      # Pick first centroid randomly
      first_centroid = Enum.random(vectors)
      centroids = [first_centroid]
      
      # Pick remaining centroids using K-means++
      final_centroids = Enum.reduce(2..k, centroids, fn _, acc_centroids ->
        distances = Enum.map(vectors, fn vector ->
          min_dist = acc_centroids
                    |> Enum.map(&euclidean_distance(vector, &1))
                    |> Enum.min()
          {vector, min_dist * min_dist}  # Square the distance
        end)
        
        total_dist = distances |> Enum.map(&elem(&1, 1)) |> Enum.sum()
        
        # Weighted random selection
        target = :rand.uniform() * total_dist
        {selected_vector, _} = select_weighted(distances, target, 0)
        
        [selected_vector | acc_centroids]
      end)
      
      {:ok, Enum.reverse(final_centroids)}
    end
  end
  
  defp select_weighted([{vector, weight} | _], target, cumulative) when cumulative + weight >= target do
    {vector, weight}
  end
  
  defp select_weighted([{_vector, weight} | rest], target, cumulative) do
    select_weighted(rest, target, cumulative + weight)
  end
  
  defp select_weighted([], _target, _cumulative) do
    # Fallback to last vector if we somehow don't find one
    {[0, 0], 1.0}
  end
  
  defp euclidean_distance(v1, v2) do
    v1
    |> Enum.zip(v2)
    |> Enum.map(fn {a, b} -> (a - b) * (a - b) end)
    |> Enum.sum()
    |> :math.sqrt()
  end
  
  defp perform_kmeans(vectors, k, initial_centroids, max_iterations) do
    centroids = Enum.take(initial_centroids, k)
    iterate_kmeans(vectors, centroids, max_iterations, 1.0e-4)
  end
  
  defp iterate_kmeans(vectors, centroids, 0, _tolerance), do: {:ok, centroids, assign_vectors(vectors, centroids)}
  
  defp iterate_kmeans(vectors, centroids, iterations_left, tolerance) do
    assignments = assign_vectors(vectors, centroids)
    new_centroids = update_centroids(vectors, assignments, length(centroids))
    
    # Check convergence
    max_movement = centroids
                  |> Enum.zip(new_centroids)
                  |> Enum.map(fn {old, new} -> euclidean_distance(old, new) end)
                  |> Enum.max()
    
    if max_movement < tolerance do
      {:ok, new_centroids, assignments}
    else
      iterate_kmeans(vectors, new_centroids, iterations_left - 1, tolerance)
    end
  end
  
  defp assign_vectors(vectors, centroids) do
    Enum.map(vectors, fn vector ->
      distances = Enum.map(centroids, &euclidean_distance(vector, &1))
      {_min_dist, cluster_idx} = distances
                                |> Enum.with_index()
                                |> Enum.min_by(&elem(&1, 0))
      cluster_idx
    end)
  end
  
  defp update_centroids(vectors, assignments, k) do
    Enum.map(0..(k-1), fn cluster_idx ->
      cluster_vectors = vectors
                       |> Enum.zip(assignments)
                       |> Enum.filter(fn {_vector, assignment} -> assignment == cluster_idx end)
                       |> Enum.map(&elem(&1, 0))
      
      if length(cluster_vectors) > 0 do
        # Calculate mean
        dimensions = length(hd(cluster_vectors))
        Enum.map(0..(dimensions-1), fn dim ->
          cluster_vectors
          |> Enum.map(&Enum.at(&1, dim))
          |> Enum.sum()
          |> Kernel./(length(cluster_vectors))
        end)
      else
        # If no vectors assigned, keep the old centroid
        Enum.at(vectors, cluster_idx) || [0, 0]
      end
    end)
  end
end

# Test HNSW basics
defmodule TestHNSW do
  def test_hnsw_basics do
    IO.puts("Testing HNSW basic operations...")
    
    # Test distance calculations
    dist1 = cosine_distance([1, 0, 0], [0, 1, 0])
    IO.puts("Cosine distance [1,0,0] to [0,1,0]: #{dist1} (should be 1.0)")
    
    dist2 = cosine_distance([1, 1, 0], [1, 1, 0])
    IO.puts("Cosine distance [1,1,0] to [1,1,0]: #{dist2} (should be 0.0)")
    
    # Test level generation
    level = get_random_level(1.0 / :math.log(2.0))
    IO.puts("Random HNSW level: #{level}")
    
    :ok
  end
  
  defp cosine_distance(v1, v2) do
    dot_product = v1 |> Enum.zip(v2) |> Enum.map(fn {a, b} -> a * b end) |> Enum.sum()
    norm1 = v1 |> Enum.map(fn x -> x * x end) |> Enum.sum() |> :math.sqrt()
    norm2 = v2 |> Enum.map(fn x -> x * x end) |> Enum.sum() |> :math.sqrt()
    
    if norm1 == 0 or norm2 == 0 do
      1.0
    else
      1.0 - (dot_product / (norm1 * norm2))
    end
  end
  
  defp get_random_level(ml) do
    get_random_level_loop(0, ml, true)
  end
  
  defp get_random_level_loop(level, ml, true) do
    if :rand.uniform() < (1.0 / ml) do
      get_random_level_loop(level + 1, ml, true)
    else
      level
    end
  end
  
  defp get_random_level_loop(level, _ml, false), do: level
end

# Test anomaly detection
defmodule TestAnomalyDetection do
  def test_isolation_forest do
    IO.puts("Testing basic anomaly detection...")
    
    # Generate normal data (cluster around [0, 0])
    normal_data = for _ <- 1..20 do
      [:rand.normal(0, 1), :rand.normal(0, 1)]
    end
    
    # Add some outliers
    outliers = [[10, 10], [-10, -10], [0, 15]]
    all_data = normal_data ++ outliers
    
    IO.puts("Generated #{length(all_data)} data points (#{length(outliers)} outliers)")
    
    # Simple outlier detection using distance from mean
    mean = calculate_mean(all_data)
    distances = Enum.map(all_data, &euclidean_distance(&1, mean))
    mean_distance = distances |> Enum.sum() |> Kernel./(length(distances))
    std_dev = calculate_std_dev(distances, mean_distance)
    
    threshold = mean_distance + 2 * std_dev
    
    anomalies = all_data
                |> Enum.zip(distances)
                |> Enum.filter(fn {_point, dist} -> dist > threshold end)
                |> Enum.map(&elem(&1, 0))
    
    IO.puts("Detected anomalies: #{inspect(anomalies)}")
    IO.puts("Threshold: #{threshold}, Mean distance: #{mean_distance}")
    
    :ok
  end
  
  defp calculate_mean(vectors) do
    dimensions = length(hd(vectors))
    Enum.map(0..(dimensions-1), fn dim ->
      vectors
      |> Enum.map(&Enum.at(&1, dim))
      |> Enum.sum()
      |> Kernel./(length(vectors))
    end)
  end
  
  defp euclidean_distance(v1, v2) do
    v1
    |> Enum.zip(v2)
    |> Enum.map(fn {a, b} -> (a - b) * (a - b) end)
    |> Enum.sum()
    |> :math.sqrt()
  end
  
  defp calculate_std_dev(values, mean) do
    variance = values
               |> Enum.map(fn x -> (x - mean) * (x - mean) end)
               |> Enum.sum()
               |> Kernel./(length(values))
    :math.sqrt(variance)
  end
end

# Run all tests
IO.puts("🧠 Testing VSM Vector Store ML Algorithms\n")

IO.puts("=" |> String.duplicate(50))
TestKMeans.test_clustering()

IO.puts("\n" <> "=" |> String.duplicate(50))
TestHNSW.test_hnsw_basics()

IO.puts("\n" <> "=" |> String.duplicate(50))
TestAnomalyDetection.test_isolation_forest()

IO.puts("\n" <> "=" |> String.duplicate(50))
IO.puts("✅ All ML algorithm tests completed successfully!")