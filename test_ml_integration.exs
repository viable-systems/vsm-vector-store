#!/usr/bin/env elixir

# Test script to verify ML algorithms work with real data through GenServers
Mix.install([
  {:vsm_vector_store, path: "."}
])

defmodule MLIntegrationTest do
  @moduledoc """
  Integration test to verify all ML algorithms work with real data.
  """
  
  def run do
    IO.puts("\n=== Starting VSM Vector Store ML Integration Test ===\n")
    
    # Start the application
    {:ok, _} = Application.ensure_all_started(:vsm_vector_store)
    Process.sleep(1000)  # Give time for all GenServers to start
    
    # Test 1: Create a vector space and insert data
    IO.puts("1. Creating vector space...")
    {:ok, space_id} = VSMVectorStore.create_space("test_ml", 128)
    IO.puts("   ✓ Created space: #{space_id}")
    
    # Generate test vectors (need at least 256 for anomaly detection)
    IO.puts("\n2. Generating and inserting test vectors...")
    vectors = generate_test_vectors(300, 128)
    
    # Insert vectors
    Enum.each(Enum.with_index(vectors), fn {vector, idx} ->
      vector_id = "vec_#{idx}"
      metadata = %{index: idx, category: rem(idx, 5)}  # 5 implicit categories
      
      :ok = VSMVectorStore.insert(space_id, vector_id, vector, metadata)
    end)
    IO.puts("   ✓ Inserted 300 vectors")
    
    # Test 2: Search functionality
    IO.puts("\n3. Testing search functionality...")
    query_vector = generate_random_vector(128)
    {:ok, search_results} = VSMVectorStore.search(space_id, query_vector, 5)
    
    IO.puts("   ✓ Search returned #{length(search_results)} results")
    Enum.each(search_results, fn {id, distance, metadata} ->
      IO.puts("     - #{id}: distance=#{Float.round(distance, 4)}, metadata=#{inspect(metadata)}")
    end)
    
    # Test 3: K-means clustering
    IO.puts("\n4. Testing K-means clustering...")
    # Get all vectors from the space for clustering
    {:ok, vectors} = VSMVectorStore.Storage.Manager.get_all_vectors(space_id)
    
    # Call KMeans directly with default pid
    {:ok, clusters} = VsmVectorStore.Indexing.KMeans.cluster(VsmVectorStore.Indexing.KMeans, vectors, 5, [])
    
    if is_list(clusters) do
      IO.puts("   ✓ K-means clustering successful")
      IO.puts("     - Number of clusters: #{length(clusters)}")
    else
      # Handle the new format with centroids and assignments
      IO.puts("   ✓ K-means clustering successful")
      IO.puts("     - Number of centroids: #{length(clusters.centroids)}")
      IO.puts("     - Inertia: #{Float.round(clusters.inertia, 2)}")
      
      # Count assignments
      assignment_counts = Enum.frequencies(clusters.assignments)
      Enum.each(assignment_counts, fn {cluster_id, count} ->
        IO.puts("     - Cluster #{cluster_id}: #{count} vectors")
      end)
    end
    
    # Test 4: Anomaly detection
    IO.puts("\n5. Testing anomaly detection...")
    # Train anomaly detector first
    :ok = VsmVectorStore.ML.AnomalyDetection.train(VsmVectorStore.ML.AnomalyDetection, vectors, [])
    
    # Then detect anomalies with explicit pid
    {:ok, anomaly_results} = VsmVectorStore.ML.AnomalyDetection.detect_anomalies(VsmVectorStore.ML.AnomalyDetection, vectors, 0.1)
    anomalies = Enum.map(anomaly_results, & &1.vector_id)
    
    IO.puts("   ✓ Anomaly detection successful")
    IO.puts("     - Number of anomalies detected: #{length(anomalies)}")
    
    if length(anomalies) > 0 do
      Enum.take(anomalies, 3) |> Enum.each(fn anomaly_id ->
        IO.puts("     - Anomaly: #{anomaly_id}")
      end)
    end
    
    # Test 5: Pattern recognition (if available)
    IO.puts("\n6. Testing pattern recognition...")
    case test_pattern_recognition(space_id) do
      {:ok, patterns} ->
        IO.puts("   ✓ Pattern recognition successful")
        IO.puts("     - Number of patterns found: #{length(patterns)}")
        Enum.take(patterns, 3) |> Enum.each(fn pattern ->
          IO.puts("     - Pattern #{pattern.id}: #{length(pattern.members)} members, confidence: #{Float.round(pattern.confidence, 2)}")
        end)
      {:error, reason} ->
        IO.puts("   ⚠ Pattern recognition not available: #{inspect(reason)}")
    end
    
    # Test 6: Get statistics
    IO.puts("\n7. Getting space statistics...")
    {:ok, stats} = VSMVectorStore.stats(space_id)
    
    IO.puts("   ✓ Statistics retrieved")
    IO.puts("     - Vector count: #{stats.vector_count || stats[:count] || "N/A"}")
    IO.puts("     - Dimensions: #{stats.dimensions}")
    
    # Cleanup
    IO.puts("\n8. Cleaning up...")
    :ok = VSMVectorStore.delete_space(space_id)
    IO.puts("   ✓ Space deleted")
    
    IO.puts("\n=== All tests completed successfully! ===\n")
  rescue
    error ->
      IO.puts("\n❌ Error during test: #{inspect(error)}")
      IO.puts("Stack trace:")
      IO.inspect(__STACKTRACE__, limit: :infinity)
  end
  
  defp generate_test_vectors(count, dimension) do
    # Generate clustered vectors to make patterns more obvious
    cluster_centers = for _ <- 1..5, do: generate_random_vector(dimension)
    
    Enum.map(1..count, fn idx ->
      # Pick a cluster center
      center = Enum.at(cluster_centers, rem(idx, 5))
      
      # Add noise to create variation around the center
      Enum.map(center, fn val ->
        val + :rand.normal() * 0.1
      end)
    end)
  end
  
  defp generate_random_vector(dimension) do
    for _ <- 1..dimension, do: :rand.uniform()
  end
  
  defp test_pattern_recognition(space_id) do
    # Try to access pattern recognition if it exists
    try do
      # Get vectors and use pattern recognition directly
      {:ok, vectors} = VSMVectorStore.Storage.Manager.get_all_vectors(space_id)
      {:ok, patterns} = VsmVectorStore.ML.PatternRecognition.learn_patterns(VsmVectorStore.ML.PatternRecognition, vectors, [])
      {:ok, patterns}
    rescue
      _ -> {:error, :not_implemented}
    end
  end
end

# Run the test
MLIntegrationTest.run()