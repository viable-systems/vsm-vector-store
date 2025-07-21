defmodule VsmVectorStore.Clustering.KMeansTest do
  @moduledoc """
  Comprehensive unit tests for K-means clustering implementation.
  
  Tests clustering algorithm, initialization strategies, convergence, and edge cases.
  """
  
  use ExUnit.Case, async: true
  use ExUnitProperties
  
  alias VsmVectorStore.TestHelpers
  alias VsmVectorStore.Clustering.KMeans
  
  describe "K-means Initialization" do
    test "random initialization creates k centroids" do
      vectors = TestHelpers.generate_random_vectors(50, 3, seed: 123)
      k = 4
      
      {:ok, pid} = KMeans.start_link([])
      result = KMeans.cluster(pid, vectors, k, init_method: :random)
      
      assert %{centroids: centroids, assignments: assignments} = result
      assert length(centroids) == k
      assert length(assignments) == length(vectors)
      
      # All centroids should have correct dimensionality
      Enum.each(centroids, fn centroid ->
        assert length(centroid) == 3
      end)
      
      # All assignments should be valid cluster IDs
      assert Enum.all?(assignments, fn assignment ->
        assignment >= 0 and assignment < k
      end)
      
      GenServer.stop(pid)
    end
    
    test "k-means++ initialization provides better initial centroids" do
      # Create vectors with clear clusters for better testing
      vectors = TestHelpers.generate_clustered_vectors(150, 2, 3, seed: 456)
      k = 3
      
      {:ok, pid_random} = KMeans.start_link([])
      {:ok, pid_plus_plus} = KMeans.start_link([])
      
      # Test random initialization
      result_random = KMeans.cluster(pid_random, vectors, k, 
        init_method: :random, max_iterations: 10)
      
      # Test k-means++ initialization  
      result_plus_plus = KMeans.cluster(pid_plus_plus, vectors, k,
        init_method: :kmeans_plus_plus, max_iterations: 10)
      
      # K-means++ should generally achieve better initial results
      # (lower initial inertia or faster convergence)
      assert result_plus_plus.iterations <= result_random.iterations + 2
      
      GenServer.stop(pid_random)
      GenServer.stop(pid_plus_plus)
    end
    
    test "farthest first initialization spreads centroids" do
      vectors = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], 
                 [10.0, 10.0], [11.0, 10.0], [10.0, 11.0], [11.0, 11.0]]
      k = 2
      
      {:ok, pid} = KMeans.start_link([])
      result = KMeans.cluster(pid, vectors, k, init_method: :farthest_first)
      
      centroids = result.centroids
      assert length(centroids) == 2
      
      # Centroids should be far apart (representing the two clear clusters)
      distance = TestHelpers.euclidean_distance(hd(centroids), Enum.at(centroids, 1))
      assert distance > 5.0  # Should be much greater than intra-cluster distances
      
      GenServer.stop(pid)
    end
    
    property "initialization methods create valid initial states" do
      check all vectors <- TestHelpers.vector_batch_generator(10, 50, 4),
                k <- integer(2..min(10, length(vectors) - 1)),
                init_method <- member_of([:random, :kmeans_plus_plus, :farthest_first]) do
        
        {:ok, pid} = KMeans.start_link([])
        result = KMeans.cluster(pid, vectors, k, 
          init_method: init_method, max_iterations: 1)  # Just test initialization
        
        assert %{centroids: centroids, assignments: assignments} = result
        assert length(centroids) == k
        assert length(assignments) == length(vectors)
        
        # Verify centroid dimensionality
        expected_dims = length(hd(vectors))
        Enum.each(centroids, fn centroid ->
          assert length(centroid) == expected_dims
        end)
        
        # Verify assignments are valid
        assert Enum.all?(assignments, &(&1 >= 0 and &1 < k))
        
        GenServer.stop(pid)
      end
    end
  end
  
  describe "K-means Clustering Algorithm" do
    test "converges on simple clustered data" do
      # Create two clear clusters
      cluster1 = TestHelpers.generate_vectors_around_center([0.0, 0.0], 25, 0.5)
      cluster2 = TestHelpers.generate_vectors_around_center([5.0, 5.0], 25, 0.5)
      vectors = cluster1 ++ cluster2
      
      {:ok, pid} = KMeans.start_link([])
      result = KMeans.cluster(pid, vectors, 2, tolerance: 1.0e-6)
      
      assert result.iterations < 50  # Should converge quickly on clear clusters
      
      # Validate clustering quality
      inertia = TestHelpers.validate_kmeans_result(vectors, result, 2)
      assert inertia < 100.0  # Should achieve low inertia with clear clusters
      
      GenServer.stop(pid)
    end
    
    test "handles edge cases gracefully" do
      {:ok, pid} = KMeans.start_link([])
      
      # Test with k = 1
      vectors = TestHelpers.generate_random_vectors(10, 3, seed: 789)
      result = KMeans.cluster(pid, vectors, 1)
      
      assert length(result.centroids) == 1
      assert Enum.all?(result.assignments, &(&1 == 0))
      
      # Test with k equal to number of vectors
      small_vectors = TestHelpers.generate_random_vectors(3, 2, seed: 321)
      result = KMeans.cluster(pid, small_vectors, 3)
      
      assert length(result.centroids) == 3
      assert length(Set.new(result.assignments)) == 3  # All different clusters
      
      GenServer.stop(pid)
    end
    
    test "respects maximum iterations parameter" do
      vectors = TestHelpers.generate_random_vectors(100, 5, seed: 654)
      max_iter = 5
      
      {:ok, pid} = KMeans.start_link([])
      result = KMeans.cluster(pid, vectors, 4, 
        max_iterations: max_iter, tolerance: 1.0e-10)  # Very strict tolerance
      
      assert result.iterations <= max_iter
      
      GenServer.stop(pid)
    end
    
    test "achieves convergence within tolerance" do
      vectors = TestHelpers.generate_clustered_vectors(60, 3, 3, seed: 987)
      tolerance = 1.0e-4
      
      {:ok, pid} = KMeans.start_link([])
      
      # Run clustering twice with same parameters
      result1 = KMeans.cluster(pid, vectors, 3, tolerance: tolerance, seed: 111)
      result2 = KMeans.cluster(pid, vectors, 3, tolerance: tolerance, seed: 111)
      
      # Results should be very similar (within tolerance)
      centroids1 = result1.centroids
      centroids2 = result2.centroids
      
      # Sort centroids by first coordinate to compare
      sorted_centroids1 = Enum.sort_by(centroids1, &hd/1)
      sorted_centroids2 = Enum.sort_by(centroids2, &hd/1)
      
      Enum.zip(sorted_centroids1, sorted_centroids2)
      |> Enum.each(fn {c1, c2} ->
        distance = TestHelpers.euclidean_distance(c1, c2)
        assert distance <= tolerance * 10  # Allow some numerical variation
      end)
      
      GenServer.stop(pid)
    end
    
    property "clustering preserves total data characteristics" do
      check all vectors <- TestHelpers.vector_batch_generator(20, 100, 3),
                k <- integer(2..min(8, div(length(vectors), 2))) do
        
        {:ok, pid} = KMeans.start_link([])
        result = KMeans.cluster(pid, vectors, k)
        
        # Calculate overall centroid of original data
        dimensions = length(hd(vectors))
        overall_centroid = 
          0..(dimensions - 1)
          |> Enum.map(fn dim ->
            vectors
            |> Enum.map(&Enum.at(&1, dim))
            |> Enum.sum()
            |> Kernel./(length(vectors))
          end)
        
        # Calculate weighted centroid of clusters
        cluster_counts = 
          result.assignments
          |> Enum.reduce(%{}, fn assignment, acc ->
            Map.update(acc, assignment, 1, &(&1 + 1))
          end)
        
        weighted_centroid = 
          0..(dimensions - 1)
          |> Enum.map(fn dim ->
            result.centroids
            |> Enum.with_index()
            |> Enum.map(fn {centroid, cluster_id} ->
              weight = Map.get(cluster_counts, cluster_id, 0) / length(vectors)
              Enum.at(centroid, dim) * weight
            end)
            |> Enum.sum()
          end)
        
        # Weighted cluster centroid should be close to overall centroid
        TestHelpers.assert_vectors_equal(overall_centroid, weighted_centroid, 0.1)
        
        GenServer.stop(pid)
      end
    end
  end
  
  describe "K-means Performance and Scalability" do
    test "handles large datasets efficiently" do
      large_vectors = TestHelpers.generate_random_vectors(1000, 8, seed: 2468)
      
      {:ok, pid} = KMeans.start_link([])
      
      {result, time_ms} = TestHelpers.measure_time(fn ->
        KMeans.cluster(pid, large_vectors, 5)
      end)
      
      # Should complete within reasonable time (adjust based on hardware)
      TestHelpers.assert_performance_bounds(time_ms, 5000)  # 5 seconds max
      
      # Verify result quality
      assert length(result.centroids) == 5
      assert length(result.assignments) == 1000
      assert result.iterations < 100  # Should converge reasonably fast
      
      GenServer.stop(pid)
    end
    
    test "memory usage scales linearly with data size" do
      base_size = 100
      dimensions = 4
      
      # Test with different dataset sizes
      memory_usage = [100, 200, 400]
      |> Enum.map(fn size ->
        vectors = TestHelpers.generate_random_vectors(size, dimensions, seed: size)
        
        {_result, memory_used} = TestHelpers.measure_memory(fn ->
          {:ok, pid} = KMeans.start_link([])
          result = KMeans.cluster(pid, vectors, 3)
          GenServer.stop(pid)
          result
        end)
        
        {size, memory_used}
      end)
      
      # Memory should scale roughly linearly (not quadratically)
      [{_, mem1}, {_, mem2}, {_, mem3}] = memory_usage
      
      # 4x data should not use more than 8x memory (allowing overhead)
      assert mem3 <= mem1 * 8
    end
    
    test "concurrent clustering operations are safe" do
      vectors = TestHelpers.generate_random_vectors(100, 3, seed: 1357)
      
      tasks = 1..5
      |> Enum.map(fn i ->
        Task.async(fn ->
          {:ok, pid} = KMeans.start_link([])
          result = KMeans.cluster(pid, vectors, 3, seed: i * 100)
          GenServer.stop(pid)
          result
        end)
      end)
      
      results = Task.await_many(tasks, 10000)
      
      # All tasks should complete successfully
      assert length(results) == 5
      Enum.each(results, fn result ->
        assert %{centroids: centroids, assignments: assignments} = result
        assert length(centroids) == 3
        assert length(assignments) == 100
      end)
    end
  end
  
  describe "K-means Quality Metrics" do
    test "calculates silhouette score correctly" do
      # Create data with clear clusters for high silhouette score
      vectors = TestHelpers.generate_clustered_vectors(90, 2, 3, seed: 1111)
      
      {:ok, pid} = KMeans.start_link([])
      result = KMeans.cluster(pid, vectors, 3, calculate_silhouette: true)
      
      # Should have good silhouette score for clear clusters
      assert Map.has_key?(result, :silhouette_score)
      assert result.silhouette_score > 0.5  # Good clustering
      
      GenServer.stop(pid)
    end
    
    test "calculates within-cluster sum of squares (WCSS)" do
      vectors = TestHelpers.generate_random_vectors(50, 4, seed: 2222)
      
      {:ok, pid} = KMeans.start_link([])
      result = KMeans.cluster(pid, vectors, 4)
      
      # Calculate WCSS manually to verify
      manual_wcss = calculate_manual_wcss(vectors, result.centroids, result.assignments)
      
      assert Map.has_key?(result, :inertia)
      assert_in_delta(result.inertia, manual_wcss, 1.0e-6)
      
      GenServer.stop(pid)
    end
    
    test "elbow method helps determine optimal k" do
      vectors = TestHelpers.generate_clustered_vectors(120, 3, 4, seed: 3333)
      
      # Test k values from 1 to 8
      wcss_values = 1..8
      |> Enum.map(fn k ->
        {:ok, pid} = KMeans.start_link([])
        result = KMeans.cluster(pid, vectors, k, max_iterations: 50)
        GenServer.stop(pid)
        {k, result.inertia}
      end)
      
      inertia_values = Enum.map(wcss_values, &elem(&1, 1))
      
      # WCSS should decrease as k increases
      assert inertia_values == Enum.sort(inertia_values, :desc)
      
      # Find the "elbow" - should be around k=4 for our test data
      elbow_k = find_elbow_point(wcss_values)
      assert elbow_k >= 3 and elbow_k <= 5  # Should be close to true k=4
    end
    
    property "clustering quality metrics are consistent" do
      check all vectors <- TestHelpers.vector_batch_generator(30, 80, 3),
                k <- integer(2..min(6, div(length(vectors), 3))) do
        
        {:ok, pid} = KMeans.start_link([])
        result = KMeans.cluster(pid, vectors, k, calculate_silhouette: true)
        
        # Verify metrics are in expected ranges
        assert result.inertia >= 0.0
        assert result.silhouette_score >= -1.0 and result.silhouette_score <= 1.0
        assert result.iterations >= 1
        
        # Better clustering should have lower inertia
        if k < div(length(vectors), 3) do
          {:ok, pid2} = KMeans.start_link([])
          result_more_clusters = KMeans.cluster(pid2, vectors, k + 1)
          assert result_more_clusters.inertia <= result.inertia
          GenServer.stop(pid2)
        end
        
        GenServer.stop(pid)
      end
    end
  end
  
  describe "K-means Error Handling and Edge Cases" do
    test "handles empty cluster assignment gracefully" do
      # Create a scenario where a cluster might become empty
      vectors = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [10.0, 10.0]]  # 3 close, 1 far
      
      {:ok, pid} = KMeans.start_link([])
      result = KMeans.cluster(pid, vectors, 3)  # More clusters than natural groups
      
      # Should handle empty clusters by reinitializing
      assert length(result.centroids) == 3
      assert length(result.assignments) == 4
      assert Enum.all?(result.assignments, &(&1 >= 0 and &1 < 3))
      
      GenServer.stop(pid)
    end
    
    test "validates input parameters" do
      {:ok, pid} = KMeans.start_link([])
      vectors = TestHelpers.generate_random_vectors(10, 3)
      
      # Test invalid k values
      assert_raise ArgumentError, fn ->
        KMeans.cluster(pid, vectors, 0)
      end
      
      assert_raise ArgumentError, fn ->
        KMeans.cluster(pid, vectors, -1)
      end
      
      assert_raise ArgumentError, fn ->
        KMeans.cluster(pid, vectors, length(vectors) + 1)
      end
      
      # Test empty vector list
      assert_raise ArgumentError, fn ->
        KMeans.cluster(pid, [], 2)
      end
      
      GenServer.stop(pid)
    end
    
    test "handles vectors with different dimensionalities" do
      {:ok, pid} = KMeans.start_link([])
      
      # Mixed dimensionality vectors should raise error
      bad_vectors = [[1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0]]
      
      assert_raise ArgumentError, fn ->
        KMeans.cluster(pid, bad_vectors, 2)
      end
      
      GenServer.stop(pid)
    end
    
    test "handles numerical edge cases" do
      {:ok, pid} = KMeans.start_link([])
      
      # Test with very small values
      tiny_vectors = [[1.0e-10, 2.0e-10], [3.0e-10, 4.0e-10], [5.0e-10, 6.0e-10]]
      result = KMeans.cluster(pid, tiny_vectors, 2)
      assert length(result.centroids) == 2
      
      # Test with very large values
      huge_vectors = [[1.0e10, 2.0e10], [3.0e10, 4.0e10], [5.0e10, 6.0e10]]
      result = KMeans.cluster(pid, huge_vectors, 2)
      assert length(result.centroids) == 2
      
      GenServer.stop(pid)
    end
  end
  
  # Helper functions
  
  defp calculate_manual_wcss(vectors, centroids, assignments) do
    vectors
    |> Enum.zip(assignments)
    |> Enum.map(fn {vector, cluster_id} ->
      centroid = Enum.at(centroids, cluster_id)
      distance = TestHelpers.euclidean_distance(vector, centroid)
      distance * distance
    end)
    |> Enum.sum()
  end
  
  defp find_elbow_point(wcss_values) do
    # Simple elbow detection using second derivative
    second_derivatives = wcss_values
    |> Enum.map(&elem(&1, 1))
    |> calculate_second_derivatives()
    
    # Find k with maximum second derivative (steepest change in slope)
    {max_k, _} = 
      second_derivatives
      |> Enum.with_index(3)  # Start from k=3 (index offset)
      |> Enum.max_by(&elem(&1, 0))
    
    max_k
  end
  
  defp calculate_second_derivatives(values) when length(values) >= 3 do
    values
    |> Enum.chunk_every(3, 1, :discard)
    |> Enum.map(fn [a, b, c] -> a - 2*b + c end)
  end
  defp calculate_second_derivatives(_), do: []
end