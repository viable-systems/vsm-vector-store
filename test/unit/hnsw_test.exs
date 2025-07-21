defmodule VsmVectorStore.HNSW.Test do
  @moduledoc """
  Comprehensive unit tests for HNSW (Hierarchical Navigable Small World) implementation.
  
  Tests graph construction, insertion, search algorithms, and performance characteristics.
  """
  
  use ExUnit.Case, async: true
  use ExUnitProperties
  
  alias VsmVectorStore.TestHelpers
  alias VsmVectorStore.HNSW.{Graph, Index, Search}
  
  describe "HNSW Graph Construction" do
    test "creates empty graph with correct initial state" do
      graph = Graph.new(dimensions: 128, max_connections: 16, ml_constant: 1/Math.log(2))
      
      assert graph.layers == %{}
      assert graph.entry_points == %{}
      assert graph.node_metadata == %{}
      assert graph.connections == %{}
      assert graph.dimensions == 128
      assert graph.max_connections == 16
    end
    
    test "inserts single vector correctly" do
      graph = Graph.new(dimensions: 3, max_connections: 8, ml_constant: 1/Math.log(2))
      vector = [1.0, 2.0, 3.0]
      
      {:ok, updated_graph} = Graph.insert_vector(graph, 0, vector, %{label: "test"})
      
      assert Map.has_key?(updated_graph.layers, 0)
      assert updated_graph.node_metadata[0][:vector] == vector
      assert updated_graph.node_metadata[0][:metadata][:label] == "test"
    end
    
    test "maintains layer structure with multiple insertions" do
      graph = Graph.new(dimensions: 2, max_connections: 4, ml_constant: 1/Math.log(2))
      vectors = TestHelpers.generate_random_vectors(20, 2, seed: 42)
      
      final_graph = 
        vectors
        |> Enum.with_index()
        |> Enum.reduce(graph, fn {vector, id}, acc_graph ->
          {:ok, updated} = Graph.insert_vector(acc_graph, id, vector)
          updated
        end)
      
      # Check that we have at least layer 0
      assert Map.has_key?(final_graph.layers, 0)
      assert length(Map.get(final_graph.layers, 0)) == 20
      
      # Check connections exist and respect max_connections limit
      Enum.each(final_graph.connections, fn {_layer, layer_connections} ->
        Enum.each(layer_connections, fn {_node_id, connections} ->
          assert length(connections) <= 4
        end)
      end)
    end
    
    property "inserted vectors maintain distance relationships" do
      check all vectors <- TestHelpers.vector_batch_generator(5, 15, 3),
                max_connections <- integer(2..8) do
        
        graph = Graph.new(dimensions: 3, max_connections: max_connections, ml_constant: 1/Math.log(2))
        
        final_graph = 
          vectors
          |> Enum.with_index()
          |> Enum.reduce(graph, fn {vector, id}, acc ->
            {:ok, updated} = Graph.insert_vector(acc, id, vector)
            updated
          end)
        
        # Verify all vectors are stored correctly
        stored_vectors = 
          0..(length(vectors) - 1)
          |> Enum.map(fn id ->
            final_graph.node_metadata[id][:vector]
          end)
        
        assert stored_vectors == vectors
        
        # Verify connections respect distance constraints
        layer_0_connections = Map.get(final_graph.connections, 0, %{})
        
        Enum.each(layer_0_connections, fn {node_id, neighbors} ->
          node_vector = final_graph.node_metadata[node_id][:vector]
          
          neighbor_distances = 
            neighbors
            |> Enum.map(fn neighbor_id ->
              neighbor_vector = final_graph.node_metadata[neighbor_id][:vector]
              TestHelpers.euclidean_distance(node_vector, neighbor_vector)
            end)
            |> Enum.sort()
          
          # Neighbors should generally be closer than random vectors
          # (This is a heuristic check - HNSW is approximate)
          if length(neighbor_distances) > 1 do
            assert hd(neighbor_distances) < Enum.sum(neighbor_distances) / length(neighbor_distances)
          end
        end)
      end
    end
  end
  
  describe "HNSW Search Algorithm" do
    setup do
      # Create a test graph with known structure
      vectors = TestHelpers.generate_clustered_vectors(100, 4, 3, seed: 123)
      
      graph = Graph.new(dimensions: 4, max_connections: 16, ml_constant: 1/Math.log(2))
      
      final_graph = 
        vectors
        |> Enum.with_index()
        |> Enum.reduce(graph, fn {vector, id}, acc ->
          {:ok, updated} = Graph.insert_vector(acc, id, vector)
          updated
        end)
      
      %{graph: final_graph, vectors: vectors}
    end
    
    test "finds exact match when query vector exists in graph", %{graph: graph, vectors: vectors} do
      query_vector = Enum.at(vectors, 42)
      
      results = Search.search_knn(graph, query_vector, 1)
      
      assert length(results) == 1
      result = hd(results)
      assert result.id == 42
      assert_in_delta(result.distance, 0.0, 1.0e-10)
    end
    
    test "returns k nearest neighbors", %{graph: graph, vectors: vectors} do
      query_vector = [0.0, 0.0, 0.0, 0.0]  # Origin query
      k = 5
      
      results = Search.search_knn(graph, query_vector, k)
      
      assert length(results) == k
      TestHelpers.validate_hnsw_search(query_vector, results, vectors, k)
    end
    
    test "search results are sorted by distance", %{graph: graph} do
      query_vector = TestHelpers.generate_random_vectors(1, 4, seed: 999) |> hd()
      k = 10
      
      results = Search.search_knn(graph, query_vector, k)
      distances = Enum.map(results, & &1.distance)
      
      assert distances == Enum.sort(distances)
      assert hd(distances) <= List.last(distances)
    end
    
    test "handles edge cases gracefully", %{graph: graph} do
      # Test with k larger than graph size
      query_vector = [0.0, 0.0, 0.0, 0.0]
      
      results = Search.search_knn(graph, query_vector, 200)
      assert length(results) <= 100  # Can't return more than what exists
      
      # Test with k = 0
      results = Search.search_knn(graph, query_vector, 0)
      assert results == []
    end
    
    property "search quality improves with higher ef parameter" do
      check all query_vector <- TestHelpers.vector_generator(4),
                k <- integer(1..10) do
        
        vectors = TestHelpers.generate_random_vectors(50, 4, seed: 456)
        graph = build_test_graph(vectors, 4)
        
        # Test with different ef parameters
        results_low_ef = Search.search_knn(graph, query_vector, k, ef: k)
        results_high_ef = Search.search_knn(graph, query_vector, k, ef: k * 3)
        
        # Higher ef should generally find better (closer) results
        if length(results_low_ef) == k and length(results_high_ef) == k do
          avg_distance_low = Enum.map(results_low_ef, & &1.distance) |> Enum.sum() / k
          avg_distance_high = Enum.map(results_high_ef, & &1.distance) |> Enum.sum() / k
          
          # High ef should find same or better results
          assert avg_distance_high <= avg_distance_low * 1.1  # Allow 10% tolerance
        end
      end
    end
  end
  
  describe "HNSW Performance Characteristics" do
    test "search time scales logarithmically with dataset size" do
      dimensions = 8
      base_size = 100
      
      # Test with different dataset sizes
      sizes_and_times = [100, 500, 1000, 2000]
      |> Enum.map(fn size ->
        vectors = TestHelpers.generate_random_vectors(size, dimensions, seed: size)
        graph = build_test_graph(vectors, dimensions)
        query = TestHelpers.generate_random_vectors(1, dimensions, seed: 9999) |> hd()
        
        {_result, time_ms} = TestHelpers.measure_time(fn ->
          Search.search_knn(graph, query, 10)
        end)
        
        {size, time_ms}
      end)
      
      # Extract times and check scaling
      times = Enum.map(sizes_and_times, &elem(&1, 1))
      
      # Time should not increase dramatically (no more than 10x for 20x data)
      max_time = Enum.max(times)
      min_time = Enum.min(times)
      
      assert max_time <= min_time * 10
    end
    
    test "memory usage is reasonable for large graphs" do
      vectors = TestHelpers.generate_random_vectors(1000, 16)
      
      {graph, memory_used} = TestHelpers.measure_memory(fn ->
        build_test_graph(vectors, 16)
      end)
      
      # Rough memory estimate: should not exceed 10x the raw vector data
      vector_memory = 1000 * 16 * 8  # count * dims * bytes_per_float
      assert memory_used <= vector_memory * 10
      
      # Graph should have expected structure
      assert Map.has_key?(graph.layers, 0)
      assert length(Map.get(graph.layers, 0)) == 1000
    end
    
    test "concurrent search operations are safe" do
      vectors = TestHelpers.generate_random_vectors(200, 4, seed: 777)
      graph = build_test_graph(vectors, 4)
      
      # Run multiple searches concurrently
      tasks = 1..10
      |> Enum.map(fn i ->
        Task.async(fn ->
          query = TestHelpers.generate_random_vectors(1, 4, seed: i * 100) |> hd()
          Search.search_knn(graph, query, 5)
        end)
      end)
      
      results = Task.await_many(tasks, 5000)
      
      # All searches should complete successfully
      assert length(results) == 10
      Enum.each(results, fn result ->
        assert is_list(result)
        assert length(result) <= 5
        Enum.each(result, fn item ->
          assert Map.has_key?(item, :id)
          assert Map.has_key?(item, :distance)
        end)
      end)
    end
  end
  
  describe "HNSW Graph Maintenance" do
    test "handles node deletion correctly" do
      vectors = TestHelpers.generate_random_vectors(50, 3, seed: 888)
      graph = build_test_graph(vectors, 3)
      
      # Delete a node and verify graph integrity
      {:ok, updated_graph} = Graph.delete_vector(graph, 25)
      
      # Node should be removed from all layers
      Enum.each(updated_graph.layers, fn {_layer, nodes} ->
        refute Enum.member?(nodes, 25)
      end)
      
      # Node should be removed from all connection lists
      Enum.each(updated_graph.connections, fn {_layer, layer_connections} ->
        refute Map.has_key?(layer_connections, 25)
        
        Enum.each(layer_connections, fn {_node_id, neighbors} ->
          refute Enum.member?(neighbors, 25)
        end)
      end)
      
      # Metadata should be cleaned up
      refute Map.has_key?(updated_graph.node_metadata, 25)
    end
    
    test "rebuilds graph structure when needed" do
      vectors = TestHelpers.generate_random_vectors(30, 2, seed: 555)
      graph = build_test_graph(vectors, 2)
      
      # Get initial performance
      query = [0.0, 0.0]
      {initial_results, initial_time} = TestHelpers.measure_time(fn ->
        Search.search_knn(graph, query, 5)
      end)
      
      # Rebuild graph
      {:ok, rebuilt_graph} = Graph.rebuild(graph)
      
      # Test performance after rebuild
      {rebuilt_results, rebuilt_time} = TestHelpers.measure_time(fn ->
        Search.search_knn(rebuilt_graph, query, 5)
      end)
      
      # Results should be similar quality (within 20% distance tolerance)
      if length(initial_results) == length(rebuilt_results) do
        initial_avg_dist = Enum.map(initial_results, & &1.distance) |> Enum.sum() / 5
        rebuilt_avg_dist = Enum.map(rebuilt_results, & &1.distance) |> Enum.sum() / 5
        
        assert rebuilt_avg_dist <= initial_avg_dist * 1.2
      end
    end
  end
  
  describe "HNSW Distance Metrics" do
    test "supports different distance metrics correctly" do
      vectors = [
        [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]
      ]
      
      # Test with Euclidean distance
      graph_euclidean = Graph.new(dimensions: 2, max_connections: 4, 
                                  ml_constant: 1/Math.log(2), distance_metric: :euclidean)
      
      final_euclidean = 
        vectors
        |> Enum.with_index()
        |> Enum.reduce(graph_euclidean, fn {vector, id}, acc ->
          {:ok, updated} = Graph.insert_vector(acc, id, vector)
          updated
        end)
      
      # Test with Cosine distance
      graph_cosine = Graph.new(dimensions: 2, max_connections: 4,
                              ml_constant: 1/Math.log(2), distance_metric: :cosine)
      
      final_cosine =
        vectors
        |> Enum.with_index()
        |> Enum.reduce(graph_cosine, fn {vector, id}, acc ->
          {:ok, updated} = Graph.insert_vector(acc, id, vector)
          updated
        end)
      
      query = [0.5, 0.5]
      
      euclidean_results = Search.search_knn(final_euclidean, query, 2)
      cosine_results = Search.search_knn(final_cosine, query, 2)
      
      # Results might differ due to different distance metrics
      assert length(euclidean_results) == 2
      assert length(cosine_results) == 2
      
      # Verify distance calculations
      Enum.each(euclidean_results, fn %{id: id, distance: distance} ->
        vector = Enum.at(vectors, id)
        expected_distance = TestHelpers.euclidean_distance(query, vector)
        assert_in_delta(distance, expected_distance, 1.0e-6)
      end)
      
      Enum.each(cosine_results, fn %{id: id, distance: distance} ->
        vector = Enum.at(vectors, id)
        expected_distance = TestHelpers.cosine_distance(query, vector)
        assert_in_delta(distance, expected_distance, 1.0e-6)
      end)
    end
  end
  
  # Helper functions
  
  defp build_test_graph(vectors, dimensions) do
    graph = Graph.new(dimensions: dimensions, max_connections: 16, ml_constant: 1/Math.log(2))
    
    vectors
    |> Enum.with_index()
    |> Enum.reduce(graph, fn {vector, id}, acc ->
      {:ok, updated} = Graph.insert_vector(acc, id, vector)
      updated
    end)
  end
end