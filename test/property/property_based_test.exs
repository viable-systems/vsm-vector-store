defmodule VsmVectorStore.PropertyBasedTest do
  @moduledoc """
  Property-based tests for VSM Vector Store algorithms.
  
  Uses StreamData to generate test cases and verify mathematical properties,
  invariants, and correctness conditions across the entire ML pipeline.
  """
  
  use ExUnit.Case, async: true
  use ExUnitProperties
  
  alias VsmVectorStore.TestHelpers
  alias VsmVectorStore.{HNSW, Clustering, Quantization, AnomalyDetection}
  
  # Property test configuration
  @default_max_runs 100
  @quick_max_runs 25
  
  describe "Vector Operations Properties" do
    property "vector distance metrics satisfy mathematical properties" do
      check all v1 <- TestHelpers.vector_generator(10),
                v2 <- TestHelpers.vector_generator(10),
                v3 <- TestHelpers.vector_generator(10),
                max_runs: @default_max_runs do
        
        # Test Euclidean distance properties
        euclidean_d12 = TestHelpers.euclidean_distance(v1, v2)
        euclidean_d21 = TestHelpers.euclidean_distance(v2, v1)
        euclidean_d11 = TestHelpers.euclidean_distance(v1, v1)
        euclidean_d13 = TestHelpers.euclidean_distance(v1, v3)
        euclidean_d23 = TestHelpers.euclidean_distance(v2, v3)
        
        # Symmetry: d(a,b) = d(b,a)
        assert_in_delta(euclidean_d12, euclidean_d21, 1.0e-10)
        
        # Identity: d(a,a) = 0
        assert_in_delta(euclidean_d11, 0.0, 1.0e-10)
        
        # Non-negativity: d(a,b) >= 0
        assert euclidean_d12 >= 0.0
        
        # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        assert euclidean_d13 <= euclidean_d12 + euclidean_d23 + 1.0e-10
        
        # Test Cosine distance properties (if vectors are not zero)
        if TestHelpers.vector_norm(v1) > 1.0e-10 and TestHelpers.vector_norm(v2) > 1.0e-10 do
          cosine_d12 = TestHelpers.cosine_distance(v1, v2)
          cosine_d21 = TestHelpers.cosine_distance(v2, v1)
          
          # Symmetry
          assert_in_delta(cosine_d12, cosine_d21, 1.0e-10)
          
          # Bounded: 0 <= cosine_distance <= 2
          assert cosine_d12 >= 0.0
          assert cosine_d12 <= 2.0
        end
      end
    end
    
    property "vector normalization preserves direction" do
      check all vector <- TestHelpers.vector_generator(8, -100.0..100.0),
                max_runs: @default_max_runs do
        
        norm = TestHelpers.vector_norm(vector)
        
        if norm > 1.0e-10 do  # Skip zero vectors
          normalized = Enum.map(vector, &(&1 / norm))
          normalized_norm = TestHelpers.vector_norm(normalized)
          
          # Normalized vector should have unit norm
          assert_in_delta(normalized_norm, 1.0, 1.0e-10)
          
          # Direction should be preserved (cosine distance ~ 0)
          cosine_dist = TestHelpers.cosine_distance(vector, normalized)
          assert_in_delta(cosine_dist, 0.0, 1.0e-10)
        end
      end
    end
    
    property "vector arithmetic operations are consistent" do
      check all v1 <- TestHelpers.vector_generator(6),
                v2 <- TestHelpers.vector_generator(6),
                scalar <- float(min: -10.0, max: 10.0),
                max_runs: @default_max_runs do
        
        # Vector addition is commutative
        sum1 = Enum.zip_with(v1, v2, &+/2)
        sum2 = Enum.zip_with(v2, v1, &+/2)
        TestHelpers.assert_vectors_equal(sum1, sum2, 1.0e-10)
        
        # Scalar multiplication distributes over addition
        scaled_sum = Enum.map(sum1, &(&1 * scalar))
        sum_of_scaled = 
          Enum.zip_with(
            Enum.map(v1, &(&1 * scalar)),
            Enum.map(v2, &(&1 * scalar)),
            &+/2
          )
        TestHelpers.assert_vectors_equal(scaled_sum, sum_of_scaled, 1.0e-10)
        
        # Dot product is commutative
        dot1 = Enum.zip_with(v1, v2, &*/2) |> Enum.sum()
        dot2 = Enum.zip_with(v2, v1, &*/2) |> Enum.sum()
        assert_in_delta(dot1, dot2, 1.0e-10)
      end
    end
  end
  
  describe "HNSW Algorithm Properties" do
    property "HNSW search results respect distance ordering" do
      check all vectors <- TestHelpers.vector_batch_generator(20, 100, 4),
                query_vector <- TestHelpers.vector_generator(4),
                k <- integer(1..min(10, length(vectors))),
                max_runs: @quick_max_runs do
        
        # Build HNSW graph
        graph = build_test_hnsw_graph(vectors, 4)
        
        # Search for k nearest neighbors
        results = HNSW.Search.search_knn(graph, query_vector, k)
        
        # Results should be sorted by distance
        distances = Enum.map(results, & &1.distance)
        sorted_distances = Enum.sort(distances)
        
        assert distances == sorted_distances
        
        # All returned distances should be valid
        Enum.each(distances, fn distance ->
          assert distance >= 0.0
          assert is_float(distance)
          refute Float.nan?(distance)
          refute distance == Float.infinity()
        end)
        
        # Number of results should not exceed k or available vectors
        assert length(results) <= k
        assert length(results) <= length(vectors)
        
        # All vector IDs should be valid
        vector_ids = Enum.map(results, & &1.id)
        assert Enum.all?(vector_ids, &(&1 >= 0 and &1 < length(vectors)))
        assert length(Enum.uniq(vector_ids)) == length(vector_ids)  # No duplicates
      end
    end
    
    property "HNSW insertion preserves graph connectivity" do
      check all initial_vectors <- TestHelpers.vector_batch_generator(10, 30, 3),
                new_vectors <- TestHelpers.vector_batch_generator(5, 15, 3),
                max_runs: @quick_max_runs do
        
        # Build initial graph
        graph = build_test_hnsw_graph(initial_vectors, 3)
        
        # Insert new vectors one by one
        final_graph = 
          new_vectors
          |> Enum.with_index(length(initial_vectors))
          |> Enum.reduce(graph, fn {vector, id}, acc_graph ->
            {:ok, updated_graph} = HNSW.Graph.insert_vector(acc_graph, id, vector)
            updated_graph
          end)
        
        total_vectors = length(initial_vectors) + length(new_vectors)
        
        # Graph should contain all vectors
        assert Map.has_key?(final_graph.layers, 0)
        layer_0_nodes = Map.get(final_graph.layers, 0)
        assert length(layer_0_nodes) == total_vectors
        
        # All nodes should have valid connections
        layer_0_connections = Map.get(final_graph.connections, 0, %{})
        
        Enum.each(0..(total_vectors - 1), fn node_id ->
          if Map.has_key?(layer_0_connections, node_id) do
            connections = layer_0_connections[node_id]
            
            # Connections should be within graph bounds
            assert Enum.all?(connections, &(&1 >= 0 and &1 < total_vectors))
            
            # No self-connections
            refute Enum.member?(connections, node_id)
            
            # Connection count should respect max_connections
            assert length(connections) <= final_graph.max_connections
          end
        end)
      end
    end
    
    property "HNSW search approximates exact nearest neighbors" do
      check all vectors <- TestHelpers.vector_batch_generator(30, 80, 5),
                query_vector <- TestHelpers.vector_generator(5),
                k <- integer(3..min(10, length(vectors))),
                max_runs: @quick_max_runs do
        
        # Calculate exact k nearest neighbors using brute force
        exact_neighbors = 
          vectors
          |> Enum.with_index()
          |> Enum.map(fn {vector, idx} ->
            {idx, TestHelpers.euclidean_distance(query_vector, vector)}
          end)
          |> Enum.sort_by(&elem(&1, 1))
          |> Enum.take(k)
        
        exact_distances = Enum.map(exact_neighbors, &elem(&1, 1))
        exact_ids = Enum.map(exact_neighbors, &elem(&1, 0))
        
        # Get HNSW approximate results
        graph = build_test_hnsw_graph(vectors, 5)
        hnsw_results = HNSW.Search.search_knn(graph, query_vector, k)
        
        hnsw_distances = Enum.map(hnsw_results, & &1.distance)
        hnsw_ids = Enum.map(hnsw_results, & &1.id)
        
        # HNSW should return the requested number of results
        assert length(hnsw_results) == k
        
        # Calculate recall (how many exact neighbors were found)
        recall = 
          MapSet.intersection(MapSet.new(exact_ids), MapSet.new(hnsw_ids))
          |> MapSet.size()
          |> Kernel./(k)
        
        # HNSW should achieve reasonable recall (at least 60%)
        assert recall >= 0.6, "HNSW recall too low: #{recall}"
        
        # HNSW distances should be close to exact distances
        avg_exact_dist = Enum.sum(exact_distances) / k
        avg_hnsw_dist = Enum.sum(hnsw_distances) / k
        
        # Average HNSW distance should not be much worse than exact
        distance_ratio = avg_hnsw_dist / (avg_exact_dist + 1.0e-10)
        assert distance_ratio <= 1.5, "HNSW distances too far from optimal: #{distance_ratio}"
      end
    end
  end
  
  describe "K-means Clustering Properties" do
    property "K-means centroids minimize within-cluster variance" do
      check all vectors <- TestHelpers.vector_batch_generator(30, 100, 4),
                k <- integer(2..min(8, div(length(vectors), 3))),
                max_runs: @quick_max_runs do
        
        {:ok, pid} = Clustering.KMeans.start_link([])
        result = Clustering.KMeans.cluster(pid, vectors, k, max_iterations: 30)
        GenServer.stop(pid)
        
        %{centroids: centroids, assignments: assignments, inertia: inertia} = result
        
        # Basic properties
        assert length(centroids) == k
        assert length(assignments) == length(vectors)
        assert Enum.all?(assignments, &(&1 >= 0 and &1 < k))
        
        # Each centroid should be the mean of its assigned vectors
        0..(k-1)
        |> Enum.each(fn cluster_id ->
          cluster_vectors = 
            assignments
            |> Enum.with_index()
            |> Enum.filter(fn {assignment, _idx} -> assignment == cluster_id end)
            |> Enum.map(fn {_assignment, idx} -> Enum.at(vectors, idx) end)
          
          if length(cluster_vectors) > 0 do
            dimensions = length(hd(vectors))
            
            # Calculate actual centroid
            actual_centroid = 
              0..(dimensions - 1)
              |> Enum.map(fn dim ->
                cluster_vectors
                |> Enum.map(&Enum.at(&1, dim))
                |> Enum.sum()
                |> Kernel./(length(cluster_vectors))
              end)
            
            expected_centroid = Enum.at(centroids, cluster_id)
            
            # Centroid should be close to the mean of assigned vectors
            TestHelpers.assert_vectors_equal(expected_centroid, actual_centroid, 0.01)
          end
        end)
        
        # Inertia should be non-negative and finite
        assert inertia >= 0.0
        assert is_float(inertia)
        refute Float.infinity?(inertia)
        refute Float.nan?(inertia)
        
        # Manual inertia calculation should match
        manual_inertia = 
          vectors
          |> Enum.zip(assignments)
          |> Enum.map(fn {vector, cluster_id} ->
            centroid = Enum.at(centroids, cluster_id)
            distance = TestHelpers.euclidean_distance(vector, centroid)
            distance * distance
          end)
          |> Enum.sum()
        
        assert_in_delta(inertia, manual_inertia, 0.001)
      end
    end
    
    property "K-means clustering is deterministic with fixed seed" do
      check all vectors <- TestHelpers.vector_batch_generator(20, 50, 3),
                k <- integer(2..min(6, div(length(vectors), 2))),
                seed <- integer(1..1000),
                max_runs: @quick_max_runs do
        
        # Run clustering twice with same seed
        {:ok, pid1} = Clustering.KMeans.start_link([])
        result1 = Clustering.KMeans.cluster(pid1, vectors, k, seed: seed, max_iterations: 20)
        GenServer.stop(pid1)
        
        {:ok, pid2} = Clustering.KMeans.start_link([])
        result2 = Clustering.KMeans.cluster(pid2, vectors, k, seed: seed, max_iterations: 20)
        GenServer.stop(pid2)
        
        # Results should be identical (or very close due to numerical precision)
        assert_in_delta(result1.inertia, result2.inertia, 0.001)
        
        # Centroids should match (possibly in different order)
        centroids1_sorted = Enum.sort_by(result1.centroids, &hd/1)
        centroids2_sorted = Enum.sort_by(result2.centroids, &hd/1)
        
        Enum.zip(centroids1_sorted, centroids2_sorted)
        |> Enum.each(fn {c1, c2} ->
          TestHelpers.assert_vectors_equal(c1, c2, 0.01)
        end)
      end
    end
    
    property "increasing k decreases inertia" do
      check all vectors <- TestHelpers.vector_batch_generator(40, 80, 4),
                k_base <- integer(2..6),
                max_runs: @quick_max_runs do
        
        if k_base + 1 <= div(length(vectors), 2) do
          # Cluster with k and k+1
          {:ok, pid1} = Clustering.KMeans.start_link([])
          result_k = Clustering.KMeans.cluster(pid1, vectors, k_base, max_iterations: 30)
          GenServer.stop(pid1)
          
          {:ok, pid2} = Clustering.KMeans.start_link([])
          result_k_plus_1 = Clustering.KMeans.cluster(pid2, vectors, k_base + 1, max_iterations: 30)
          GenServer.stop(pid2)
          
          # Higher k should have lower or equal inertia
          assert result_k_plus_1.inertia <= result_k.inertia + 0.001
        end
      end
    end
  end
  
  describe "Vector Quantization Properties" do
    property "product quantization preserves relative distances" do
      check all vectors <- TestHelpers.vector_batch_generator(30, 80, 8),
                subspaces <- integer(2..4),
                bits_per_code <- integer(3..6),
                max_runs: @quick_max_runs do
        
        dimensions = 8
        if rem(dimensions, subspaces) == 0 and length(vectors) >= :math.pow(2, bits_per_code) do
          # Train quantizer
          pq = Quantization.ProductQuantization.train(vectors, subspaces, bits_per_code)
          
          # Quantize and reconstruct
          reconstructed_vectors = 
            vectors
            |> Enum.map(&Quantization.ProductQuantization.quantize(pq, &1))
            |> Enum.map(&Quantization.ProductQuantization.reconstruct(pq, &1))
          
          # Test distance preservation on a subset of pairs
          test_pairs = 
            if length(vectors) >= 6 do
              [{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}]
            else
              [{0, 1}]
            end
          
          Enum.each(test_pairs, fn {i, j} ->
            if i < length(vectors) and j < length(vectors) do
              # Original distance
              original_dist = TestHelpers.euclidean_distance(
                Enum.at(vectors, i), 
                Enum.at(vectors, j)
              )
              
              # Reconstructed distance
              reconstructed_dist = TestHelpers.euclidean_distance(
                Enum.at(reconstructed_vectors, i), 
                Enum.at(reconstructed_vectors, j)
              )
              
              # Distance should be preserved reasonably well
              if original_dist > 1.0e-6 do  # Skip very small distances
                relative_error = abs(original_dist - reconstructed_dist) / original_dist
                assert relative_error <= 0.5, "Distance preservation failed: #{relative_error}"
              end
            end
          end)
          
          # Reconstruction error should be bounded
          Enum.zip(vectors, reconstructed_vectors)
          |> Enum.each(fn {original, reconstructed} ->
            error = TestHelpers.euclidean_distance(original, reconstructed)
            norm = TestHelpers.vector_norm(original)
            
            if norm > 1.0e-6 do
              relative_error = error / norm
              assert relative_error <= 0.8, "Reconstruction error too high: #{relative_error}"
            end
          end)
        end
      end
    end
    
    property "scalar quantization is monotonic" do
      check all values <- list_of(float(min: -100.0, max: 100.0), min_length: 20, max_length: 100),
                bits <- integer(4..8),
                max_runs: @quick_max_runs do
        
        # Create vectors from scalar values (1D vectors)
        vectors = Enum.map(values, &[&1])
        
        # Train scalar quantizer
        sq = Quantization.ScalarQuantization.train(vectors, bits, :uniform)
        
        # Test monotonicity on sorted values
        sorted_values = Enum.sort(values)
        
        quantized_values = 
          sorted_values
          |> Enum.map(&[&1])
          |> Enum.map(&Quantization.ScalarQuantization.quantize(sq, &1))
          |> Enum.map(&hd/1)
        
        # Quantized values should maintain relative ordering (monotonic)
        sorted_values
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.zip(Enum.chunk_every(quantized_values, 2, 1, :discard))
        |> Enum.each(fn {[v1, v2], [q1, q2]} ->
          if v1 < v2 do
            assert q1 <= q2, "Monotonicity violated: #{v1} < #{v2} but #{q1} > #{q2}"
          end
        end)
        
        # Dequantized values should be within quantization bounds
        max_quantized_value = round(:math.pow(2, bits)) - 1
        
        Enum.each(quantized_values, fn qval ->
          assert qval >= 0
          assert qval <= max_quantized_value
          assert is_integer(qval)
        end)
      end
    end
    
    property "quantization is deterministic and reversible" do
      check all vector <- TestHelpers.vector_generator(6, -50.0..50.0),
                max_runs: @default_max_runs do
        
        # Create a small dataset for training
        training_vectors = [
          vector,
          Enum.map(vector, &(&1 * 1.1)),
          Enum.map(vector, &(&1 * 0.9)),
          Enum.map(vector, &(&1 + 0.1))
        ]
        
        # Test Product Quantization
        if rem(6, 2) == 0 do  # 6 dimensions divisible by 2
          pq = Quantization.ProductQuantization.train(training_vectors, 2, 4)
          
          codes1 = Quantization.ProductQuantization.quantize(pq, vector)
          codes2 = Quantization.ProductQuantization.quantize(pq, vector)
          
          # Quantization should be deterministic
          assert codes1 == codes2
          
          # Reconstruction should be deterministic
          recon1 = Quantization.ProductQuantization.reconstruct(pq, codes1)
          recon2 = Quantization.ProductQuantization.reconstruct(pq, codes2)
          
          TestHelpers.assert_vectors_equal(recon1, recon2, 1.0e-10)
        end
        
        # Test Scalar Quantization
        sq = Quantization.ScalarQuantization.train(training_vectors, 6, :uniform)
        
        quantized1 = Quantization.ScalarQuantization.quantize(sq, vector)
        quantized2 = Quantization.ScalarQuantization.quantize(sq, vector)
        
        assert quantized1 == quantized2
        
        dequantized1 = Quantization.ScalarQuantization.dequantize(sq, quantized1)
        dequantized2 = Quantization.ScalarQuantization.dequantize(sq, quantized2)
        
        TestHelpers.assert_vectors_equal(dequantized1, dequantized2, 1.0e-10)
      end
    end
  end
  
  describe "Anomaly Detection Properties" do
    property "isolation forest anomaly scores are bounded and consistent" do
      check all vectors <- TestHelpers.vector_batch_generator(50, 150, 5),
                contamination <- float(min: 0.05, max: 0.3),
                max_runs: @quick_max_runs do
        
        {:ok, pid} = AnomalyDetection.IsolationForest.start_link(
          contamination: contamination, 
          n_estimators: 20  # Fewer estimators for faster property tests
        )
        
        :ok = AnomalyDetection.IsolationForest.train_baseline(pid, vectors)
        anomalies = AnomalyDetection.IsolationForest.detect_anomalies(pid, vectors)
        
        GenServer.stop(pid)
        
        # Number of anomalies should respect contamination rate (within tolerance)
        expected_count = round(contamination * length(vectors))
        actual_count = length(anomalies)
        tolerance = max(1, round(expected_count * 0.7))  # 70% tolerance for randomness
        
        assert abs(actual_count - expected_count) <= tolerance
        
        # All anomaly scores should be valid
        Enum.each(anomalies, fn %{anomaly_score: score, vector_id: id} ->
          assert score >= 0.0 and score <= 1.0
          assert is_integer(id)
          assert id >= 0 and id < length(vectors)
        end)
        
        # Test individual scoring consistency
        if length(vectors) >= 3 do
          test_vector = Enum.at(vectors, 0)
          
          {:ok, pid2} = AnomalyDetection.IsolationForest.start_link(
            contamination: contamination, 
            n_estimators: 20,
            random_seed: 42  # Fixed seed for determinism
          )
          
          :ok = AnomalyDetection.IsolationForest.train_baseline(pid2, vectors)
          
          result1 = AnomalyDetection.IsolationForest.is_anomaly?(pid2, test_vector)
          result2 = AnomalyDetection.IsolationForest.is_anomaly?(pid2, test_vector)
          
          # Same vector should get same result
          assert result1.is_anomaly == result2.is_anomaly
          assert_in_delta(result1.score, result2.score, 0.001)
          
          GenServer.stop(pid2)
        end
      end
    end
    
    property "anomaly detection handles edge cases gracefully" do
      check all base_vector <- TestHelpers.vector_generator(4, -10.0..10.0),
                scale_factor <- float(min: 0.1, max: 10.0),
                max_runs: @quick_max_runs do
        
        # Create dataset with potential edge cases
        normal_vectors = [
          base_vector,
          Enum.map(base_vector, &(&1 * 1.1)),
          Enum.map(base_vector, &(&1 * 0.9)),
          Enum.map(base_vector, &(&1 + 0.1)),
          Enum.map(base_vector, &(&1 - 0.1))
        ]
        
        # Add some scaled versions (potential outliers)
        scaled_vectors = [
          Enum.map(base_vector, &(&1 * scale_factor)),
          Enum.map(base_vector, &(&1 / scale_factor))
        ]
        
        all_vectors = normal_vectors ++ scaled_vectors
        
        # Test that system doesn't crash with edge cases
        {:ok, pid} = AnomalyDetection.IsolationForest.start_link(contamination: 0.2)
        
        # Should not raise exception
        :ok = AnomalyDetection.IsolationForest.train_baseline(pid, all_vectors)
        anomalies = AnomalyDetection.IsolationForest.detect_anomalies(pid, all_vectors)
        
        # Basic sanity checks
        assert is_list(anomalies)
        assert length(anomalies) <= length(all_vectors)
        
        # All results should have valid structure
        Enum.each(anomalies, fn anomaly ->
          assert Map.has_key?(anomaly, :vector_id)
          assert Map.has_key?(anomaly, :anomaly_score)
          assert is_integer(anomaly.vector_id)
          assert is_float(anomaly.anomaly_score)
        end)
        
        GenServer.stop(pid)
      end
    end
    
    property "ensemble detection provides more robust results" do
      check all vectors <- TestHelpers.vector_batch_generator(40, 100, 4),
                max_runs: @quick_max_runs do
        
        # Create data with some clear outliers
        normal_data = Enum.take(vectors, div(length(vectors) * 3, 4))
        outlier_data = 
          vectors
          |> Enum.drop(length(normal_data))
          |> Enum.map(fn v -> Enum.map(v, &(&1 * 3.0)) end)  # Scale to make outliers
        
        combined_data = normal_data ++ outlier_data
        
        # Test individual methods
        {:ok, if_pid} = AnomalyDetection.IsolationForest.start_link(contamination: 0.3)
        :ok = AnomalyDetection.IsolationForest.train_baseline(if_pid, combined_data)
        if_anomalies = AnomalyDetection.IsolationForest.detect_anomalies(if_pid, combined_data)
        GenServer.stop(if_pid)
        
        {:ok, stat_pid} = AnomalyDetection.StatisticalOutlier.start_link(method: :z_score, contamination: 0.3)
        :ok = AnomalyDetection.StatisticalOutlier.train_baseline(stat_pid, combined_data)
        stat_anomalies = AnomalyDetection.StatisticalOutlier.detect_anomalies(stat_pid, combined_data)
        GenServer.stop(stat_pid)
        
        # Test ensemble method
        {:ok, ens_pid} = AnomalyDetection.EnsembleDetector.start_link(
          methods: [:isolation_forest, :statistical], 
          contamination: 0.3,
          voting_strategy: :majority
        )
        :ok = AnomalyDetection.EnsembleDetector.train_baseline(ens_pid, combined_data)
        ens_anomalies = AnomalyDetection.EnsembleDetector.detect_anomalies(ens_pid, combined_data)
        GenServer.stop(ens_pid)
        
        # Ensemble results should have consensus information
        if length(ens_anomalies) > 0 do
          sample_anomaly = hd(ens_anomalies)
          assert Map.has_key?(sample_anomaly, :consensus_score)
          assert Map.has_key?(sample_anomaly, :method_votes)
          
          assert is_float(sample_anomaly.consensus_score)
          assert sample_anomaly.consensus_score >= 0.0 and sample_anomaly.consensus_score <= 1.0
        end
        
        # All methods should detect some anomalies
        assert length(if_anomalies) > 0
        assert length(stat_anomalies) > 0
        assert length(ens_anomalies) > 0
        
        # Ensemble count should be reasonable compared to individual methods
        individual_avg = (length(if_anomalies) + length(stat_anomalies)) / 2
        ensemble_count = length(ens_anomalies)
        
        # Ensemble should not be drastically different from individual methods
        ratio = ensemble_count / individual_avg
        assert ratio >= 0.3 and ratio <= 3.0
      end
    end
  end
  
  describe "Integration Properties" do
    property "complete ML workflow maintains data consistency" do
      check all vectors <- TestHelpers.vector_batch_generator(30, 80, 6),
                k <- integer(3..min(8, div(length(vectors), 3))),
                max_runs: @quick_max_runs do
        
        original_count = length(vectors)
        
        # Step 1: Insert vectors
        {:ok, vector_ids} = System1.Operations.insert_batch(vectors, %{workflow: "property_test"})
        assert length(vector_ids) == original_count
        
        # Step 2: Verify retrieval
        retrieved_count = System1.Operations.get_vector_count()
        assert retrieved_count.total_vectors >= original_count
        
        # Step 3: Clustering
        {:ok, pid} = Clustering.KMeans.start_link([])
        cluster_result = Clustering.KMeans.cluster(pid, vectors, k)
        GenServer.stop(pid)
        
        # Clustering should preserve vector count
        assert length(cluster_result.assignments) == original_count
        
        # Step 4: Search operations
        if length(vectors) > 0 do
          query_vector = hd(vectors)
          search_results = HNSW.Search.search_knn(
            build_test_hnsw_graph(vectors, 6), 
            query_vector, 
            min(5, length(vectors))
          )
          
          # Search should return valid results
          assert length(search_results) > 0
          assert length(search_results) <= min(5, length(vectors))
          
          # First result should be the query vector itself (distance ~0)
          first_result = hd(search_results)
          assert first_result.distance < 0.001
        end
        
        # Step 5: Anomaly detection
        {:ok, ad_pid} = AnomalyDetection.IsolationForest.start_link(contamination: 0.1)
        :ok = AnomalyDetection.IsolationForest.train_baseline(ad_pid, vectors)
        anomalies = AnomalyDetection.IsolationForest.detect_anomalies(ad_pid, vectors)
        GenServer.stop(ad_pid)
        
        # Anomaly detection should process all vectors
        all_vector_ids = MapSet.new(0..(length(vectors) - 1))
        detected_ids = MapSet.new(Enum.map(anomalies, & &1.vector_id))
        
        # All detected anomaly IDs should be valid
        assert MapSet.subset?(detected_ids, all_vector_ids)
      end
    end
    
    property "system maintains consistency under concurrent operations" do
      check all vectors <- TestHelpers.vector_batch_generator(20, 60, 4),
                max_runs: @quick_max_runs do
        
        # Split vectors for concurrent operations
        {batch1, batch2} = Enum.split(vectors, div(length(vectors), 2))
        
        # Concurrent insertions
        task1 = Task.async(fn ->
          System1.Operations.insert_batch(batch1, %{batch: "concurrent_1"})
        end)
        
        task2 = Task.async(fn ->
          System1.Operations.insert_batch(batch2, %{batch: "concurrent_2"})
        end)
        
        {:ok, ids1} = Task.await(task1, 5000)
        {:ok, ids2} = Task.await(task2, 5000)
        
        # All insertions should succeed
        assert length(ids1) == length(batch1)
        assert length(ids2) == length(batch2)
        
        # IDs should be unique
        all_ids = ids1 ++ ids2
        assert length(Enum.uniq(all_ids)) == length(all_ids)
        
        # System should be in consistent state
        total_count = System1.Operations.get_vector_count()
        assert total_count.total_vectors >= length(vectors)
        
        # Concurrent searches should work
        if length(vectors) > 0 do
          query1 = hd(batch1)
          query2 = if length(batch2) > 0, do: hd(batch2), else: hd(batch1)
          
          search_task1 = Task.async(fn ->
            graph = build_test_hnsw_graph(vectors, 4)
            HNSW.Search.search_knn(graph, query1, 3)
          end)
          
          search_task2 = Task.async(fn ->
            graph = build_test_hnsw_graph(vectors, 4)
            HNSW.Search.search_knn(graph, query2, 3)
          end)
          
          results1 = Task.await(search_task1, 5000)
          results2 = Task.await(search_task2, 5000)
          
          # Both searches should succeed
          assert length(results1) > 0
          assert length(results2) > 0
          
          # Results should be properly formatted
          Enum.each(results1 ++ results2, fn result ->
            assert Map.has_key?(result, :id)
            assert Map.has_key?(result, :distance)
            assert is_integer(result.id)
            assert is_float(result.distance)
            assert result.distance >= 0.0
          end)
        end
      end
    end
  end
  
  # Helper function for building test HNSW graphs
  defp build_test_hnsw_graph(vectors, dimensions) do
    graph = HNSW.Graph.new(
      dimensions: dimensions, 
      max_connections: 8,  # Smaller for faster property tests
      ml_constant: 1/:math.log(2)
    )
    
    vectors
    |> Enum.with_index()
    |> Enum.reduce(graph, fn {vector, id}, acc_graph ->
      {:ok, updated_graph} = HNSW.Graph.insert_vector(acc_graph, id, vector)
      updated_graph
    end)
  end
end