defmodule VsmVectorStore.AnomalyDetection.Test do
  @moduledoc """
  Comprehensive unit tests for anomaly detection algorithms.
  
  Tests Isolation Forest, Local Outlier Factor (LOF), statistical methods,
  and ensemble anomaly detection approaches.
  """
  
  use ExUnit.Case, async: true
  use ExUnitProperties
  
  alias VsmVectorStore.TestHelpers
  alias VsmVectorStore.AnomalyDetection.{IsolationForest, LocalOutlierFactor, StatisticalOutlier, EnsembleDetector}
  
  describe "Isolation Forest Algorithm" do
    test "builds isolation trees correctly" do
      vectors = TestHelpers.generate_random_vectors(100, 4, seed: 4001)
      
      {:ok, pid} = IsolationForest.start_link(n_estimators: 50, contamination: 0.1)
      :ok = IsolationForest.train_baseline(pid, vectors)
      
      # Get the trained model
      model = IsolationForest.get_model(pid)
      
      assert length(model.trees) == 50
      assert model.contamination == 0.1
      assert is_float(model.threshold)
      
      # Each tree should have proper structure
      Enum.each(model.trees, fn tree ->
        assert validate_isolation_tree(tree)
      end)
      
      GenServer.stop(pid)
    end
    
    test "detects anomalies in mixed dataset" do
      # Create dataset with known outliers
      normal_vectors = TestHelpers.generate_random_vectors(180, 3, seed: 4002, scale: 1.0)
      outlier_vectors = TestHelpers.generate_random_vectors(20, 3, seed: 4003, scale: 5.0)
      all_vectors = normal_vectors ++ outlier_vectors
      
      {:ok, pid} = IsolationForest.start_link(contamination: 0.15)  # Expect ~15% anomalies
      :ok = IsolationForest.train_baseline(pid, all_vectors)
      
      anomalies = IsolationForest.detect_anomalies(pid, all_vectors)
      
      # Validate anomaly detection results
      TestHelpers.validate_anomaly_detection(all_vectors, anomalies, 0.15)
      
      # Check that more outliers are detected than normal vectors
      outlier_indices = 180..199 |> Enum.to_list()  # Indices of known outliers
      detected_outlier_count = 
        anomalies
        |> Enum.count(fn %{vector_id: id} -> id in outlier_indices end)
      
      # Should detect at least 50% of the outliers
      assert detected_outlier_count >= 10
      
      GenServer.stop(pid)
    end
    
    test "calculates anomaly scores correctly" do
      vectors = TestHelpers.generate_clustered_vectors(200, 5, 3, seed: 4004)
      
      {:ok, pid} = IsolationForest.start_link(n_estimators: 100)
      :ok = IsolationForest.train_baseline(pid, vectors)
      
      # Test individual anomaly scoring
      test_vector = hd(vectors)
      is_anomaly_result = IsolationForest.is_anomaly?(pid, test_vector)
      
      assert %{is_anomaly: is_anomaly, score: score} = is_anomaly_result
      assert is_boolean(is_anomaly)
      assert is_float(score)
      assert score >= 0.0 and score <= 1.0
      
      # Outlier vector should have higher anomaly score
      outlier_vector = Enum.map(test_vector, &(&1 * 10))  # Scale up dramatically
      outlier_result = IsolationForest.is_anomaly?(pid, outlier_vector)
      
      assert outlier_result.score > is_anomaly_result.score
      
      GenServer.stop(pid)
    end
    
    property "isolation forest properties hold" do
      check all vectors <- TestHelpers.vector_batch_generator(50, 200, 4),
                n_estimators <- integer(10..50),
                contamination <- float(min: 0.01, max: 0.3) do
        
        {:ok, pid} = IsolationForest.start_link(
          n_estimators: n_estimators, 
          contamination: contamination
        )
        :ok = IsolationForest.train_baseline(pid, vectors)
        
        anomalies = IsolationForest.detect_anomalies(pid, vectors)
        
        # Anomaly count should respect contamination rate (within tolerance)
        expected_count = round(contamination * length(vectors))
        actual_count = length(anomalies)
        tolerance = max(1, round(expected_count * 0.5))  # 50% tolerance
        
        assert abs(actual_count - expected_count) <= tolerance
        
        # All anomaly scores should be valid
        Enum.each(anomalies, fn %{anomaly_score: score} ->
          assert score >= 0.0 and score <= 1.0
        end)
        
        GenServer.stop(pid)
      end
    end
    
    test "handles edge cases gracefully" do
      {:ok, pid} = IsolationForest.start_link([])
      
      # Test with minimal data
      tiny_vectors = [[1.0, 2.0], [1.1, 2.1]]
      :ok = IsolationForest.train_baseline(pid, tiny_vectors)
      
      result = IsolationForest.is_anomaly?(pid, [1.05, 2.05])
      assert %{is_anomaly: _, score: _} = result
      
      # Test with identical vectors
      identical_vectors = List.duplicate([1.0, 2.0, 3.0], 10)
      :ok = IsolationForest.train_baseline(pid, identical_vectors)
      
      # Should handle without crashing
      anomalies = IsolationForest.detect_anomalies(pid, identical_vectors)
      assert is_list(anomalies)
      
      GenServer.stop(pid)
    end
  end
  
  describe "Local Outlier Factor (LOF)" do
    test "calculates local density correctly" do
      # Create data with clear density differences
      dense_cluster = TestHelpers.generate_vectors_around_center([0.0, 0.0], 50, 0.3)
      sparse_points = [[5.0, 5.0], [5.1, 5.1], [5.2, 5.2]]  # Sparse region
      vectors = dense_cluster ++ sparse_points
      
      {:ok, pid} = LocalOutlierFactor.start_link(k_neighbors: 5)
      :ok = LocalOutlierFactor.train_baseline(pid, vectors)
      
      # Dense cluster points should have low LOF scores
      dense_point = hd(dense_cluster)
      dense_result = LocalOutlierFactor.is_anomaly?(pid, dense_point)
      
      # Sparse points should have higher LOF scores
      sparse_point = [5.0, 5.0]
      sparse_result = LocalOutlierFactor.is_anomaly?(pid, sparse_point)
      
      assert sparse_result.score > dense_result.score
      
      GenServer.stop(pid)
    end
    
    test "identifies local outliers correctly" do
      # Create dataset with local and global structure
      main_cluster = TestHelpers.generate_vectors_around_center([0.0, 0.0], 80, 0.5)
      secondary_cluster = TestHelpers.generate_vectors_around_center([10.0, 10.0], 15, 0.3)
      outliers = [[5.0, 5.0], [15.0, 0.0]]  # Points between clusters
      
      vectors = main_cluster ++ secondary_cluster ++ outliers
      
      {:ok, pid} = LocalOutlierFactor.start_link(k_neighbors: 10, contamination: 0.1)
      :ok = LocalOutlierFactor.train_baseline(pid, vectors)
      
      anomalies = LocalOutlierFactor.detect_anomalies(pid, vectors)
      
      # Should detect the outliers between clusters
      outlier_indices = (length(main_cluster) + length(secondary_cluster))..(length(vectors) - 1) |> Enum.to_list()
      
      detected_outliers = 
        anomalies
        |> Enum.count(fn %{vector_id: id} -> id in outlier_indices end)
      
      # Should detect at least one of the outliers
      assert detected_outliers >= 1
      
      GenServer.stop(pid)
    end
    
    test "respects k_neighbors parameter" do
      vectors = TestHelpers.generate_random_vectors(100, 3, seed: 4005)
      
      # Test with different k values
      k_values = [3, 10, 20]
      
      results = 
        k_values
        |> Enum.map(fn k ->
          {:ok, pid} = LocalOutlierFactor.start_link(k_neighbors: k, contamination: 0.1)
          :ok = LocalOutlierFactor.train_baseline(pid, vectors)
          
          anomalies = LocalOutlierFactor.detect_anomalies(pid, vectors)
          GenServer.stop(pid)
          
          {k, length(anomalies)}
        end)
      
      # Different k values should potentially give different results
      anomaly_counts = Enum.map(results, &elem(&1, 1))
      
      # All should detect some anomalies
      Enum.each(anomaly_counts, fn count ->
        assert count > 0
        assert count <= 20  # Shouldn't detect more than 20% with contamination=0.1
      end)
    end
    
    property "LOF scores are consistent and bounded" do
      check all vectors <- TestHelpers.vector_batch_generator(30, 100, 3),
                k <- integer(3..min(15, div(length(vectors), 2))) do
        
        {:ok, pid} = LocalOutlierFactor.start_link(k_neighbors: k)
        :ok = LocalOutlierFactor.train_baseline(pid, vectors)
        
        # Test multiple points
        test_points = Enum.take(vectors, 5)
        
        Enum.each(test_points, fn point ->
          result = LocalOutlierFactor.is_anomaly?(pid, point)
          
          assert %{is_anomaly: is_anomaly, score: score} = result
          assert is_boolean(is_anomaly)
          assert is_float(score)
          assert score >= 0.0  # LOF scores should be non-negative
        end)
        
        GenServer.stop(pid)
      end
    end
  end
  
  describe "Statistical Outlier Detection" do
    test "detects outliers using z-score method" do
      # Create normal data with some extreme outliers
      normal_data = TestHelpers.generate_random_vectors(100, 2, seed: 4006, scale: 1.0)
      outliers = [[10.0, 10.0], [-10.0, -10.0], [15.0, -15.0]]
      vectors = normal_data ++ outliers
      
      {:ok, pid} = StatisticalOutlier.start_link(method: :z_score, threshold: 3.0)
      :ok = StatisticalOutlier.train_baseline(pid, vectors)
      
      anomalies = StatisticalOutlier.detect_anomalies(pid, vectors)
      
      # Should detect the extreme outliers
      outlier_indices = 100..102 |> Enum.to_list()
      detected_outliers = 
        anomalies
        |> Enum.count(fn %{vector_id: id} -> id in outlier_indices end)
      
      assert detected_outliers >= 2  # Should catch most extreme outliers
      
      GenServer.stop(pid)
    end
    
    test "uses modified z-score for robustness" do
      # Create data with outliers that might skew mean/std
      base_data = TestHelpers.generate_random_vectors(80, 3, seed: 4007, scale: 1.0)
      extreme_outliers = List.duplicate([100.0, 100.0, 100.0], 5)  # Extreme outliers
      mild_outliers = [[3.0, 3.0, 3.0], [3.1, 3.1, 3.1]]
      vectors = base_data ++ extreme_outliers ++ mild_outliers
      
      # Regular z-score (uses mean/std, affected by outliers)
      {:ok, pid_zscore} = StatisticalOutlier.start_link(method: :z_score, threshold: 2.5)
      :ok = StatisticalOutlier.train_baseline(pid_zscore, vectors)
      zscore_anomalies = StatisticalOutlier.detect_anomalies(pid_zscore, vectors)
      
      # Modified z-score (uses median/MAD, more robust)
      {:ok, pid_modified} = StatisticalOutlier.start_link(method: :modified_z_score, threshold: 3.5)
      :ok = StatisticalOutlier.train_baseline(pid_modified, vectors)
      modified_anomalies = StatisticalOutlier.detect_anomalies(pid_modified, vectors)
      
      # Modified z-score should be more robust to extreme outliers
      extreme_outlier_indices = 80..84 |> Enum.to_list()
      
      modified_detected = 
        modified_anomalies
        |> Enum.count(fn %{vector_id: id} -> id in extreme_outlier_indices end)
      
      assert modified_detected >= 3  # Should detect most extreme outliers
      
      GenServer.stop(pid_zscore)
      GenServer.stop(pid_modified)
    end
    
    test "handles different statistical methods" do
      vectors = TestHelpers.generate_random_vectors(200, 4, seed: 4008)
      methods = [:z_score, :modified_z_score, :iqr, :isolation_forest_statistical]
      
      method_results = 
        methods
        |> Enum.map(fn method ->
          {:ok, pid} = StatisticalOutlier.start_link(method: method, contamination: 0.05)
          :ok = StatisticalOutlier.train_baseline(pid, vectors)
          
          anomalies = StatisticalOutlier.detect_anomalies(pid, vectors)
          GenServer.stop(pid)
          
          {method, length(anomalies)}
        end)
      
      # All methods should detect some anomalies
      Enum.each(method_results, fn {method, count} ->
        assert count > 0, "Method #{method} detected no anomalies"
        assert count <= 20, "Method #{method} detected too many anomalies: #{count}"
      end)
    end
    
    test "IQR method detects box plot outliers correctly" do
      # Create data with known quartile structure
      data_values = Enum.to_list(1..100) |> Enum.map(&(&1 / 10.0))  # 0.1 to 10.0
      outliers_values = [20.0, 25.0, -5.0]  # Clear outliers beyond IQR range
      
      # Convert to 1D vectors for simplicity
      vectors = (Enum.map(data_values, &[&1]) ++ Enum.map(outliers_values, &[&1]))
      
      {:ok, pid} = StatisticalOutlier.start_link(method: :iqr, iqr_factor: 1.5)
      :ok = StatisticalOutlier.train_baseline(pid, vectors)
      
      anomalies = StatisticalOutlier.detect_anomalies(pid, vectors)
      
      # Should detect the extreme outliers
      outlier_indices = 100..102 |> Enum.to_list()
      detected_outliers = 
        anomalies
        |> Enum.count(fn %{vector_id: id} -> id in outlier_indices end)
      
      assert detected_outliers >= 2
      
      GenServer.stop(pid)
    end
  end
  
  describe "Ensemble Anomaly Detection" do
    test "combines multiple detection methods" do
      vectors = TestHelpers.generate_vectors_with_outliers(180, 4, 20, seed: 4009)
      
      methods = [:isolation_forest, :lof, :statistical]
      {:ok, pid} = EnsembleDetector.start_link(methods: methods, voting_strategy: :majority)
      
      :ok = EnsembleDetector.train_baseline(pid, vectors)
      anomalies = EnsembleDetector.detect_anomalies(pid, vectors)
      
      # Ensemble should provide robust detection
      TestHelpers.validate_anomaly_detection(vectors, anomalies, 0.12)  # Expect ~10-12% anomalies
      
      # Should have consensus scores
      Enum.each(anomalies, fn anomaly ->
        assert Map.has_key?(anomaly, :consensus_score)
        assert anomaly.consensus_score >= 0.0 and anomaly.consensus_score <= 1.0
        
        assert Map.has_key?(anomaly, :method_votes)
        assert is_map(anomaly.method_votes)
      end)
      
      GenServer.stop(pid)
    end
    
    test "uses different voting strategies" do
      vectors = TestHelpers.generate_vectors_with_outliers(100, 3, 15, seed: 4010)
      methods = [:isolation_forest, :lof]
      
      # Test majority voting
      {:ok, pid_majority} = EnsembleDetector.start_link(methods: methods, voting_strategy: :majority)
      :ok = EnsembleDetector.train_baseline(pid_majority, vectors)
      majority_anomalies = EnsembleDetector.detect_anomalies(pid_majority, vectors)
      
      # Test average voting
      {:ok, pid_average} = EnsembleDetector.start_link(methods: methods, voting_strategy: :average)
      :ok = EnsembleDetector.train_baseline(pid_average, vectors)
      average_anomalies = EnsembleDetector.detect_anomalies(pid_average, vectors)
      
      # Test max voting (most sensitive)
      {:ok, pid_max} = EnsembleDetector.start_link(methods: methods, voting_strategy: :max)
      :ok = EnsembleDetector.train_baseline(pid_max, vectors)
      max_anomalies = EnsembleDetector.detect_anomalies(pid_max, vectors)
      
      # Different strategies should potentially give different results
      counts = [length(majority_anomalies), length(average_anomalies), length(max_anomalies)]
      
      # Max voting should generally detect the most anomalies
      assert length(max_anomalies) >= length(majority_anomalies)
      
      GenServer.stop(pid_majority)
      GenServer.stop(pid_average)
      GenServer.stop(pid_max)
    end
    
    test "provides method breakdown in results" do
      vectors = TestHelpers.generate_random_vectors(100, 5, seed: 4011)
      methods = [:isolation_forest, :lof, :statistical]
      
      {:ok, pid} = EnsembleDetector.start_link(methods: methods)
      :ok = EnsembleDetector.train_baseline(pid, vectors)
      
      # Test single vector analysis
      test_vector = hd(vectors)
      result = EnsembleDetector.is_anomaly?(pid, test_vector)
      
      assert %{
        is_anomaly: _,
        consensus_score: consensus,
        method_votes: votes,
        method_scores: scores
      } = result
      
      assert is_float(consensus)
      assert is_map(votes)
      assert is_map(scores)
      
      # Should have results from all methods
      Enum.each(methods, fn method ->
        assert Map.has_key?(votes, method)
        assert Map.has_key?(scores, method)
        assert is_boolean(votes[method])
        assert is_float(scores[method])
      end)
      
      GenServer.stop(pid)
    end
    
    property "ensemble detection is robust" do
      check all vectors <- TestHelpers.vector_batch_generator(50, 150, 3),
                contamination <- float(min: 0.05, max: 0.2) do
        
        methods = [:isolation_forest, :statistical]  # Faster methods for property testing
        {:ok, pid} = EnsembleDetector.start_link(methods: methods, contamination: contamination)
        :ok = EnsembleDetector.train_baseline(pid, vectors)
        
        anomalies = EnsembleDetector.detect_anomalies(pid, vectors)
        
        # Basic validation
        assert is_list(anomalies)
        assert length(anomalies) <= length(vectors)
        
        # Ensemble scores should be valid
        Enum.each(anomalies, fn anomaly ->
          assert anomaly.consensus_score >= 0.0 and anomaly.consensus_score <= 1.0
          assert anomaly.anomaly_score >= 0.0 and anomaly.anomaly_score <= 1.0
        end)
        
        GenServer.stop(pid)
      end
    end
  end
  
  describe "Anomaly Detection Performance and Scalability" do
    test "handles large datasets efficiently" do
      large_vectors = TestHelpers.generate_random_vectors(2000, 8, seed: 5001)
      
      {:ok, pid} = IsolationForest.start_link(n_estimators: 50)
      
      # Measure training time
      {_, train_time_ms} = TestHelpers.measure_time(fn ->
        IsolationForest.train_baseline(pid, large_vectors)
      end)
      
      # Measure detection time
      {anomalies, detect_time_ms} = TestHelpers.measure_time(fn ->
        IsolationForest.detect_anomalies(pid, large_vectors)
      end)
      
      # Should complete within reasonable time
      TestHelpers.assert_performance_bounds(train_time_ms, 10000)   # 10 seconds for training
      TestHelpers.assert_performance_bounds(detect_time_ms, 5000)   # 5 seconds for detection
      
      # Verify results
      assert is_list(anomalies)
      assert length(anomalies) > 0
      
      GenServer.stop(pid)
    end
    
    test "memory usage scales reasonably" do
      base_size = 200
      dimensions = 6
      
      # Test with different dataset sizes
      memory_usage = [200, 400, 800]
      |> Enum.map(fn size ->
        vectors = TestHelpers.generate_random_vectors(size, dimensions, seed: size)
        
        {_result, memory_used} = TestHelpers.measure_memory(fn ->
          {:ok, pid} = IsolationForest.start_link(n_estimators: 20)
          :ok = IsolationForest.train_baseline(pid, vectors)
          anomalies = IsolationForest.detect_anomalies(pid, vectors)
          GenServer.stop(pid)
          anomalies
        end)
        
        {size, memory_used}
      end)
      
      # Memory should scale sub-quadratically
      [{_, mem1}, {_, mem2}, {_, mem3}] = memory_usage
      
      # 4x data should not use more than 6x memory
      assert mem3 <= mem1 * 6
    end
    
    test "concurrent anomaly detection is safe" do
      vectors = TestHelpers.generate_random_vectors(500, 4, seed: 5002)
      
      {:ok, pid} = IsolationForest.start_link(n_estimators: 30)
      :ok = IsolationForest.train_baseline(pid, vectors)
      
      # Run concurrent detection tasks
      tasks = 1..8
      |> Enum.map(fn i ->
        Task.async(fn ->
          subset_start = (i - 1) * 60
          subset_end = min(i * 60, length(vectors)) - 1
          subset = Enum.slice(vectors, subset_start..subset_end)
          
          IsolationForest.detect_anomalies(pid, subset)
        end)
      end)
      
      results = Task.await_many(tasks, 10000)
      
      # All tasks should complete successfully
      assert length(results) == 8
      Enum.each(results, fn result ->
        assert is_list(result)
        
        Enum.each(result, fn anomaly ->
          assert Map.has_key?(anomaly, :vector_id)
          assert Map.has_key?(anomaly, :anomaly_score)
        end)
      end)
      
      GenServer.stop(pid)
    end
  end
  
  describe "Anomaly Detection Edge Cases" do
    test "handles all identical vectors" do
      identical_vectors = List.duplicate([1.0, 2.0, 3.0], 50)
      
      {:ok, pid} = IsolationForest.start_link([])
      :ok = IsolationForest.train_baseline(pid, identical_vectors)
      
      # Should not crash, but probably won't detect anomalies in identical data
      anomalies = IsolationForest.detect_anomalies(pid, identical_vectors)
      assert is_list(anomalies)
      
      GenServer.stop(pid)
    end
    
    test "handles vectors with extreme values" do
      normal_vectors = TestHelpers.generate_random_vectors(80, 3, seed: 5003)
      extreme_vectors = [[1.0e10, 1.0e10, 1.0e10], [-1.0e10, -1.0e10, -1.0e10]]
      vectors = normal_vectors ++ extreme_vectors
      
      {:ok, pid} = StatisticalOutlier.start_link(method: :modified_z_score)
      :ok = StatisticalOutlier.train_baseline(pid, vectors)
      
      anomalies = StatisticalOutlier.detect_anomalies(pid, vectors)
      
      # Should detect the extreme vectors
      extreme_indices = [80, 81]
      detected_extremes = 
        anomalies
        |> Enum.count(fn %{vector_id: id} -> id in extreme_indices end)
      
      assert detected_extremes >= 1
      
      GenServer.stop(pid)
    end
    
    test "validates input parameters" do
      vectors = TestHelpers.generate_random_vectors(20, 3)
      
      # Invalid contamination rates
      assert_raise ArgumentError, fn ->
        {:ok, _} = IsolationForest.start_link(contamination: -0.1)
      end
      
      assert_raise ArgumentError, fn ->
        {:ok, _} = IsolationForest.start_link(contamination: 1.1)
      end
      
      # Invalid n_estimators
      assert_raise ArgumentError, fn ->
        {:ok, _} = IsolationForest.start_link(n_estimators: 0)
      end
      
      # Invalid k_neighbors for LOF
      assert_raise ArgumentError, fn ->
        {:ok, _} = LocalOutlierFactor.start_link(k_neighbors: 0)
      end
    end
    
    test "handles insufficient training data gracefully" do
      tiny_vectors = [[1.0, 2.0]]  # Only one vector
      
      {:ok, pid} = IsolationForest.start_link([])
      
      # Should handle gracefully (not crash)
      assert_raise ArgumentError, fn ->
        IsolationForest.train_baseline(pid, tiny_vectors)
      end
      
      GenServer.stop(pid)
    end
  end
  
  # Helper functions
  
  defp validate_isolation_tree(tree) do
    case tree do
      %{type: :leaf, size: size} ->
        is_integer(size) and size >= 0
      
      %{type: :internal, split_attr: attr, split_value: val, left: left, right: right} ->
        is_integer(attr) and is_float(val) and 
        validate_isolation_tree(left) and validate_isolation_tree(right)
      
      _ -> false
    end
  end
end