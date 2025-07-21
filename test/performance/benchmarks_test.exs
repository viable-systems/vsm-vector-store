defmodule VsmVectorStore.Performance.BenchmarksTest do
  @moduledoc """
  Performance benchmarks and stress tests for VSM Vector Store.
  
  Tests scalability, memory efficiency, throughput, and latency characteristics
  with large datasets and high-concurrency scenarios.
  """
  
  use ExUnit.Case, async: false  # Performance tests need dedicated resources
  
  alias VsmVectorStore.TestHelpers
  alias VsmVectorStore.{HNSW, Clustering, Quantization, AnomalyDetection}
  
  @large_dataset_size 10_000
  @stress_dataset_size 50_000
  @high_dimensions 128
  @medium_dimensions 32
  @low_dimensions 8
  
  # Skip performance tests in CI unless explicitly requested
  @moduletag :performance
  @moduletag timeout: 300_000  # 5 minutes for performance tests
  
  describe "Large Dataset Performance" do
    @tag :slow
    test "HNSW search scales logarithmically with dataset size" do
      dimensions = @medium_dimensions
      
      # Test with increasing dataset sizes
      dataset_sizes = [1_000, 5_000, 10_000, 25_000]
      
      performance_results = 
        dataset_sizes
        |> Enum.map(fn size ->
          IO.puts("Testing HNSW with #{size} vectors...")
          
          # Generate test dataset
          vectors = TestHelpers.generate_random_vectors(size, dimensions, seed: size)
          
          # Build HNSW index
          {graph, build_time} = TestHelpers.measure_time(fn ->
            build_hnsw_index(vectors, dimensions)
          end)
          
          # Measure search performance
          query_vector = TestHelpers.generate_random_vectors(1, dimensions, seed: 99999) |> hd()
          
          {search_results, search_time} = TestHelpers.measure_time(fn ->
            HNSW.Search.search_knn(graph, query_vector, 10)
          end)
          
          # Measure memory usage
          {_, memory_usage} = TestHelpers.measure_memory(fn ->
            :erlang.size(graph)
          end)
          
          %{
            size: size,
            build_time_ms: build_time,
            search_time_ms: search_time,
            memory_kb: div(memory_usage, 1024),
            results_count: length(search_results),
            search_quality: calculate_search_quality(search_results, query_vector, vectors)
          }
        end)
      
      # Analyze scaling characteristics
      analyze_scaling_performance(performance_results)
      
      # Verify logarithmic scaling
      verify_logarithmic_scaling(performance_results)
      
      # Print performance summary
      print_performance_summary("HNSW Scaling", performance_results)
    end
    
    @tag :slow
    test "K-means clustering performance with large datasets" do
      dimensions = @medium_dimensions
      dataset_sizes = [2_000, 5_000, 10_000, 20_000]
      k_values = [5, 10, 15]
      
      performance_matrix = 
        for size <- dataset_sizes,
            k <- k_values do
          
          IO.puts("Testing K-means: #{size} vectors, k=#{k}")
          
          vectors = TestHelpers.generate_clustered_vectors(size, dimensions, k, seed: size * k)
          
          {:ok, pid} = Clustering.KMeans.start_link([])
          
          {result, clustering_time} = TestHelpers.measure_time(fn ->
            Clustering.KMeans.cluster(pid, vectors, k, max_iterations: 50)
          end)
          
          GenServer.stop(pid)
          
          %{
            size: size,
            k: k,
            clustering_time_ms: clustering_time,
            iterations: result.iterations,
            inertia: result.inertia,
            converged: result.iterations < 50,
            time_per_vector: clustering_time / size,
            memory_efficiency: calculate_memory_efficiency(size, dimensions)
          }
        end
      
      # Analyze clustering performance
      analyze_clustering_performance(performance_matrix)
      
      # Verify acceptable performance bounds
      verify_clustering_bounds(performance_matrix)
      
      print_performance_matrix("K-means Performance", performance_matrix)
    end
    
    @tag :slow  
    test "vector quantization compression and speed" do
      dimensions = @high_dimensions
      vector_count = @large_dataset_size
      
      vectors = TestHelpers.generate_random_vectors(vector_count, dimensions, seed: 12345)
      IO.puts("Testing quantization with #{vector_count} #{dimensions}D vectors...")
      
      quantization_benchmarks = %{
        product_quantization: benchmark_product_quantization(vectors),
        scalar_quantization: benchmark_scalar_quantization(vectors)
      }
      
      # Analyze compression ratios and performance
      Enum.each(quantization_benchmarks, fn {method, results} ->
        IO.puts("\n#{method} Results:")
        IO.puts("  Training time: #{results.train_time_ms}ms")
        IO.puts("  Quantization time: #{results.quantize_time_ms}ms")
        IO.puts("  Reconstruction time: #{results.reconstruct_time_ms}ms")
        IO.puts("  Compression ratio: #{results.compression_ratio}:1")
        IO.puts("  Average error: #{Float.round(results.avg_reconstruction_error, 4)}")
        
        # Verify performance bounds
        assert results.train_time_ms < 30_000      # < 30 seconds training
        assert results.quantize_time_ms < 10_000   # < 10 seconds quantization
        assert results.compression_ratio > 4.0     # > 4x compression
        assert results.avg_reconstruction_error < 0.3  # < 30% error
      end
    end
    
    @tag :slow
    test "anomaly detection scalability" do
      dimensions = @medium_dimensions
      dataset_sizes = [1_000, 5_000, 10_000, 25_000]
      
      detection_performance = 
        dataset_sizes
        |> Enum.map(fn size ->
          IO.puts("Testing anomaly detection with #{size} vectors...")
          
          # Generate data with known outliers (10%)
          normal_count = round(size * 0.9)
          outlier_count = size - normal_count
          
          normal_vectors = TestHelpers.generate_random_vectors(normal_count, dimensions, 
                                                              seed: size, scale: 1.0)
          outlier_vectors = TestHelpers.generate_random_vectors(outlier_count, dimensions,
                                                               seed: size * 2, scale: 4.0)
          vectors = normal_vectors ++ outlier_vectors
          
          # Benchmark Isolation Forest
          {:ok, pid} = AnomalyDetection.IsolationForest.start_link(contamination: 0.1)
          
          {_, train_time} = TestHelpers.measure_time(fn ->
            AnomalyDetection.IsolationForest.train_baseline(pid, vectors)
          end)
          
          {anomalies, detect_time} = TestHelpers.measure_time(fn ->
            AnomalyDetection.IsolationForest.detect_anomalies(pid, vectors)
          end)
          
          GenServer.stop(pid)
          
          # Calculate detection quality
          true_outliers = normal_count..(size - 1) |> Enum.to_list()
          detected_outliers = Enum.map(anomalies, & &1.vector_id)
          
          precision = calculate_precision(detected_outliers, true_outliers)
          recall = calculate_recall(detected_outliers, true_outliers)
          
          %{
            size: size,
            train_time_ms: train_time,
            detect_time_ms: detect_time,
            anomalies_detected: length(anomalies),
            precision: precision,
            recall: recall,
            f1_score: 2 * precision * recall / (precision + recall + 1.0e-10)
          }
        end)
      
      # Verify scaling and quality
      analyze_anomaly_detection_performance(detection_performance)
      print_performance_summary("Anomaly Detection", detection_performance)
    end
  end
  
  describe "High-Concurrency Stress Tests" do
    @tag :stress
    test "concurrent vector insertions" do
      vector_count = 5_000
      dimensions = @medium_dimensions
      concurrency_levels = [1, 4, 8, 16]
      
      concurrency_results = 
        concurrency_levels
        |> Enum.map(fn concurrency ->
          IO.puts("Testing #{concurrency} concurrent insertions...")
          
          vectors = TestHelpers.generate_random_vectors(vector_count, dimensions, seed: concurrency)
          batches = chunk_vectors(vectors, concurrency)
          
          {_, total_time} = TestHelpers.measure_time(fn ->
            tasks = 
              batches
              |> Enum.map(fn batch ->
                Task.async(fn ->
                  # Each task inserts its batch
                  System1.Operations.insert_batch(batch, %{thread_id: self()})
                end)
              end)
            
            Task.await_many(tasks, 30_000)
          end)
          
          throughput = vector_count / (total_time / 1000)  # vectors per second
          
          %{
            concurrency: concurrency,
            total_time_ms: total_time,
            throughput_vps: throughput,
            speedup: if(concurrency == 1, do: 1.0, else: calculate_speedup(concurrency_results, concurrency, throughput))
          }
        end)
      
      analyze_concurrency_performance(concurrency_results)
      print_performance_summary("Concurrent Insertions", concurrency_results)
    end
    
    @tag :stress
    test "concurrent search operations" do
      # Pre-populate with vectors
      setup_vectors = TestHelpers.generate_random_vectors(10_000, @medium_dimensions, seed: 54321)
      System1.Operations.insert_batch(setup_vectors, %{})
      
      search_count = 1_000
      concurrency_levels = [1, 4, 8, 16, 32]
      
      search_results = 
        concurrency_levels
        |> Enum.map(fn concurrency ->
          IO.puts("Testing #{concurrency} concurrent searches...")
          
          query_vectors = TestHelpers.generate_random_vectors(search_count, @medium_dimensions, 
                                                              seed: concurrency * 1000)
          query_batches = chunk_vectors(query_vectors, concurrency)
          
          {results, total_time} = TestHelpers.measure_time(fn ->
            tasks = 
              query_batches
              |> Enum.map(fn batch ->
                Task.async(fn ->
                  Enum.map(batch, fn query ->
                    System1.Operations.search_knn(query, 10, %{})
                  end)
                end)
              end)
            
            Task.await_many(tasks, 60_000)
          end)
          
          successful_searches = count_successful_searches(results)
          throughput = successful_searches / (total_time / 1000)
          
          %{
            concurrency: concurrency,
            total_time_ms: total_time,
            successful_searches: successful_searches,
            throughput_qps: throughput,
            avg_latency_ms: total_time / successful_searches,
            error_rate: (search_count - successful_searches) / search_count
          }
        end)
      
      analyze_search_concurrency(search_results)
      print_performance_summary("Concurrent Search", search_results)
    end
    
    @tag :stress
    test "mixed workload stress test" do
      IO.puts("Running mixed workload stress test...")
      
      # Setup initial data
      initial_vectors = TestHelpers.generate_random_vectors(5_000, @medium_dimensions, seed: 77777)
      System1.Operations.insert_batch(initial_vectors, %{})
      
      stress_duration_seconds = 30
      
      # Define workload mix
      workload_tasks = [
        # 40% searches
        create_search_workload(0.4, stress_duration_seconds),
        
        # 30% insertions  
        create_insertion_workload(0.3, stress_duration_seconds),
        
        # 20% clustering
        create_clustering_workload(0.2, stress_duration_seconds),
        
        # 10% anomaly detection
        create_anomaly_workload(0.1, stress_duration_seconds)
      ]
      
      # Run mixed workload
      start_time = System.monotonic_time(:millisecond)
      
      task_results = Task.await_many(workload_tasks, (stress_duration_seconds + 10) * 1000)
      
      end_time = System.monotonic_time(:millisecond)
      actual_duration = end_time - start_time
      
      # Analyze mixed workload results
      mixed_workload_analysis = analyze_mixed_workload(task_results, actual_duration)
      
      # Verify system stability under load
      verify_system_stability(mixed_workload_analysis)
      
      print_mixed_workload_results(mixed_workload_analysis)
    end
  end
  
  describe "Memory Efficiency Tests" do
    @tag :memory
    test "memory usage scaling with vector count" do
      dimensions = @medium_dimensions
      vector_counts = [1_000, 5_000, 10_000, 25_000]
      
      memory_scaling = 
        vector_counts
        |> Enum.map(fn count ->
          IO.puts("Testing memory usage with #{count} vectors...")
          
          # Measure baseline memory
          :erlang.garbage_collect()
          baseline_memory = get_system_memory()
          
          # Insert vectors and measure memory growth
          vectors = TestHelpers.generate_random_vectors(count, dimensions, seed: count)
          
          {_, memory_after_generation} = TestHelpers.measure_memory(fn ->
            vectors  # Just hold the vectors in memory
          end)
          
          # Insert into system
          {_, memory_after_insertion} = TestHelpers.measure_memory(fn ->
            System1.Operations.insert_batch(vectors, %{})
          end)
          
          # Build HNSW index
          {_, memory_after_indexing} = TestHelpers.measure_memory(fn ->
            build_hnsw_index(vectors, dimensions)
          end)
          
          :erlang.garbage_collect()
          final_memory = get_system_memory()
          
          %{
            vector_count: count,
            baseline_memory_mb: baseline_memory / 1024 / 1024,
            after_generation_mb: (baseline_memory + memory_after_generation) / 1024 / 1024,
            after_insertion_mb: (baseline_memory + memory_after_insertion) / 1024 / 1024,
            after_indexing_mb: (baseline_memory + memory_after_indexing) / 1024 / 1024,
            final_memory_mb: final_memory / 1024 / 1024,
            memory_per_vector_bytes: (final_memory - baseline_memory) / count,
            memory_efficiency: calculate_memory_efficiency_ratio(count, dimensions, final_memory - baseline_memory)
          }
        end)
      
      analyze_memory_scaling(memory_scaling)
      print_performance_summary("Memory Scaling", memory_scaling)
      
      # Verify memory usage is reasonable (should be roughly linear)
      verify_memory_scaling(memory_scaling)
    end
    
    @tag :memory
    test "memory fragmentation under load" do
      IO.puts("Testing memory fragmentation...")
      
      initial_memory = get_system_memory()
      fragmentation_data = []
      
      # Perform many allocation/deallocation cycles
      1..100
      |> Enum.each(fn iteration ->
        # Allocate vectors
        vectors = TestHelpers.generate_random_vectors(1000, 16, seed: iteration)
        System1.Operations.insert_batch(vectors, %{batch_id: iteration})
        
        # Delete some vectors periodically
        if rem(iteration, 10) == 0 do
          # Delete older batches to create fragmentation
          batches_to_delete = (iteration - 30)..(iteration - 20) |> Enum.to_list()
          Enum.each(batches_to_delete, fn batch_id ->
            System1.Operations.delete_by_metadata(%{batch_id: batch_id})
          end)
          
          # Force garbage collection
          :erlang.garbage_collect()
          
          current_memory = get_system_memory()
          fragmentation_data = [
            %{
              iteration: iteration,
              memory_mb: current_memory / 1024 / 1024,
              fragmentation_ratio: calculate_fragmentation_ratio()
            } | fragmentation_data
          ]
        end
        
        if rem(iteration, 10) == 0 do
          IO.puts("  Iteration #{iteration}: Memory usage #{trunc(get_system_memory() / 1024 / 1024)}MB")
        end
      end)
      
      final_memory = get_system_memory()
      
      fragmentation_analysis = %{
        initial_memory_mb: initial_memory / 1024 / 1024,
        final_memory_mb: final_memory / 1024 / 1024,
        memory_growth_mb: (final_memory - initial_memory) / 1024 / 1024,
        fragmentation_samples: Enum.reverse(fragmentation_data),
        avg_fragmentation: Enum.map(fragmentation_data, & &1.fragmentation_ratio) |> Enum.sum() / length(fragmentation_data)
      }
      
      # Verify fragmentation is within acceptable bounds
      assert fragmentation_analysis.avg_fragmentation < 0.3  # Less than 30% fragmentation
      assert fragmentation_analysis.memory_growth_mb < 500   # Less than 500MB total growth
      
      IO.puts("Memory fragmentation analysis:")
      IO.puts("  Average fragmentation: #{Float.round(fragmentation_analysis.avg_fragmentation * 100, 2)}%")
      IO.puts("  Total memory growth: #{Float.round(fragmentation_analysis.memory_growth_mb, 2)}MB")
    end
  end
  
  # Helper functions for benchmarking
  
  defp build_hnsw_index(vectors, dimensions) do
    graph = HNSW.Graph.new(dimensions: dimensions, max_connections: 16, ml_constant: 1/:math.log(2))
    
    vectors
    |> Enum.with_index()
    |> Enum.reduce(graph, fn {vector, id}, acc_graph ->
      {:ok, updated_graph} = HNSW.Graph.insert_vector(acc_graph, id, vector)
      updated_graph
    end)
  end
  
  defp benchmark_product_quantization(vectors) do
    dimensions = length(hd(vectors))
    subspaces = 8
    bits_per_code = 4
    
    {pq, train_time} = TestHelpers.measure_time(fn ->
      Quantization.ProductQuantization.train(vectors, subspaces, bits_per_code)
    end)
    
    {quantized_vectors, quantize_time} = TestHelpers.measure_time(fn ->
      Enum.map(vectors, &Quantization.ProductQuantization.quantize(pq, &1))
    end)
    
    {reconstructed_vectors, reconstruct_time} = TestHelpers.measure_time(fn ->
      Enum.map(quantized_vectors, &Quantization.ProductQuantization.reconstruct(pq, &1))
    end)
    
    # Calculate metrics
    original_size = length(vectors) * dimensions * 8  # 64-bit floats
    quantized_size = length(vectors) * subspaces * (bits_per_code / 8)
    compression_ratio = original_size / quantized_size
    
    avg_error = 
      vectors
      |> Enum.zip(reconstructed_vectors)
      |> Enum.map(fn {orig, recon} -> 
        TestHelpers.euclidean_distance(orig, recon) / (TestHelpers.vector_norm(orig) + 1.0e-8)
      end)
      |> Enum.sum()
      |> Kernel./(length(vectors))
    
    %{
      train_time_ms: train_time,
      quantize_time_ms: quantize_time,
      reconstruct_time_ms: reconstruct_time,
      compression_ratio: compression_ratio,
      avg_reconstruction_error: avg_error
    }
  end
  
  defp benchmark_scalar_quantization(vectors) do
    bits = 8
    
    {sq, train_time} = TestHelpers.measure_time(fn ->
      Quantization.ScalarQuantization.train(vectors, bits, :uniform)
    end)
    
    {quantized_vectors, quantize_time} = TestHelpers.measure_time(fn ->
      Enum.map(vectors, &Quantization.ScalarQuantization.quantize(sq, &1))
    end)
    
    {dequantized_vectors, dequantize_time} = TestHelpers.measure_time(fn ->
      Enum.map(quantized_vectors, &Quantization.ScalarQuantization.dequantize(sq, &1))
    end)
    
    # Calculate metrics
    dimensions = length(hd(vectors))
    original_size = length(vectors) * dimensions * 8  # 64-bit floats
    quantized_size = length(vectors) * dimensions * (bits / 8)
    compression_ratio = original_size / quantized_size
    
    avg_error = 
      vectors
      |> Enum.zip(dequantized_vectors)
      |> Enum.map(fn {orig, deq} ->
        TestHelpers.euclidean_distance(orig, deq) / (TestHelpers.vector_norm(orig) + 1.0e-8)
      end)
      |> Enum.sum()
      |> Kernel./(length(vectors))
    
    %{
      train_time_ms: train_time,
      quantize_time_ms: quantize_time,
      reconstruct_time_ms: dequantize_time,
      compression_ratio: compression_ratio,
      avg_reconstruction_error: avg_error
    }
  end
  
  defp chunk_vectors(vectors, num_chunks) do
    chunk_size = div(length(vectors), num_chunks)
    Enum.chunk_every(vectors, chunk_size)
  end
  
  defp calculate_speedup(results, current_concurrency, current_throughput) do
    baseline = Enum.find(results, fn r -> r.concurrency == 1 end)
    if baseline do
      current_throughput / baseline.throughput_vps
    else
      1.0
    end
  end
  
  defp count_successful_searches(results) do
    results
    |> List.flatten()
    |> Enum.count(fn result ->
      match?({:ok, _}, result)
    end)
  end
  
  defp get_system_memory() do
    :erlang.memory(:total)
  end
  
  defp calculate_fragmentation_ratio() do
    # Simplified fragmentation estimation
    memory_info = :erlang.memory()
    processes = Keyword.get(memory_info, :processes, 0)
    system = Keyword.get(memory_info, :system, 0)
    total = Keyword.get(memory_info, :total, 1)
    
    unused = total - processes - system
    unused / total
  end
  
  # Analysis functions
  
  defp analyze_scaling_performance(results) do
    IO.puts("\nScaling Performance Analysis:")
    
    Enum.each(results, fn %{size: size, search_time_ms: search_time, build_time_ms: build_time} ->
      IO.puts("  #{size} vectors: search=#{Float.round(search_time, 2)}ms, build=#{Float.round(build_time, 2)}ms")
    end)
  end
  
  defp verify_logarithmic_scaling(results) do
    # Verify that search time grows logarithmically with dataset size
    sorted_results = Enum.sort_by(results, & &1.size)
    
    search_times = Enum.map(sorted_results, & &1.search_time_ms)
    sizes = Enum.map(sorted_results, & &1.size)
    
    # Calculate correlation between log(size) and search_time
    log_sizes = Enum.map(sizes, &:math.log/1)
    correlation = calculate_correlation(log_sizes, search_times)
    
    # Should have reasonable correlation with logarithmic growth
    assert correlation > 0.5, "Search time should scale logarithmically (correlation: #{correlation})"
  end
  
  defp calculate_correlation(list1, list2) do
    n = length(list1)
    mean1 = Enum.sum(list1) / n
    mean2 = Enum.sum(list2) / n
    
    numerator = 
      Enum.zip(list1, list2)
      |> Enum.map(fn {x, y} -> (x - mean1) * (y - mean2) end)
      |> Enum.sum()
    
    denom1 = 
      list1
      |> Enum.map(fn x -> (x - mean1) * (x - mean1) end)
      |> Enum.sum()
      |> :math.sqrt()
    
    denom2 = 
      list2
      |> Enum.map(fn y -> (y - mean2) * (y - mean2) end)
      |> Enum.sum()
      |> :math.sqrt()
    
    if denom1 > 0 and denom2 > 0 do
      numerator / (denom1 * denom2)
    else
      0.0
    end
  end
  
  defp analyze_clustering_performance(results) do
    IO.puts("\nClustering Performance Analysis:")
    
    grouped_by_k = Enum.group_by(results, & &1.k)
    
    Enum.each(grouped_by_k, fn {k, k_results} ->
      IO.puts("  k=#{k}:")
      Enum.each(k_results, fn %{size: size, clustering_time_ms: time, iterations: iters} ->
        IO.puts("    #{size} vectors: #{Float.round(time, 2)}ms (#{iters} iterations)")
      end)
    end)
  end
  
  defp verify_clustering_bounds(results) do
    # Verify clustering completes within reasonable time bounds
    Enum.each(results, fn %{size: size, clustering_time_ms: time, k: k} ->
      max_expected_time = size * k * 0.1  # 0.1ms per vector per cluster (rough bound)
      assert time < max_expected_time, 
        "Clustering too slow: #{time}ms for #{size} vectors, k=#{k} (expected < #{max_expected_time}ms)"
    end)
  end
  
  defp analyze_anomaly_detection_performance(results) do
    IO.puts("\nAnomaly Detection Performance Analysis:")
    
    Enum.each(results, fn %{size: size, train_time_ms: train_time, detect_time_ms: detect_time, f1_score: f1} ->
      IO.puts("  #{size} vectors: train=#{Float.round(train_time, 2)}ms, detect=#{Float.round(detect_time, 2)}ms, F1=#{Float.round(f1, 3)}")
    end)
    
    # Verify detection quality remains good as size increases
    f1_scores = Enum.map(results, & &1.f1_score)
    avg_f1 = Enum.sum(f1_scores) / length(f1_scores)
    
    assert avg_f1 > 0.6, "Average F1 score should be > 0.6, got #{avg_f1}"
  end
  
  defp calculate_precision(detected, true_outliers) do
    true_positives = MapSet.intersection(MapSet.new(detected), MapSet.new(true_outliers)) |> MapSet.size()
    if length(detected) > 0 do
      true_positives / length(detected)
    else
      0.0
    end
  end
  
  defp calculate_recall(detected, true_outliers) do
    true_positives = MapSet.intersection(MapSet.new(detected), MapSet.new(true_outliers)) |> MapSet.size()
    if length(true_outliers) > 0 do
      true_positives / length(true_outliers)
    else
      0.0
    end
  end
  
  defp analyze_concurrency_performance(results) do
    IO.puts("\nConcurrency Performance Analysis:")
    
    Enum.each(results, fn %{concurrency: c, throughput_vps: throughput, speedup: speedup} ->
      IO.puts("  #{c} threads: #{Float.round(throughput, 2)} vectors/sec (#{Float.round(speedup, 2)}x speedup)")
    end)
  end
  
  defp analyze_search_concurrency(results) do
    IO.puts("\nSearch Concurrency Analysis:")
    
    Enum.each(results, fn %{concurrency: c, throughput_qps: qps, avg_latency_ms: latency, error_rate: error_rate} ->
      IO.puts("  #{c} threads: #{Float.round(qps, 2)} queries/sec, #{Float.round(latency, 2)}ms avg latency, #{Float.round(error_rate * 100, 2)}% errors")
    end)
  end
  
  defp calculate_memory_efficiency(size, dimensions) do
    theoretical_minimum = size * dimensions * 8  # Raw vector data in bytes
    actual_usage = get_system_memory()
    theoretical_minimum / actual_usage
  end
  
  defp calculate_memory_efficiency_ratio(vector_count, dimensions, actual_memory) do
    theoretical_minimum = vector_count * dimensions * 8  # 64-bit floats
    theoretical_minimum / actual_memory
  end
  
  defp analyze_memory_scaling(results) do
    IO.puts("\nMemory Scaling Analysis:")
    
    Enum.each(results, fn %{vector_count: count, memory_per_vector_bytes: per_vector, memory_efficiency: efficiency} ->
      IO.puts("  #{count} vectors: #{Float.round(per_vector, 2)} bytes/vector, #{Float.round(efficiency * 100, 2)}% efficiency")
    end)
  end
  
  defp verify_memory_scaling(results) do
    # Memory per vector should remain roughly constant (indicating linear scaling)
    per_vector_bytes = Enum.map(results, & &1.memory_per_vector_bytes)
    min_per_vector = Enum.min(per_vector_bytes)
    max_per_vector = Enum.max(per_vector_bytes)
    
    variation_ratio = max_per_vector / min_per_vector
    
    assert variation_ratio < 2.0, "Memory per vector varies too much: #{variation_ratio}x (should be < 2x)"
  end
  
  defp print_performance_summary(test_name, results) do
    IO.puts("\n=== #{test_name} Performance Summary ===")
    
    case results do
      [%{size: _} | _] ->
        Enum.each(results, fn result ->
          IO.puts("#{inspect(result, pretty: true)}")
        end)
      
      _ ->
        Enum.each(results, fn result ->
          IO.puts("#{inspect(result, pretty: true)}")
        end)
    end
    
    IO.puts("=" <> String.duplicate("=", String.length(test_name) + 25))
  end
  
  defp print_performance_matrix(test_name, matrix) do
    IO.puts("\n=== #{test_name} Matrix ===")
    
    grouped = Enum.group_by(matrix, & &1.size)
    
    Enum.each(grouped, fn {size, size_results} ->
      IO.puts("Size #{size}:")
      Enum.each(size_results, fn result ->
        IO.puts("  #{inspect(result, pretty: true)}")
      end)
    end)
    
    IO.puts("=" <> String.duplicate("=", String.length(test_name) + 12))
  end
  
  # Mixed workload helpers
  
  defp create_search_workload(percentage, duration_seconds) do
    Task.async(fn ->
      end_time = System.monotonic_time(:millisecond) + (duration_seconds * 1000)
      search_count = 0
      successful_searches = 0
      
      search_loop(end_time, search_count, successful_searches)
    end)
  end
  
  defp search_loop(end_time, search_count, successful_searches) do
    if System.monotonic_time(:millisecond) < end_time do
      query = TestHelpers.generate_random_vectors(1, @medium_dimensions) |> hd()
      
      case System1.Operations.search_knn(query, 10, %{}) do
        {:ok, _results} ->
          search_loop(end_time, search_count + 1, successful_searches + 1)
        {:error, _} ->
          search_loop(end_time, search_count + 1, successful_searches)
      end
    else
      %{operation: :search, total: search_count, successful: successful_searches}
    end
  end
  
  defp create_insertion_workload(percentage, duration_seconds) do
    Task.async(fn ->
      end_time = System.monotonic_time(:millisecond) + (duration_seconds * 1000)
      insertion_batches = 0
      successful_insertions = 0
      
      insertion_loop(end_time, insertion_batches, successful_insertions)
    end)
  end
  
  defp insertion_loop(end_time, batches, successful) do
    if System.monotonic_time(:millisecond) < end_time do
      vectors = TestHelpers.generate_random_vectors(50, @medium_dimensions)
      
      case System1.Operations.insert_batch(vectors, %{}) do
        {:ok, _ids} ->
          insertion_loop(end_time, batches + 1, successful + 1)
        {:error, _} ->
          insertion_loop(end_time, batches + 1, successful)
      end
    else
      %{operation: :insertion, total: batches, successful: successful}
    end
  end
  
  defp create_clustering_workload(percentage, duration_seconds) do
    Task.async(fn ->
      end_time = System.monotonic_time(:millisecond) + (duration_seconds * 1000)
      clustering_attempts = 0
      successful_clusters = 0
      
      clustering_loop(end_time, clustering_attempts, successful_clusters)
    end)
  end
  
  defp clustering_loop(end_time, attempts, successful) do
    if System.monotonic_time(:millisecond) < end_time do
      case System4.ML.cluster_kmeans(5, %{}) do
        {:ok, _result} ->
          Process.sleep(1000)  # Clustering is expensive, don't do it too frequently
          clustering_loop(end_time, attempts + 1, successful + 1)
        {:error, _} ->
          clustering_loop(end_time, attempts + 1, successful)
      end
    else
      %{operation: :clustering, total: attempts, successful: successful}
    end
  end
  
  defp create_anomaly_workload(percentage, duration_seconds) do
    Task.async(fn ->
      end_time = System.monotonic_time(:millisecond) + (duration_seconds * 1000)
      anomaly_detections = 0
      successful_detections = 0
      
      anomaly_loop(end_time, anomaly_detections, successful_detections)
    end)
  end
  
  defp anomaly_loop(end_time, detections, successful) do
    if System.monotonic_time(:millisecond) < end_time do
      case System4.AnomalyDetection.detect(%{}) do
        {:ok, _anomalies} ->
          Process.sleep(2000)  # Anomaly detection is expensive
          anomaly_loop(end_time, detections + 1, successful + 1)
        {:error, _} ->
          anomaly_loop(end_time, detections + 1, successful)
      end
    else
      %{operation: :anomaly_detection, total: detections, successful: successful}
    end
  end
  
  defp analyze_mixed_workload(results, duration_ms) do
    total_operations = Enum.map(results, & &1.total) |> Enum.sum()
    total_successful = Enum.map(results, & &1.successful) |> Enum.sum()
    
    overall_throughput = total_successful / (duration_ms / 1000)
    success_rate = total_successful / total_operations
    
    operation_breakdown = 
      results
      |> Enum.map(fn %{operation: op, total: total, successful: success} ->
        {op, %{total: total, successful: success, success_rate: success / total}}
      end)
      |> Map.new()
    
    %{
      duration_ms: duration_ms,
      total_operations: total_operations,
      successful_operations: total_successful,
      overall_throughput: overall_throughput,
      overall_success_rate: success_rate,
      operation_breakdown: operation_breakdown
    }
  end
  
  defp verify_system_stability(analysis) do
    # System should maintain reasonable success rate under mixed load
    assert analysis.overall_success_rate > 0.85, 
      "System success rate too low under load: #{analysis.overall_success_rate}"
    
    # Each operation type should have reasonable success rate
    Enum.each(analysis.operation_breakdown, fn {operation, metrics} ->
      assert metrics.success_rate > 0.8,
        "#{operation} success rate too low: #{metrics.success_rate}"
    end)
  end
  
  defp print_mixed_workload_results(analysis) do
    IO.puts("\n=== Mixed Workload Results ===")
    IO.puts("Duration: #{analysis.duration_ms}ms")
    IO.puts("Total operations: #{analysis.total_operations}")
    IO.puts("Successful operations: #{analysis.successful_operations}")
    IO.puts("Overall throughput: #{Float.round(analysis.overall_throughput, 2)} ops/sec")
    IO.puts("Overall success rate: #{Float.round(analysis.overall_success_rate * 100, 2)}%")
    
    IO.puts("\nOperation Breakdown:")
    Enum.each(analysis.operation_breakdown, fn {operation, metrics} ->
      IO.puts("  #{operation}: #{metrics.total} total, #{metrics.successful} successful (#{Float.round(metrics.success_rate * 100, 2)}%)")
    end)
    
    IO.puts("=" <> String.duplicate("=", 30))
  end
end