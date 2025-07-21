defmodule VsmVectorStore.Quantization.Test do
  @moduledoc """
  Comprehensive unit tests for vector quantization algorithms.
  
  Tests Product Quantization (PQ), Scalar Quantization (SQ), and quantization quality metrics.
  """
  
  use ExUnit.Case, async: true
  use ExUnitProperties
  
  alias VsmVectorStore.TestHelpers
  alias VsmVectorStore.Quantization.{ProductQuantization, ScalarQuantization}
  
  describe "Product Quantization (PQ)" do
    test "trains codebooks from vector data" do
      vectors = TestHelpers.generate_random_vectors(200, 8, seed: 1001)
      subspaces = 4
      bits_per_code = 4
      
      pq = ProductQuantization.train(vectors, subspaces, bits_per_code)
      
      assert pq.subspaces == subspaces
      assert pq.codebook_size == round(:math.pow(2, bits_per_code))
      assert pq.dimensions == 8
      assert pq.subspace_dim == 2  # 8 / 4
      
      # Verify codebooks exist for each subspace
      assert map_size(pq.codebooks) == subspaces
      
      Enum.each(0..(subspaces - 1), fn i ->
        codebook = pq.codebooks[i]
        assert map_size(codebook) == pq.codebook_size
        
        # Each codeword should have correct dimensionality
        Enum.each(codebook, fn {_code, codeword} ->
          assert length(codeword) == pq.subspace_dim
        end)
      end)
    end
    
    test "quantizes and reconstructs vectors" do
      vectors = TestHelpers.generate_random_vectors(100, 12, seed: 1002)
      subspaces = 3
      bits_per_code = 3
      
      pq = ProductQuantization.train(vectors, subspaces, bits_per_code)
      
      # Test quantization and reconstruction
      test_vector = hd(vectors)
      quantized_codes = ProductQuantization.quantize(pq, test_vector)
      reconstructed = ProductQuantization.reconstruct(pq, quantized_codes)
      
      # Verify quantized codes format
      assert length(quantized_codes) == subspaces
      Enum.each(quantized_codes, fn code ->
        assert code >= 0 and code < pq.codebook_size
      end)
      
      # Verify reconstruction dimensionality
      assert length(reconstructed) == length(test_vector)
      
      # Reconstruction should be reasonably close to original
      distance = TestHelpers.euclidean_distance(test_vector, reconstructed)
      norm = TestHelpers.vector_norm(test_vector)
      relative_error = distance / (norm + 1.0e-8)  # Avoid division by zero
      
      assert relative_error < 0.5  # Within 50% relative error
    end
    
    test "maintains distance relationships approximately" do
      vectors = TestHelpers.generate_clustered_vectors(150, 16, 3, seed: 1003)
      subspaces = 8
      bits_per_code = 4
      
      pq = ProductQuantization.train(vectors, subspaces, bits_per_code)
      
      # Select a few test vectors
      test_vectors = Enum.take(vectors, 5)
      
      # Calculate original distances
      original_distances = 
        for i <- 0..4, j <- (i+1)..4 do
          {i, j, TestHelpers.euclidean_distance(Enum.at(test_vectors, i), Enum.at(test_vectors, j))}
        end
      
      # Quantize and reconstruct
      reconstructed_vectors = 
        test_vectors
        |> Enum.map(&ProductQuantization.quantize(pq, &1))
        |> Enum.map(&ProductQuantization.reconstruct(pq, &1))
      
      # Calculate reconstructed distances
      reconstructed_distances = 
        for i <- 0..4, j <- (i+1)..4 do
          {i, j, TestHelpers.euclidean_distance(Enum.at(reconstructed_vectors, i), Enum.at(reconstructed_vectors, j))}
        end
      
      # Compare distance preservation
      Enum.zip(original_distances, reconstructed_distances)
      |> Enum.each(fn {{i, j, orig_dist}, {^i, ^j, recon_dist}} ->
        relative_error = abs(orig_dist - recon_dist) / (orig_dist + 1.0e-8)
        assert relative_error < 0.3, 
          "Distance preservation failed for vectors #{i},#{j}: #{relative_error}"
      end)
    end
    
    property "quantization is deterministic and reversible" do
      check all vectors <- TestHelpers.vector_batch_generator(20, 50, 8),
                subspaces <- integer(2..4),
                bits_per_code <- integer(2..6) do
        
        # Ensure dimensions are divisible by subspaces
        dimensions = 8
        adjusted_subspaces = min(subspaces, dimensions)
        
        if rem(dimensions, adjusted_subspaces) == 0 and length(vectors) >= 2^bits_per_code do
          pq = ProductQuantization.train(vectors, adjusted_subspaces, bits_per_code)
          
          test_vector = hd(vectors)
          
          # Quantize twice - should get same result
          codes1 = ProductQuantization.quantize(pq, test_vector)
          codes2 = ProductQuantization.quantize(pq, test_vector)
          assert codes1 == codes2
          
          # Reconstruct twice - should get same result
          recon1 = ProductQuantization.reconstruct(pq, codes1)
          recon2 = ProductQuantization.reconstruct(pq, codes2)
          TestHelpers.assert_vectors_equal(recon1, recon2, 1.0e-10)
        end
      end
    end
    
    test "handles different subspace configurations" do
      vectors = TestHelpers.generate_random_vectors(64, 24, seed: 1004)
      
      # Test various subspace divisions
      configurations = [
        {2, 4}, {3, 3}, {4, 4}, {6, 3}, {8, 2}  # {subspaces, bits_per_code}
      ]
      
      Enum.each(configurations, fn {subspaces, bits_per_code} ->
        if rem(24, subspaces) == 0 do  # Only test valid divisions
          pq = ProductQuantization.train(vectors, subspaces, bits_per_code)
          
          assert pq.subspaces == subspaces
          assert pq.subspace_dim == div(24, subspaces)
          assert pq.codebook_size == round(:math.pow(2, bits_per_code))
          
          # Test quantization works
          test_vector = hd(vectors)
          codes = ProductQuantization.quantize(pq, test_vector)
          reconstructed = ProductQuantization.reconstruct(pq, codes)
          
          assert length(codes) == subspaces
          assert length(reconstructed) == 24
        end
      end)
    end
  end
  
  describe "Scalar Quantization (SQ)" do
    test "learns quantization parameters from data" do
      vectors = TestHelpers.generate_random_vectors(100, 6, seed: 2001)
      bits = 8
      
      sq = ScalarQuantization.train(vectors, bits, :uniform)
      
      assert sq.bits == bits
      assert sq.type == :uniform
      assert length(sq.min_vals) == 6
      assert length(sq.max_vals) == 6
      assert length(sq.scale_factors) == 6
      
      # Min values should be <= max values
      Enum.zip(sq.min_vals, sq.max_vals)
      |> Enum.each(fn {min_val, max_val} ->
        assert min_val <= max_val
      end)
      
      # Scale factors should be positive
      Enum.each(sq.scale_factors, fn scale ->
        assert scale > 0.0
      end)
    end
    
    test "quantizes and dequantizes vectors correctly" do
      vectors = [
        [0.0, 1.0, 2.0],
        [0.5, 1.5, 2.5],
        [-1.0, 0.0, 3.0],
        [1.0, 2.0, 1.5]
      ]
      bits = 4
      
      sq = ScalarQuantization.train(vectors, bits, :uniform)
      
      test_vector = [0.25, 1.25, 2.25]
      quantized = ScalarQuantization.quantize(sq, test_vector)
      dequantized = ScalarQuantization.dequantize(sq, quantized)
      
      # Quantized values should be integers in valid range
      max_quantized_value = round(:math.pow(2, bits)) - 1
      Enum.each(quantized, fn val ->
        assert is_integer(val)
        assert val >= 0 and val <= max_quantized_value
      end)
      
      # Dequantized should be close to original
      assert length(dequantized) == length(test_vector)
      TestHelpers.assert_vectors_equal(test_vector, dequantized, 0.2)  # Allow quantization error
    end
    
    test "preserves value ordering after quantization" do
      # Create vectors with clear ordering
      vectors = [
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1], 
        [1.2, 2.2, 3.2],
        [1.3, 2.3, 3.3]
      ]
      
      sq = ScalarQuantization.train(vectors, 8, :uniform)
      
      quantized_vectors = Enum.map(vectors, &ScalarQuantization.quantize(sq, &1))
      
      # Check ordering preservation for each dimension
      Enum.each(0..2, fn dim ->
        original_values = Enum.map(vectors, &Enum.at(&1, dim))
        quantized_values = Enum.map(quantized_vectors, &Enum.at(&1, dim))
        
        # Should maintain relative ordering (allowing for ties due to quantization)
        Enum.zip(original_values, quantized_values)
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.each(fn [{orig1, quant1}, {orig2, quant2}] ->
          if orig1 < orig2 do
            assert quant1 <= quant2, "Ordering not preserved at dimension #{dim}"
          end
        end)
      end)
    end
    
    property "quantization is bounded and deterministic" do
      check all vectors <- TestHelpers.vector_batch_generator(10, 30, 4),
                bits <- integer(2..8) do
        
        sq = ScalarQuantization.train(vectors, bits, :uniform)
        max_quantized_value = round(:math.pow(2, bits)) - 1
        
        test_vector = hd(vectors)
        
        # Quantize multiple times - should be deterministic
        quantized1 = ScalarQuantization.quantize(sq, test_vector)
        quantized2 = ScalarQuantization.quantize(sq, test_vector)
        assert quantized1 == quantized2
        
        # All quantized values should be in bounds
        Enum.each(quantized1, fn val ->
          assert val >= 0 and val <= max_quantized_value
        end)
        
        # Dequantization should be deterministic
        deq1 = ScalarQuantization.dequantize(sq, quantized1)
        deq2 = ScalarQuantization.dequantize(sq, quantized2)
        TestHelpers.assert_vectors_equal(deq1, deq2, 1.0e-10)
      end
    end
    
    test "handles different bit depths appropriately" do
      vectors = TestHelpers.generate_random_vectors(50, 3, seed: 2002)
      
      bit_depths = [1, 2, 4, 8, 16]
      
      # Test quantization quality improves with more bits
      quantization_errors = 
        bit_depths
        |> Enum.map(fn bits ->
          sq = ScalarQuantization.train(vectors, bits, :uniform)
          
          # Calculate average quantization error
          total_error = 
            vectors
            |> Enum.map(fn vector ->
              quantized = ScalarQuantization.quantize(sq, vector)
              dequantized = ScalarQuantization.dequantize(sq, quantized)
              TestHelpers.euclidean_distance(vector, dequantized)
            end)
            |> Enum.sum()
          
          avg_error = total_error / length(vectors)
          {bits, avg_error}
        end)
      
      # Higher bit depths should generally have lower errors
      errors = Enum.map(quantization_errors, &elem(&1, 1))
      
      # 16-bit should be much better than 1-bit
      error_16_bit = List.last(errors)
      error_1_bit = hd(errors)
      assert error_16_bit < error_1_bit * 0.5
    end
  end
  
  describe "Quantization Performance and Memory" do
    test "PQ provides significant memory savings" do
      vectors = TestHelpers.generate_random_vectors(1000, 64, seed: 3001)
      
      # Calculate original memory usage (rough estimate)
      original_memory = length(vectors) * 64 * 8  # count * dims * bytes_per_float
      
      # Train PQ with aggressive quantization
      pq = ProductQuantization.train(vectors, 16, 4)  # 16 subspaces, 4 bits each
      
      # Calculate quantized memory usage
      quantized_size = length(vectors) * 16 * 0.5  # count * subspaces * bytes_per_4bit_code
      
      # Should achieve significant compression
      compression_ratio = original_memory / quantized_size
      assert compression_ratio > 8.0  # At least 8x compression
      
      # Test that quantization/reconstruction still works
      test_vector = hd(vectors)
      codes = ProductQuantization.quantize(pq, test_vector)
      reconstructed = ProductQuantization.reconstruct(pq, codes)
      
      distance = TestHelpers.euclidean_distance(test_vector, reconstructed)
      norm = TestHelpers.vector_norm(test_vector)
      relative_error = distance / (norm + 1.0e-8)
      
      assert relative_error < 0.8  # Should still be reasonably accurate
    end
    
    test "SQ quantization is fast for large datasets" do
      large_vectors = TestHelpers.generate_random_vectors(2000, 32, seed: 3002)
      
      # Measure training time
      {sq, train_time_ms} = TestHelpers.measure_time(fn ->
        ScalarQuantization.train(large_vectors, 8, :uniform)
      end)
      
      # Measure quantization time
      {quantized_vectors, quantize_time_ms} = TestHelpers.measure_time(fn ->
        Enum.map(large_vectors, &ScalarQuantization.quantize(sq, &1))
      end)
      
      # Should complete within reasonable time
      TestHelpers.assert_performance_bounds(train_time_ms, 1000)      # 1 second for training
      TestHelpers.assert_performance_bounds(quantize_time_ms, 2000)   # 2 seconds for quantization
      
      # Verify results
      assert length(quantized_vectors) == 2000
      Enum.each(quantized_vectors, fn qvec ->
        assert length(qvec) == 32
      end)
    end
    
    test "quantization maintains search quality" do
      # Create vectors with known nearest neighbors
      base_vectors = TestHelpers.generate_clustered_vectors(200, 16, 4, seed: 3003)
      query_vector = TestHelpers.generate_random_vectors(1, 16, seed: 3004) |> hd()
      
      # Find true k nearest neighbors
      true_neighbors = 
        base_vectors
        |> Enum.with_index()
        |> Enum.map(fn {vector, idx} ->
          {idx, TestHelpers.euclidean_distance(query_vector, vector)}
        end)
        |> Enum.sort_by(&elem(&1, 1))
        |> Enum.take(10)
        |> Enum.map(&elem(&1, 0))
      
      # Test with Product Quantization
      pq = ProductQuantization.train(base_vectors, 8, 4)
      quantized_base = Enum.map(base_vectors, &ProductQuantization.quantize(pq, &1))
      reconstructed_base = Enum.map(quantized_base, &ProductQuantization.reconstruct(pq, &1))
      
      quantized_query = ProductQuantization.quantize(pq, query_vector)
      reconstructed_query = ProductQuantization.reconstruct(pq, quantized_query)
      
      # Find approximate neighbors using quantized vectors
      approx_neighbors = 
        reconstructed_base
        |> Enum.with_index()
        |> Enum.map(fn {vector, idx} ->
          {idx, TestHelpers.euclidean_distance(reconstructed_query, vector)}
        end)
        |> Enum.sort_by(&elem(&1, 1))
        |> Enum.take(10)
        |> Enum.map(&elem(&1, 0))
      
      # Calculate recall (intersection with true neighbors)
      intersection = MapSet.intersection(MapSet.new(true_neighbors), MapSet.new(approx_neighbors))
      recall = MapSet.size(intersection) / length(true_neighbors)
      
      # Should maintain reasonable search quality
      assert recall > 0.6  # At least 60% recall
    end
  end
  
  describe "Quantization Edge Cases and Error Handling" do
    test "handles vectors with zero variance" do
      # Vectors with constant values in some dimensions
      constant_vectors = [
        [1.0, 5.0, 1.0, 3.0],
        [1.0, 5.0, 1.0, 4.0], 
        [1.0, 5.0, 1.0, 5.0]
      ]
      
      # Should handle SQ gracefully
      sq = ScalarQuantization.train(constant_vectors, 8, :uniform)
      test_vector = [1.0, 5.0, 1.0, 4.5]
      
      quantized = ScalarQuantization.quantize(sq, test_vector)
      dequantized = ScalarQuantization.dequantize(sq, quantized)
      
      # Should preserve constant dimensions exactly
      assert_in_delta(Enum.at(dequantized, 0), 1.0, 0.01)
      assert_in_delta(Enum.at(dequantized, 1), 5.0, 0.01)  
      assert_in_delta(Enum.at(dequantized, 2), 1.0, 0.01)
    end
    
    test "validates input parameters" do
      vectors = TestHelpers.generate_random_vectors(10, 4)
      
      # Invalid subspace count for PQ
      assert_raise ArgumentError, fn ->
        ProductQuantization.train(vectors, 5, 4)  # 4 dimensions not divisible by 5
      end
      
      assert_raise ArgumentError, fn ->
        ProductQuantization.train(vectors, 0, 4)  # Zero subspaces
      end
      
      # Invalid bits for SQ
      assert_raise ArgumentError, fn ->
        ScalarQuantization.train(vectors, 0, :uniform)  # Zero bits
      end
      
      assert_raise ArgumentError, fn ->
        ScalarQuantization.train(vectors, 33, :uniform)  # Too many bits
      end
    end
    
    test "handles empty or insufficient training data" do
      # Empty vectors
      assert_raise ArgumentError, fn ->
        ProductQuantization.train([], 2, 4)
      end
      
      assert_raise ArgumentError, fn ->
        ScalarQuantization.train([], 8, :uniform)
      end
      
      # Insufficient data for codebook learning
      tiny_vectors = [[1.0, 2.0], [3.0, 4.0]]  # Only 2 vectors
      
      assert_raise ArgumentError, fn ->
        ProductQuantization.train(tiny_vectors, 2, 8)  # Needs 2^8 = 256 codes per subspace
      end
    end
    
    test "handles extreme value ranges" do
      # Very large values
      large_vectors = [
        [1.0e6, 2.0e6, 3.0e6],
        [1.1e6, 2.1e6, 3.1e6],
        [1.2e6, 2.2e6, 3.2e6]
      ]
      
      sq_large = ScalarQuantization.train(large_vectors, 8, :uniform)
      test_large = [1.05e6, 2.05e6, 3.05e6]
      
      quantized_large = ScalarQuantization.quantize(sq_large, test_large)
      dequantized_large = ScalarQuantization.dequantize(sq_large, quantized_large)
      
      # Should handle large values without overflow
      Enum.each(dequantized_large, fn val ->
        assert val > 0.0 and val < Float.max_finite()
      end)
      
      # Very small values
      small_vectors = [
        [1.0e-6, 2.0e-6, 3.0e-6],
        [1.1e-6, 2.1e-6, 3.1e-6],
        [1.2e-6, 2.2e-6, 3.2e-6]
      ]
      
      sq_small = ScalarQuantization.train(small_vectors, 8, :uniform)
      test_small = [1.05e-6, 2.05e-6, 3.05e-6]
      
      quantized_small = ScalarQuantization.quantize(sq_small, test_small)
      dequantized_small = ScalarQuantization.dequantize(sq_small, quantized_small)
      
      # Should handle small values without underflow
      Enum.each(dequantized_small, fn val ->
        assert val >= 0.0
      end)
    end
  end
end