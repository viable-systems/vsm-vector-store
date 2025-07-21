defmodule VSMVectorStoreIntegrationTest do
  use ExUnit.Case
  require Logger
  
  @moduledoc """
  Comprehensive integration test demonstrating the full functionality
  of the VSM Vector Store with actual vector storage and search.
  """
  
  setup do
    # Ensure the application is started
    {:ok, _} = Application.ensure_all_started(:vsm_vector_store)
    
    # Clean up any existing test spaces
    case VSMVectorStore.list_spaces() do
      {:ok, spaces} ->
        spaces
        |> Enum.filter(fn space -> String.starts_with?(space.name, "test_") end)
        |> Enum.each(fn space -> VSMVectorStore.delete_space(space.id) end)
      _ -> :ok
    end
    
    :ok
  end
  
  describe "Vector Space Management" do
    test "creates and manages vector spaces" do
      # Create a space
      {:ok, space_id} = VSMVectorStore.create_space("test_space", 128)
      assert is_binary(space_id)
      
      # List spaces
      {:ok, spaces} = VSMVectorStore.list_spaces()
      assert Enum.any?(spaces, fn s -> s.id == space_id end)
      
      # Get space info
      {:ok, space_info} = VSMVectorStore.get_space_info(space_id)
      assert space_info.name == "test_space"
      assert space_info.dimensions == 128
      assert space_info.vector_count == 0
      
      # Delete space
      :ok = VSMVectorStore.delete_space(space_id)
      
      # Verify deletion
      {:ok, spaces} = VSMVectorStore.list_spaces()
      refute Enum.any?(spaces, fn s -> s.id == space_id end)
    end
  end
  
  describe "Vector Operations" do
    setup do
      {:ok, space_id} = VSMVectorStore.create_space("test_vectors", 64)
      on_exit(fn -> VSMVectorStore.delete_space(space_id) end)
      {:ok, space_id: space_id}
    end
    
    test "inserts and retrieves vectors", %{space_id: space_id} do
      # Create a test vector
      vector = Enum.map(1..64, fn i -> i / 64.0 end)
      metadata = %{type: "test", index: 1}
      
      # Insert vector
      :ok = VSMVectorStore.insert(space_id, "vec1", vector, metadata)
      
      # Retrieve vector
      {:ok, {retrieved_vector, retrieved_metadata}} = VSMVectorStore.get(space_id, "vec1")
      assert retrieved_vector == vector
      assert retrieved_metadata == metadata
      
      # Get space stats
      {:ok, stats} = VSMVectorStore.stats(space_id)
      assert stats.vector_count == 1
    end
    
    test "batch inserts multiple vectors", %{space_id: space_id} do
      # Generate test vectors
      vectors = for i <- 1..10 do
        vector = Enum.map(1..64, fn _ -> :rand.uniform() end)
        {"vec#{i}", vector, %{index: i}}
      end
      
      # Batch insert
      {:ok, count} = VSMVectorStore.batch_insert(space_id, vectors)
      assert count == 10
      
      # Verify all inserted
      {:ok, stats} = VSMVectorStore.stats(space_id)
      assert stats.vector_count == 10
    end
    
    test "searches for similar vectors", %{space_id: space_id} do
      # Insert test vectors in a pattern
      base_vector = Enum.map(1..64, fn _ -> 0.5 end)
      
      # Insert similar vectors with small variations
      for i <- 1..20 do
        vector = Enum.map(base_vector, fn v -> 
          v + (:rand.uniform() - 0.5) * 0.1 * i
        end)
        :ok = VSMVectorStore.insert(space_id, "vec#{i}", vector, %{distance: i})
      end
      
      # Search for vectors similar to base
      {:ok, results} = VSMVectorStore.search(space_id, base_vector, 5)
      
      assert length(results) == 5
      assert is_list(results)
      
      # Results should be sorted by distance
      distances = Enum.map(results, fn {_id, dist, _meta} -> dist end)
      assert distances == Enum.sort(distances)
      
      # Closest vectors should have lower indices
      {closest_id, _dist, metadata} = hd(results)
      assert metadata.distance <= 5
    end
    
    test "updates vector metadata", %{space_id: space_id} do
      vector = Enum.map(1..64, fn _ -> :rand.uniform() end)
      
      # Insert with initial metadata
      :ok = VSMVectorStore.insert(space_id, "vec1", vector, %{version: 1})
      
      # Update metadata
      :ok = VSMVectorStore.update_metadata(space_id, "vec1", %{version: 2, updated: true})
      
      # Verify update
      {:ok, {_vector, metadata}} = VSMVectorStore.get(space_id, "vec1")
      assert metadata.version == 2
      assert metadata.updated == true
    end
    
    test "deletes vectors", %{space_id: space_id} do
      vector = Enum.map(1..64, fn _ -> :rand.uniform() end)
      
      # Insert and verify
      :ok = VSMVectorStore.insert(space_id, "vec1", vector, %{})
      {:ok, _} = VSMVectorStore.get(space_id, "vec1")
      
      # Delete
      :ok = VSMVectorStore.delete(space_id, "vec1")
      
      # Verify deletion
      assert {:error, :vector_not_found} = VSMVectorStore.get(space_id, "vec1")
    end
  end
  
  describe "Advanced Features" do
    setup do
      {:ok, space_id} = VSMVectorStore.create_space("test_advanced", 32)
      on_exit(fn -> VSMVectorStore.delete_space(space_id) end)
      {:ok, space_id: space_id}
    end
    
    test "handles high-dimensional vectors", %{space_id: _} do
      # Create a high-dimensional space
      {:ok, hd_space} = VSMVectorStore.create_space("test_high_dim", 1024)
      
      # Insert high-dimensional vectors
      for i <- 1..5 do
        vector = Enum.map(1..1024, fn _ -> :rand.uniform() end)
        :ok = VSMVectorStore.insert(hd_space, "hd_vec#{i}", vector, %{})
      end
      
      # Search in high dimensions
      query = Enum.map(1..1024, fn _ -> :rand.uniform() end)
      {:ok, results} = VSMVectorStore.search(hd_space, query, 3)
      assert length(results) == 3
      
      VSMVectorStore.delete_space(hd_space)
    end
    
    test "performs similarity search with different metrics", %{space_id: space_id} do
      # Insert normalized vectors for cosine similarity
      for i <- 1..10 do
        vector = Enum.map(1..32, fn j -> 
          :math.sin(i * j * :math.pi() / 32)
        end)
        normalized = VsmVectorStore.Storage.VectorOps.normalize(vector)
        :ok = VSMVectorStore.insert(space_id, "norm_vec#{i}", normalized, %{wave: i})
      end
      
      # Search with a similar wave pattern
      query = Enum.map(1..32, fn j -> :math.sin(5.5 * j * :math.pi() / 32) end)
      normalized_query = VsmVectorStore.Storage.VectorOps.normalize(query)
      
      {:ok, results} = VSMVectorStore.search(space_id, normalized_query, 3)
      
      # Should find vectors with similar wave patterns (around wave 5-6)
      top_waves = results
      |> Enum.map(fn {_id, _dist, meta} -> meta.wave end)
      |> Enum.sort()
      
      assert 5 in top_waves or 6 in top_waves
    end
    
    test "handles concurrent operations", %{space_id: space_id} do
      # Spawn multiple processes to insert vectors concurrently
      tasks = for i <- 1..20 do
        Task.async(fn ->
          vector = Enum.map(1..32, fn _ -> :rand.uniform() end)
          VSMVectorStore.insert(space_id, "concurrent_#{i}", vector, %{task: i})
        end)
      end
      
      # Wait for all insertions
      results = Task.await_many(tasks)
      assert Enum.all?(results, &(&1 == :ok))
      
      # Verify all vectors were inserted
      {:ok, stats} = VSMVectorStore.stats(space_id)
      assert stats.vector_count == 20
      
      # Concurrent searches
      search_tasks = for _ <- 1..10 do
        Task.async(fn ->
          query = Enum.map(1..32, fn _ -> :rand.uniform() end)
          VSMVectorStore.search(space_id, query, 5)
        end)
      end
      
      search_results = Task.await_many(search_tasks)
      assert Enum.all?(search_results, fn {:ok, results} -> length(results) == 5 end)
    end
  end
  
  describe "Performance and Scale" do
    @tag :performance
    test "handles large-scale insertions and searches" do
      {:ok, space_id} = VSMVectorStore.create_space("test_scale", 128)
      
      try do
        # Insert 1000 vectors
        Logger.info("Inserting 1000 vectors...")
        insert_start = System.monotonic_time(:millisecond)
        
        for i <- 1..1000 do
          vector = Enum.map(1..128, fn _ -> :rand.uniform() end)
          :ok = VSMVectorStore.insert(space_id, "scale_#{i}", vector, %{index: i})
          
          if rem(i, 100) == 0 do
            Logger.info("Inserted #{i} vectors")
          end
        end
        
        insert_time = System.monotonic_time(:millisecond) - insert_start
        Logger.info("Insert time: #{insert_time}ms (#{1000 * 1000 / insert_time} vectors/sec)")
        
        # Perform searches
        Logger.info("Performing 100 searches...")
        search_start = System.monotonic_time(:millisecond)
        
        search_times = for _ <- 1..100 do
          query = Enum.map(1..128, fn _ -> :rand.uniform() end)
          start = System.monotonic_time(:microsecond)
          {:ok, _results} = VSMVectorStore.search(space_id, query, 10)
          System.monotonic_time(:microsecond) - start
        end
        
        search_time = System.monotonic_time(:millisecond) - search_start
        avg_search_time = Enum.sum(search_times) / length(search_times) / 1000
        
        Logger.info("Search time: #{search_time}ms total")
        Logger.info("Average search time: #{Float.round(avg_search_time, 2)}ms")
        
        # Verify final state
        {:ok, stats} = VSMVectorStore.stats(space_id)
        assert stats.vector_count == 1000
        assert stats.dimension == 128
        
      after
        VSMVectorStore.delete_space(space_id)
      end
    end
  end
  
  describe "Error Handling" do
    test "handles dimension mismatches" do
      {:ok, space_id} = VSMVectorStore.create_space("test_dims", 64)
      
      try do
        # Try to insert wrong dimension
        wrong_vector = Enum.map(1..32, fn _ -> :rand.uniform() end)
        assert {:error, :dimension_mismatch} = 
          VSMVectorStore.insert(space_id, "wrong", wrong_vector, %{})
        
        # Try to search with wrong dimension
        wrong_query = Enum.map(1..128, fn _ -> :rand.uniform() end)
        assert {:error, :dimension_mismatch} = 
          VSMVectorStore.search(space_id, wrong_query, 5)
      after
        VSMVectorStore.delete_space(space_id)
      end
    end
    
    test "handles non-existent vectors and spaces" do
      # Non-existent space
      assert {:error, :space_not_found} = 
        VSMVectorStore.get("non_existent_space", "vec1")
      
      # Non-existent vector
      {:ok, space_id} = VSMVectorStore.create_space("test_errors", 32)
      
      try do
        assert {:error, :vector_not_found} = 
          VSMVectorStore.get(space_id, "non_existent")
      after
        VSMVectorStore.delete_space(space_id)
      end
    end
  end
end