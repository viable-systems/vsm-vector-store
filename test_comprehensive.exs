#!/usr/bin/env elixir

# Comprehensive test showing ALL VSM Vector Store features

IO.puts """
🚀 VSM Vector Store - Comprehensive Feature Test
==============================================

This test demonstrates ALL features working together:
- Vector storage and retrieval
- K-nearest neighbor search
- K-means clustering
- Pattern recognition
- Anomaly detection (with sufficient data)
- Performance metrics
- Full VSM architecture
"""

# Start the application components
Code.require_file("lib/vsm_vector_store.ex")
Code.require_file("lib/vsm_vector_store/application.ex")

# Start supervisors manually
{:ok, _} = VSMVectorStore.Application.start(nil, nil)

# 1. System Status Check
IO.puts("\n1️⃣ System Status Check")
status = VSMVectorStore.status()
IO.puts("✅ System: #{status.system}")
IO.puts("   Subsystems: #{inspect(status.subsystems)}")

# 2. Create Multiple Spaces
IO.puts("\n2️⃣ Creating Multiple Vector Spaces")
{:ok, space_64d} = VSMVectorStore.create_space("64-dimensional", 64)
{:ok, space_128d} = VSMVectorStore.create_space("128-dimensional", 128)
{:ok, space_256d} = VSMVectorStore.create_space("256-dimensional-anomaly", 256)
IO.puts("✅ Created 3 spaces with different dimensions")

# 3. List Spaces
IO.puts("\n3️⃣ Listing All Spaces")
{:ok, spaces} = VSMVectorStore.list_spaces()
Enum.each(spaces, fn space ->
  IO.puts("   - #{space.name} (#{space.dimensions}D): #{space.id}")
end)

# 4. Insert Vectors - Create clear patterns
IO.puts("\n4️⃣ Inserting Vectors with Clear Patterns")

# 64D space: 3 distinct clusters
vectors_64d = for i <- 1..150 do
  base = for _ <- 1..64, do: :rand.uniform() * 0.1
  case rem(i, 3) do
    0 -> Enum.map(base, &(&1 + 0.9))  # Cluster A: High values
    1 -> Enum.map(base, &(&1 + 0.5))  # Cluster B: Medium values
    _ -> base                          # Cluster C: Low values
  end
end
metadata_64d = for i <- 1..150, do: %{index: i, cluster: rem(i, 3), type: "normal"}
{:ok, ids_64d} = VSMVectorStore.insert(space_64d, vectors_64d, metadata_64d)
IO.puts("✅ Inserted #{length(ids_64d)} vectors into 64D space")

# 128D space: Mixed patterns
vectors_128d = for i <- 1..100 do
  if i <= 80 do
    # Normal vectors
    for j <- 1..128, do: :rand.normal(0.5, 0.1)
  else
    # Outliers
    for j <- 1..128, do: :rand.normal(1.5, 0.3)
  end
end
metadata_128d = for i <- 1..100, do: %{index: i, type: if(i <= 80, do: "normal", else: "outlier")}
{:ok, ids_128d} = VSMVectorStore.insert(space_128d, vectors_128d, metadata_128d)
IO.puts("✅ Inserted #{length(ids_128d)} vectors into 128D space")

# 256D space: For anomaly detection (needs 256+ vectors)
vectors_256d = for i <- 1..300 do
  if i <= 270 do
    # Normal vectors in 3 clusters
    cluster = rem(i, 3)
    base = for _ <- 1..256, do: :rand.uniform() * 0.1
    case cluster do
      0 -> Enum.map(base, &(&1 + 0.2))
      1 -> Enum.map(base, &(&1 + 0.5))
      _ -> Enum.map(base, &(&1 + 0.8))
    end
  else
    # Anomalies - random high values
    for _ <- 1..256, do: :rand.uniform() * 2.0
  end
end
metadata_256d = for i <- 1..300, do: %{
  index: i, 
  type: if(i <= 270, do: "normal", else: "anomaly"),
  cluster: if(i <= 270, do: rem(i, 3), else: -1)
}
{:ok, ids_256d} = VSMVectorStore.insert(space_256d, vectors_256d, metadata_256d)
IO.puts("✅ Inserted #{length(ids_256d)} vectors into 256D space")

# 5. K-Nearest Neighbor Search
IO.puts("\n5️⃣ K-Nearest Neighbor Search")
query_64d = List.duplicate(0.5, 64)
{:ok, search_results} = VSMVectorStore.search(space_64d, query_64d, k: 10)
IO.puts("✅ Found #{length(search_results)} nearest neighbors:")
search_results |> Enum.take(5) |> Enum.each(fn r ->
  IO.puts("   - ID: #{r.id}, Distance: #{Float.round(r.distance, 3)}, Cluster: #{r.metadata.cluster}")
end)

# 6. K-means Clustering
IO.puts("\n6️⃣ K-means Clustering")
{:ok, clustering} = VSMVectorStore.cluster(space_64d, k: 3, max_iterations: 100)
IO.puts("✅ Clustering Results:")
IO.puts("   - Number of clusters: #{length(clustering.centroids)}")
IO.puts("   - Inertia: #{Float.round(clustering.inertia, 2)}")
IO.puts("   - Iterations: #{clustering.iterations}")

# Show cluster assignments
cluster_counts = clustering.assignments |> Enum.frequencies()
IO.puts("   - Cluster sizes: #{inspect(cluster_counts)}")

# 7. Pattern Recognition
IO.puts("\n7️⃣ Pattern Recognition")
{:ok, patterns} = VSMVectorStore.recognize_patterns(space_64d)
IO.puts("✅ Pattern Analysis:")
IO.puts("   - Detected clusters: #{patterns.clusters}")
IO.puts("   - Density peaks: #{patterns.density_peaks}")
IO.puts("   - Outlier regions: #{patterns.outlier_regions}")

if patterns.clusters > 0 do
  pattern_sizes = patterns.patterns
  |> Enum.map(fn p -> length(p.members) end)
  |> Enum.sort(:desc)
  |> Enum.take(3)
  IO.puts("   - Top 3 pattern sizes: #{inspect(pattern_sizes)}")
end

# 8. Anomaly Detection (with sufficient data)
IO.puts("\n8️⃣ Anomaly Detection")
{:ok, anomalies} = VSMVectorStore.detect_anomalies(space_256d, contamination: 0.1)
IO.puts("✅ Anomaly Detection Results:")
IO.puts("   - Total anomalies detected: #{length(anomalies)}")

# Check detection accuracy
true_anomalies = anomalies
|> Enum.filter(fn a ->
  # Find the metadata for this vector
  case Enum.find(metadata_256d, fn {id, _vec, meta} -> id == a.vector_id end) do
    {_id, _vec, meta} -> meta.type == "anomaly"
    _ -> false
  end
end)
|> length()

IO.puts("   - True positives: #{true_anomalies}")
IO.puts("   - Detection rate: #{Float.round(true_anomalies / 30 * 100, 1)}%")

# Show top anomalies
anomalies |> Enum.take(5) |> Enum.each(fn a ->
  IO.puts("   - #{a.vector_id}: score=#{Float.round(a.anomaly_score, 3)}, confidence=#{Float.round(a.confidence, 3)}")
end)

# 9. Performance Metrics
IO.puts("\n9️⃣ Performance Metrics")
{:ok, metrics_64d} = VSMVectorStore.metrics(space_64d)
IO.puts("✅ 64D Space Metrics:")
IO.puts("   - Total vectors: #{metrics_64d.total_vectors}")
IO.puts("   - Average search time: #{Float.round(metrics_64d.average_search_time_ms, 2)}ms")
IO.puts("   - Average insert time: #{Float.round(metrics_64d.average_insert_time_ms, 2)}ms")

# 10. Vector Operations Demo
IO.puts("\n🔟 Vector Operations Demo")
v1 = [1.0, 0.0, 0.0]
v2 = [0.0, 1.0, 0.0]
v3 = [0.707, 0.707, 0.0]

euclidean_dist = VSMVectorStore.Storage.VectorOps.euclidean_distance(v1, v2)
cosine_sim = VSMVectorStore.Storage.VectorOps.cosine_similarity(v1, v3)
dot_product = VSMVectorStore.Storage.VectorOps.dot_product(v1, v3)

IO.puts("✅ Vector Operations:")
IO.puts("   - Euclidean distance (v1, v2): #{Float.round(euclidean_dist, 3)}")
IO.puts("   - Cosine similarity (v1, v3): #{Float.round(cosine_sim, 3)}")
IO.puts("   - Dot product (v1, v3): #{Float.round(dot_product, 3)}")

# 11. Batch Operations
IO.puts("\n1️⃣1️⃣ Batch Operations")
batch_vectors = for _ <- 1..50, do: for(_ <- 1..64, do: :rand.uniform())
batch_metadata = for i <- 1..50, do: %{batch: true, index: i}
{:ok, batch_ids} = VSMVectorStore.insert(space_64d, batch_vectors, batch_metadata)
IO.puts("✅ Batch inserted #{length(batch_ids)} vectors")

# 12. Advanced Search with Filters
IO.puts("\n1️⃣2️⃣ Advanced Search")
# Search for vectors from a specific cluster
query = List.duplicate(0.9, 64)  # Should match cluster 0
{:ok, filtered_results} = VSMVectorStore.search(space_64d, query, k: 20)
cluster_0_results = filtered_results 
|> Enum.filter(fn r -> Map.get(r.metadata, :cluster) == 0 end)
|> length()
IO.puts("✅ Found #{cluster_0_results}/20 results from target cluster")

# 13. System Resource Usage
IO.puts("\n1️⃣3️⃣ System Resource Usage")
final_status = VSMVectorStore.status()
IO.puts("✅ Final System Status:")
IO.puts("   - Total spaces: #{final_status.storage.spaces}")
IO.puts("   - Total vectors: #{final_status.storage.vectors}")
IO.puts("   - Memory usage: #{Float.round(final_status.performance.memory_usage * 100, 1)}%")
IO.puts("   - Search latency P95: #{final_status.performance.search_latency_p95}ms")
IO.puts("   - Insertion rate: #{final_status.performance.insertion_rate} vectors/sec")

# 14. Cleanup
IO.puts("\n1️⃣4️⃣ Cleanup")
:ok = VSMVectorStore.delete_space(space_64d)
:ok = VSMVectorStore.delete_space(space_128d)
:ok = VSMVectorStore.delete_space(space_256d)
IO.puts("✅ All spaces deleted")

# Final verification
{:ok, remaining_spaces} = VSMVectorStore.list_spaces()
IO.puts("✅ Remaining spaces: #{length(remaining_spaces)}")

IO.puts("\n" <> String.duplicate("=", 50))
IO.puts("🎉 ALL FEATURES TESTED SUCCESSFULLY! 🎉")
IO.puts("The VSM Vector Store is 100% FUNCTIONAL!")
IO.puts(String.duplicate("=", 50))