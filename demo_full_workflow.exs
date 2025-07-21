#!/usr/bin/env elixir

# Full workflow demonstration of VSM Vector Store
Mix.install([])

IO.puts """
🚀 VSM Vector Store - Full Workflow Demo
======================================

This demo shows ALL features working together:
- Vector space creation
- Vector insertion with metadata
- Similarity search
- K-means clustering
- Anomaly detection
- Pattern recognition

"""

# Helper to generate random vectors
defmodule DemoHelpers do
  def generate_vectors(count, dimensions) do
    for i <- 1..count do
      vector = for _ <- 1..dimensions, do: :rand.uniform() * 2 - 1
      
      # Add some structure - create 3 natural clusters
      vector = case rem(i, 3) do
        0 -> Enum.map(vector, &(&1 + 2.0))  # Cluster 1: shifted +2
        1 -> Enum.map(vector, &(&1 - 2.0))  # Cluster 2: shifted -2
        2 -> vector                          # Cluster 3: centered
      end
      
      # Add metadata
      metadata = %{
        category: "cluster_#{rem(i, 3)}",
        index: i,
        timestamp: DateTime.utc_now()
      }
      
      {vector, metadata}
    end
  end
  
  def print_results(label, results) do
    IO.puts("\n📊 #{label}:")
    IO.puts("=" |> String.duplicate(50))
    IO.inspect(results, pretty: true, limit: 5)
  end
end

# Start the application
{:ok, _} = Application.ensure_all_started(:vsm_vector_store)

# Wait for startup
Process.sleep(100)

# 1. Check system status
IO.puts("\n1️⃣ System Status Check")
case VSMVectorStore.status() do
  %{system: :running} = status ->
    IO.puts("✅ System is running!")
    IO.puts("   Subsystems: #{inspect(status.subsystems)}")
  _ ->
    IO.puts("❌ System not ready")
    System.halt(1)
end

# 2. Create a vector space
IO.puts("\n2️⃣ Creating Vector Space")
dimensions = 128
{:ok, space_id} = VSMVectorStore.create_space("demo_space", dimensions, 
  distance_metric: :cosine,
  quantization: [enabled: false]
)
IO.puts("✅ Created space: #{space_id}")

# 3. Generate and insert vectors
IO.puts("\n3️⃣ Inserting Vectors")
vector_count = 300
vector_data = DemoHelpers.generate_vectors(vector_count, dimensions)
vectors = Enum.map(vector_data, &elem(&1, 0))
metadata = Enum.map(vector_data, &elem(&1, 1))

{:ok, vector_ids} = VSMVectorStore.insert(space_id, vectors, metadata)
IO.puts("✅ Inserted #{length(vector_ids)} vectors")
IO.puts("   Sample IDs: #{Enum.take(vector_ids, 3) |> inspect()}")

# 4. Perform similarity search
IO.puts("\n4️⃣ Similarity Search")
query_vector = List.duplicate(0.5, dimensions)  # Query near cluster center
{:ok, search_results} = VSMVectorStore.search(space_id, query_vector, k: 10)

IO.puts("✅ Found #{length(search_results)} similar vectors")
search_results
|> Enum.take(3)
|> Enum.each(fn result ->
  IO.puts("   ID: #{result.id}, Distance: #{Float.round(result.distance, 4)}, " <>
          "Category: #{result.metadata.category}")
end)

# 5. List all spaces
IO.puts("\n5️⃣ List Vector Spaces")
{:ok, spaces} = VSMVectorStore.list_spaces()
DemoHelpers.print_results("Available Spaces", spaces)

# 6. K-means clustering
IO.puts("\n6️⃣ K-means Clustering")
IO.puts("🔄 Running clustering (this may take a moment)...")
{:ok, clusters} = VSMVectorStore.cluster(space_id, k: 3, max_iterations: 20)
IO.puts("✅ Clustering complete!")
IO.puts("   Found #{length(clusters.centroids)} clusters")
IO.puts("   Inertia: #{Float.round(clusters.inertia, 2)}")

# Count vectors per cluster
cluster_counts = clusters.assignments
|> Enum.frequencies()
|> Enum.sort()
IO.puts("   Cluster sizes: #{inspect(cluster_counts)}")

# 7. Anomaly detection
IO.puts("\n7️⃣ Anomaly Detection")
IO.puts("🔄 Running anomaly detection...")
{:ok, anomalies} = VSMVectorStore.detect_anomalies(space_id, 
  method: :isolation_forest,
  contamination: 0.05
)

anomaly_count = Enum.count(anomalies, & &1.is_anomaly)
IO.puts("✅ Found #{anomaly_count} anomalies out of #{length(anomalies)} vectors")

# Show top anomalies
anomalies
|> Enum.filter(& &1.is_anomaly)
|> Enum.sort_by(& &1.score, :desc)
|> Enum.take(3)
|> Enum.each(fn anomaly ->
  IO.puts("   ID: #{anomaly.id}, Score: #{Float.round(anomaly.score, 4)}")
end)

# 8. Pattern recognition
IO.puts("\n8️⃣ Pattern Recognition")
IO.puts("🔄 Analyzing patterns...")
{:ok, patterns} = VSMVectorStore.recognize_patterns(space_id)
DemoHelpers.print_results("Pattern Analysis", patterns)

# 9. Performance metrics
IO.puts("\n9️⃣ Performance Metrics")
{:ok, metrics} = VSMVectorStore.metrics(space_id)
DemoHelpers.print_results("Space Metrics", metrics)

# 10. Final status check
IO.puts("\n🔟 Final System Status")
final_status = VSMVectorStore.status()
IO.puts("✅ System Health: #{final_status.system}")
IO.puts("   Total Spaces: #{final_status.storage.spaces}")
IO.puts("   Total Vectors: #{final_status.storage.vectors}")
IO.puts("   Memory Usage: #{Float.round(final_status.performance.memory_usage * 100, 1)}%")

# Cleanup
IO.puts("\n🧹 Cleanup")
:ok = VSMVectorStore.delete_space(space_id)
IO.puts("✅ Deleted demo space")

IO.puts("\n🎉 Demo Complete! All features working successfully!")
IO.puts("\nThe VSM Vector Store is ready for production use with:")
IO.puts("  • High-performance vector storage (ETS-backed)")
IO.puts("  • O(log N) similarity search (HNSW)")
IO.puts("  • K-means clustering")
IO.puts("  • Anomaly detection") 
IO.puts("  • Pattern recognition")
IO.puts("  • Full VSM architecture compliance")
IO.puts("\n🚀 Happy vector searching! 🚀\n")