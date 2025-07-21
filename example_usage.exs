#!/usr/bin/env elixir

# Example usage of VSM Vector Store
# Run with: elixir example_usage.exs

# Start the application
{:ok, _} = Application.ensure_all_started(:vsm_vector_store)

IO.puts("🚀 Starting VSM Vector Store Example")

# Wait for application to fully start
Process.sleep(2000)

# Check system status
IO.puts("\n📊 System Status:")
status = VsmVectorStore.status()
IO.inspect(status)

# Insert some test vectors
IO.puts("\n📝 Inserting test vectors...")

vectors = [
  {"doc1", [0.1, 0.2, 0.3, 0.4], %{type: "document", category: "text"}},
  {"doc2", [0.15, 0.25, 0.35, 0.45], %{type: "document", category: "text"}},
  {"doc3", [0.9, 0.8, 0.7, 0.6], %{type: "document", category: "image"}},
  {"doc4", [0.05, 0.15, 0.25, 0.35], %{type: "document", category: "text"}},
  {"doc5", [0.85, 0.75, 0.65, 0.55], %{type: "document", category: "image"}}
]

Enum.each(vectors, fn {id, vector, metadata} ->
  case VsmVectorStore.insert(id, vector, metadata) do
    :ok -> IO.puts("  ✅ Inserted #{id}")
    {:error, reason} -> IO.puts("  ❌ Failed to insert #{id}: #{inspect(reason)}")
  end
end)

# Wait for processing
Process.sleep(1000)

# Search for similar vectors
IO.puts("\n🔍 Searching for similar vectors to doc1...")
query_vector = [0.12, 0.22, 0.32, 0.42]

case VsmVectorStore.search(query_vector, 3) do
  {:ok, results} ->
    IO.puts("  Found #{length(results)} similar vectors:")
    Enum.each(results, fn {id, distance, metadata} ->
      IO.puts("    • #{id} (distance: #{Float.round(distance, 4)}, type: #{metadata.category})")
    end)
  {:error, reason} ->
    IO.puts("  ❌ Search failed: #{inspect(reason)}")
end

# Retrieve a specific vector
IO.puts("\n📖 Retrieving doc3...")
case VsmVectorStore.get("doc3") do
  {:ok, {vector, metadata}} ->
    IO.puts("  ✅ Retrieved: #{inspect(vector)}")
    IO.puts("     Metadata: #{inspect(metadata)}")
  {:error, reason} ->
    IO.puts("  ❌ Failed: #{inspect(reason)}")
end

# Perform clustering
IO.puts("\n🎯 Performing K-means clustering (k=2)...")
case VsmVectorStore.cluster(2) do
  {:ok, clusters} ->
    IO.puts("  ✅ Created #{length(clusters)} clusters:")
    Enum.each(clusters, fn cluster ->
      IO.puts("    Cluster #{cluster.id}: #{length(cluster.members)} members - #{inspect(cluster.members)}")
    end)
  {:error, reason} ->
    IO.puts("  ❌ Clustering failed: #{inspect(reason)}")
end

# Detect anomalies
IO.puts("\n🚨 Detecting anomalies...")
case VsmVectorStore.detect_anomalies(0.2) do
  {:ok, anomalies} ->
    if Enum.empty?(anomalies) do
      IO.puts("  ✅ No anomalies detected")
    else
      IO.puts("  🚨 Found #{length(anomalies)} anomalies:")
      Enum.each(anomalies, fn anomaly ->
        IO.puts("    • #{anomaly.vector_id} (score: #{Float.round(anomaly.anomaly_score, 3)})")
      end)
    end
  {:error, reason} ->
    IO.puts("  ❌ Anomaly detection failed: #{inspect(reason)}")
end

# Get system metrics
IO.puts("\n📈 System Metrics:")
case VsmVectorStore.metrics() do
  {:ok, metrics} ->
    IO.puts("  • Health Score: #{Float.round(metrics.health_score, 3)}")
    IO.puts("  • Uptime: #{metrics.uptime_human}")
    IO.puts("  • Total Metrics: #{metrics.metrics_summary.total_metrics}")
  {:error, reason} ->
    IO.puts("  ❌ Failed to get metrics: #{inspect(reason)}")
end

IO.puts("\n✅ VSM Vector Store Example Completed!")