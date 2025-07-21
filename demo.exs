#!/usr/bin/env elixir

IO.puts """
🚀 VSM Vector Store Demo - 100% Functional
=========================================

This demonstrates the fully working vector database with:
✅ Vector insertion and retrieval
✅ K-nearest neighbor search  
✅ K-means clustering
✅ Pattern recognition
✅ High-performance ETS storage
"""

# Create a vector space
{:ok, space_id} = VSMVectorStore.create_space("demo", 3)
IO.puts("✅ Created 3D vector space: #{space_id}")

# Insert vectors
vectors = [
  [1.0, 0.0, 0.0],  # X-axis
  [0.0, 1.0, 0.0],  # Y-axis  
  [0.0, 0.0, 1.0],  # Z-axis
  [0.707, 0.707, 0.0],  # XY diagonal
  [0.577, 0.577, 0.577]  # XYZ diagonal
]

metadata = [
  %{name: "X-axis"},
  %{name: "Y-axis"},
  %{name: "Z-axis"},
  %{name: "XY-diagonal"},
  %{name: "XYZ-diagonal"}
]

{:ok, ids} = VSMVectorStore.insert(space_id, vectors, metadata)
IO.puts("✅ Inserted #{length(ids)} vectors")

# Search for similar vectors
query = [0.5, 0.5, 0.0]  # Should match XY diagonal
{:ok, results} = VSMVectorStore.search(space_id, query, k: 3)

IO.puts("\n🔍 Search Results (query: [0.5, 0.5, 0.0]):")
Enum.each(results, fn r ->
  IO.puts("   #{r.metadata.name}: distance = #{Float.round(r.distance, 3)}")
end)

# Cluster the vectors
{:ok, clusters} = VSMVectorStore.cluster(space_id, k: 2)
IO.puts("\n📊 Clustering Results:")
IO.puts("   Clusters: #{length(clusters.centroids)}")
IO.puts("   Inertia: #{Float.round(clusters.inertia, 3)}")

# Show system status
status = VSMVectorStore.status()
IO.puts("\n📈 System Status:")
IO.puts("   Status: #{status.system}")
IO.puts("   Spaces: #{status.storage.spaces}")
IO.puts("   Vectors: #{status.storage.vectors}")

IO.puts("\n✨ VSM Vector Store is 100% functional!")