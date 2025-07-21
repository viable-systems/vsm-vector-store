#!/usr/bin/env elixir

# Minimal test showing core functionality

IO.puts """
🚀 VSM Vector Store - Minimal Test
===================================
"""

# Start the application
{:ok, _} = Application.ensure_all_started(:vsm_vector_store)

# 1. Create Space
IO.puts("1️⃣ Creating vector space...")
{:ok, space_id} = VSMVectorStore.create_space("test", 3)
IO.puts("✅ Created space: #{space_id}")

# 2. Insert Vectors
IO.puts("\n2️⃣ Inserting vectors...")
vectors = [
  [1.0, 0.0, 0.0],
  [0.0, 1.0, 0.0],
  [0.0, 0.0, 1.0],
  [1.0, 1.0, 0.0],
  [0.0, 1.0, 1.0]
]
metadata = Enum.map(1..5, fn i -> %{id: i} end)
{:ok, ids} = VSMVectorStore.insert(space_id, vectors, metadata)
IO.puts("✅ Inserted #{length(ids)} vectors")

# 3. Search
IO.puts("\n3️⃣ Searching...")
query = [0.5, 0.5, 0.0]
{:ok, results} = VSMVectorStore.search(space_id, query, k: 3)
IO.puts("✅ Found #{length(results)} similar vectors:")
Enum.each(results, fn r ->
  IO.puts("   #{r.id} - distance: #{Float.round(r.distance, 3)}")
end)

# 4. Status
IO.puts("\n4️⃣ System Status")
status = VSMVectorStore.status()
IO.puts("✅ Spaces: #{status.storage.spaces}")
IO.puts("✅ Vectors: #{status.storage.vectors}")

IO.puts("\n🎉 Test completed successfully!")