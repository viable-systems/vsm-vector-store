#!/usr/bin/env elixir

# Basic usage example for VSM Vector Store
# Run with: elixir examples/basic_usage.exs

Mix.install([
  {:vsm_vector_store, path: "."}
])

# Start the application
{:ok, _} = Application.ensure_all_started(:vsm_vector_store)

IO.puts("=== VSM Vector Store Basic Usage Example ===\n")

# 1. Create a vector space
IO.puts("1. Creating vector space...")
{:ok, space_id} = VSMVectorStore.create_space("example_space", 64)
IO.puts("   Created space: #{space_id}")

# 2. Insert some vectors
IO.puts("\n2. Inserting vectors...")
vectors = [
  {"user_1", Enum.map(1..64, fn _ -> :rand.uniform() end), %{name: "Alice", category: "A"}},
  {"user_2", Enum.map(1..64, fn _ -> :rand.uniform() end), %{name: "Bob", category: "B"}},
  {"user_3", Enum.map(1..64, fn _ -> :rand.uniform() end), %{name: "Charlie", category: "A"}},
  {"user_4", Enum.map(1..64, fn _ -> :rand.uniform() end), %{name: "David", category: "C"}},
  {"user_5", Enum.map(1..64, fn _ -> :rand.uniform() end), %{name: "Eve", category: "B"}}
]

for {id, vector, metadata} <- vectors do
  :ok = VSMVectorStore.insert(space_id, id, vector, metadata)
  IO.puts("   Inserted #{id} (#{metadata.name})")
end

# 3. Search for similar vectors
IO.puts("\n3. Searching for similar vectors...")
query_vector = Enum.map(1..64, fn _ -> :rand.uniform() end)
{:ok, results} = VSMVectorStore.search(space_id, query_vector, 3)

IO.puts("   Top 3 most similar vectors:")
for {id, distance, metadata} <- results do
  IO.puts("   - #{id} (#{metadata.name}): distance = #{Float.round(distance, 4)}")
end

# 4. Get specific vector
IO.puts("\n4. Retrieving specific vector...")
{:ok, {vector, metadata}} = VSMVectorStore.get(space_id, "user_1")
IO.puts("   Retrieved user_1: #{metadata.name}, category: #{metadata.category}")
IO.puts("   Vector dimensions: #{length(vector)}")

# 5. Update metadata
IO.puts("\n5. Updating metadata...")
:ok = VSMVectorStore.update_metadata(space_id, "user_1", %{name: "Alice", category: "A", updated: true})
{:ok, {_, updated_metadata}} = VSMVectorStore.get(space_id, "user_1")
IO.puts("   Updated metadata: #{inspect(updated_metadata)}")

# 6. Get space statistics
IO.puts("\n6. Space statistics...")
{:ok, stats} = VSMVectorStore.stats(space_id)
IO.puts("   Vectors: #{stats.vector_count}")
IO.puts("   Dimensions: #{stats.dimension}")
if stats[:mean] do
  IO.puts("   Mean values (first 5 dims): #{stats.mean |> Enum.take(5) |> Enum.map(&Float.round(&1, 4)) |> inspect()}")
end

# 7. List all spaces
IO.puts("\n7. All vector spaces...")
{:ok, spaces} = VSMVectorStore.list_spaces()
for space <- spaces do
  IO.puts("   - #{space.name} (#{space.id}): #{space.vector_count} vectors")
end

# 8. Clean up
IO.puts("\n8. Cleaning up...")
:ok = VSMVectorStore.delete_space(space_id)
IO.puts("   Space deleted")

IO.puts("\n=== Example completed successfully! ===")