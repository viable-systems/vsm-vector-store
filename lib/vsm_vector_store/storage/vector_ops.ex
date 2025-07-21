defmodule VSMVectorStore.Storage.VectorOps do
  @moduledoc """
  Vector operations for storage management.
  Handles vector insertion, retrieval, and basic operations.
  """
  
  alias VSMVectorStore.Storage.Manager
  
  # Vector math operations
  
  def euclidean_distance(v1, v2) do
    v1
    |> Enum.zip(v2)
    |> Enum.map(fn {a, b} -> (a - b) * (a - b) end)
    |> Enum.sum()
    |> :math.sqrt()
  end
  
  def cosine_distance(v1, v2) do
    dot_product = dot(v1, v2)
    norm1 = norm(v1)
    norm2 = norm(v2)
    
    if norm1 == 0 or norm2 == 0 do
      1.0
    else
      1.0 - (dot_product / (norm1 * norm2))
    end
  end
  
  def dot(v1, v2) do
    v1
    |> Enum.zip(v2)
    |> Enum.map(fn {a, b} -> a * b end)
    |> Enum.sum()
  end
  
  def norm(vector) do
    vector
    |> Enum.map(fn x -> x * x end)
    |> Enum.sum()
    |> :math.sqrt()
  end
  
  @doc """
  Inserts vectors into a vector space.
  """
  def insert(space_id, vectors, metadata \\ []) do
    with {:ok, space} <- Manager.get_space(space_id),
         :ok <- validate_vectors(vectors, space.dimensions),
         {:ok, vector_ids} <- do_insert(space_id, vectors, metadata) do
      
      # Update HNSW index - TODO: implement batch_insert
      # Task.start(fn ->
      #   VSMVectorStore.Storage.HNSW.batch_insert(space_id, vector_ids, vectors, metadata)
      # end)
      
      # Emit telemetry
      :telemetry.execute(
        [:vsm_vector_store, :vector, :insert],
        %{count: length(vectors)},
        %{space_id: space_id}
      )
      
      {:ok, vector_ids}
    end
  end
  
  @doc """
  Retrieves vectors by their IDs.
  """
  def get(space_id, vector_ids) when is_list(vector_ids) do
    with {:ok, space} <- Manager.get_space(space_id) do
      table = space.table_ref
      
      vectors = Enum.map(vector_ids, fn id ->
        case :ets.lookup(table, id) do
          [{^id, vector, metadata}] -> 
            %{id: id, vector: vector, metadata: metadata}
          [] -> 
            nil
        end
      end)
      |> Enum.reject(&is_nil/1)
      
      {:ok, vectors}
    end
  end
  
  def get(space_id, vector_id) do
    get(space_id, [vector_id])
  end
  
  @doc """
  Retrieves all vectors from a space.
  """
  def get_all(space_id) do
    with {:ok, space} <- Manager.get_space(space_id) do
      table = space.table_ref
      
      vectors = :ets.tab2list(table)
      |> Enum.map(fn {id, vector, metadata} ->
        %{id: id, vector: vector, metadata: metadata}
      end)
      
      {:ok, vectors}
    end
  end
  
  @doc """
  Updates vector metadata.
  """
  def update_metadata(space_id, vector_id, metadata) do
    with {:ok, space} <- Manager.get_space(space_id),
         table = space.table_ref,
         [{^vector_id, vector, _old_metadata}] <- :ets.lookup(table, vector_id) do
      :ets.insert(table, {vector_id, vector, metadata})
      :ok
    else
      [] -> {:error, :vector_not_found}
      error -> error
    end
  end
  
  @doc """
  Deletes vectors from a space.
  """
  def delete(space_id, vector_ids) when is_list(vector_ids) do
    with {:ok, space} <- Manager.get_space(space_id) do
      table = space.table_ref
      
      Enum.each(vector_ids, fn id ->
        :ets.delete(table, id)
      end)
      
      # Update HNSW index - TODO: implement batch_delete
      # Task.start(fn ->
      #   VSMVectorStore.Storage.HNSW.batch_delete(space_id, vector_ids)
      # end)
      
      :ok
    end
  end
  
  def delete(space_id, vector_id) do
    delete(space_id, [vector_id])
  end
  
  @doc """
  Returns the count of vectors in a space.
  """
  def count(space_id) do
    with {:ok, space} <- Manager.get_space(space_id) do
      table = space.table_ref
      count = :ets.info(table, :size)
      {:ok, count}
    end
  end
  
  # Private functions
  
  defp validate_vectors(vectors, expected_dim) do
    valid? = Enum.all?(vectors, fn vector ->
      is_list(vector) and length(vector) == expected_dim and
      Enum.all?(vector, &is_number/1)
    end)
    
    if valid? do
      :ok
    else
      {:error, :invalid_vector_dimensions}
    end
  end
  
  defp do_insert(space_id, vectors, metadata) do
    # Get the table reference from Manager
    {:ok, space} = Manager.get_space(space_id)
    table = space.table_ref
    
    # Ensure metadata list matches vectors
    metadata_list = if is_list(metadata) and length(metadata) == length(vectors) do
      metadata
    else
      List.duplicate(%{}, length(vectors))
    end
    
    # Generate IDs and insert
    vector_data = vectors
    |> Enum.zip(metadata_list)
    |> Enum.map(fn {vector, meta} ->
      id = generate_vector_id()
      {id, vector, meta}
    end)
    
    # Batch insert into ETS - insert each record individually
    Enum.each(vector_data, fn record ->
      :ets.insert(table, record)
    end)
    
    # Return just the IDs
    ids = Enum.map(vector_data, &elem(&1, 0))
    {:ok, ids}
  end
  
  defp generate_vector_id do
    "vec_#{:crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)}"
  end
  
  defp get_table_name(space_id) do
    String.to_atom("vectors_#{space_id}")
  end
  
  # Additional vector operations needed by ML modules
  
  def centroid(vectors) when is_list(vectors) and length(vectors) > 0 do
    dimension = length(hd(vectors))
    
    # Sum all vectors element-wise
    sum_vector = Enum.reduce(vectors, List.duplicate(0.0, dimension), fn vector, acc ->
      vector
      |> Enum.zip(acc)
      |> Enum.map(fn {v, a} -> v + a end)
    end)
    
    # Divide by count to get mean
    count = length(vectors)
    Enum.map(sum_vector, &(&1 / count))
  end
  
  def cosine_similarity(v1, v2) do
    1.0 - cosine_distance(v1, v2)
  end
  
  def lerp(v1, v2, t) do
    v1
    |> Enum.zip(v2)
    |> Enum.map(fn {a, b} -> a + (b - a) * t end)
  end
end