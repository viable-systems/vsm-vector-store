defmodule VSMVectorStore.Indexing.Quantization do
  @moduledoc """
  Product Quantization (PQ) implementation for vector compression and fast similarity search.
  
  Product Quantization divides high-dimensional vectors into subvectors and quantizes
  each subvector independently using k-means clustering. This provides:
  - Significant memory reduction (e.g., 128D float32 -> 8 bytes)
  - Fast approximate distance calculations
  - Configurable compression ratio
  - Support for asymmetric distance computation
  
  Features:
  - Pure Elixir implementation
  - Configurable subvector count and codebook size
  - Training and encoding/decoding operations
  - Distance table computation for fast search
  - VSM patterns for error handling
  """
  
  use GenServer
  require Logger
  
  alias VsmVectorStore.Storage.VectorOps
  alias VsmVectorStore.Indexing.KMeans
  
  @type vector :: list(float())
  @type vector_id :: binary()
  @type code :: list(non_neg_integer())
  @type codebook :: list(list(vector()))
  @type subvector :: list(float())
  
  defstruct [
    dimension: 128,
    subvector_count: 8,
    codebook_size: 256,
    codebooks: [],
    trained: false
  ]
  
  ## Public API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Train the quantizer on a set of vectors.
  
  ## Parameters
  - vectors: List of training vectors
  - opts: Training options
  
  ## Returns
  - :ok on success
  - {:error, reason} on failure
  """
  @spec train(pid(), list(vector()), keyword()) :: :ok | {:error, term()}
  def train(pid \\ __MODULE__, vectors, opts \\ []) do
    GenServer.call(pid, {:train, vectors, opts}, 60_000)
  end
  
  @doc """
  Encode a vector into quantized codes.
  """
  @spec encode(pid(), vector()) :: {:ok, code()} | {:error, term()}
  def encode(pid \\ __MODULE__, vector) do
    GenServer.call(pid, {:encode, vector})
  end
  
  @doc """
  Decode quantized codes back to approximate vector.
  """
  @spec decode(pid(), code()) :: {:ok, vector()} | {:error, term()}
  def decode(pid \\ __MODULE__, codes) do
    GenServer.call(pid, {:decode, codes})
  end
  
  @doc """
  Encode multiple vectors in batch.
  """
  @spec encode_batch(pid(), list(vector())) :: {:ok, list(code())} | {:error, term()}
  def encode_batch(pid \\ __MODULE__, vectors) do
    GenServer.call(pid, {:encode_batch, vectors}, 60_000)
  end
  
  @doc """
  Compute distance table for asymmetric distance computation.
  This precomputes distances between query subvectors and all codebook centroids.
  """
  @spec compute_distance_table(pid(), vector()) :: {:ok, list(list(float()))} | {:error, term()}
  def compute_distance_table(pid \\ __MODULE__, query_vector) do
    GenServer.call(pid, {:compute_distance_table, query_vector})
  end
  
  @doc """
  Compute approximate distance using precomputed distance table.
  """
  @spec asymmetric_distance(list(list(float())), code()) :: float()
  def asymmetric_distance(distance_table, codes) do
    distance_table
    |> Enum.zip(codes)
    |> Enum.reduce(0.0, fn {distances, code}, acc ->
      acc + Enum.at(distances, code)
    end)
  end
  
  @doc """
  Get compression ratio achieved by quantization.
  """
  @spec compression_ratio(pid()) :: {:ok, float()} | {:error, term()}
  def compression_ratio(pid \\ __MODULE__) do
    GenServer.call(pid, :compression_ratio)
  end
  
  ## GenServer Callbacks
  
  @impl true
  def init(opts) do
    dimension = Keyword.get(opts, :dimension, 128)
    subvector_count = Keyword.get(opts, :subvector_count, 8)
    codebook_size = Keyword.get(opts, :codebook_size, 256)
    
    # Validate parameters
    if rem(dimension, subvector_count) != 0 do
      {:stop, {:error, :dimension_not_divisible}}
    else
      state = %__MODULE__{
        dimension: dimension,
        subvector_count: subvector_count,
        codebook_size: codebook_size,
        codebooks: [],
        trained: false
      }
      
      Logger.info("Product Quantization initialized: #{dimension}D -> #{subvector_count} subvectors of #{div(dimension, subvector_count)}D each")
      {:ok, state}
    end
  end
  
  @impl true
  def handle_call({:train, vectors, _opts}, _from, state) do
    start_time = System.monotonic_time()
    
    case train_quantizer(vectors, state) do
      {:ok, new_state} ->
        duration = System.monotonic_time() - start_time
        :telemetry.execute([:vsm_vector_store, :indexing, :quantization, :train], 
          %{duration: duration, vectors_count: length(vectors)}, 
          %{subvector_count: state.subvector_count, codebook_size: state.codebook_size})
        {:reply, :ok, new_state}
        
      {:error, reason} ->
        Logger.error("Quantizer training failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end
  
  @impl true
  def handle_call({:encode, vector}, _from, state) do
    if not state.trained do
      {:reply, {:error, :not_trained}, state}
    else
      case encode_vector(vector, state) do
        {:ok, codes} ->
          {:reply, {:ok, codes}, state}
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end
  
  @impl true
  def handle_call({:decode, codes}, _from, state) do
    if not state.trained do
      {:reply, {:error, :not_trained}, state}
    else
      case decode_codes(codes, state) do
        {:ok, vector} ->
          {:reply, {:ok, vector}, state}
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end
  
  @impl true
  def handle_call({:encode_batch, vectors}, _from, state) do
    if not state.trained do
      {:reply, {:error, :not_trained}, state}
    else
      case encode_vectors_batch(vectors, state) do
        {:ok, codes_list} ->
          {:reply, {:ok, codes_list}, state}
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end
  
  @impl true
  def handle_call({:compute_distance_table, query_vector}, _from, state) do
    if not state.trained do
      {:reply, {:error, :not_trained}, state}
    else
      case compute_query_distance_table(query_vector, state) do
        {:ok, distance_table} ->
          {:reply, {:ok, distance_table}, state}
        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    end
  end
  
  @impl true
  def handle_call(:compression_ratio, _from, state) do
    if not state.trained do
      {:reply, {:error, :not_trained}, state}
    else
      # Original: dimension * 4 bytes (float32)
      # Compressed: subvector_count * 1 byte (assuming codebook_size <= 256)
      original_bytes = state.dimension * 4
      compressed_bytes = state.subvector_count * bytes_per_code(state.codebook_size)
      ratio = original_bytes / compressed_bytes
      
      {:reply, {:ok, ratio}, state}
    end
  end
  
  ## Private Functions
  
  defp train_quantizer(vectors, state) do
    if length(vectors) < state.codebook_size do
      {:error, :insufficient_training_data}
    else
      subvector_dim = div(state.dimension, state.subvector_count)
      
      # Split all vectors into subvectors
      subvector_groups = split_vectors_into_subvectors(vectors, state.subvector_count, subvector_dim)
      
      # Train a codebook for each subvector position
      case train_codebooks(subvector_groups, state.codebook_size) do
        {:ok, codebooks} ->
          new_state = %{state | codebooks: codebooks, trained: true}
          {:ok, new_state}
        {:error, reason} ->
          {:error, reason}
      end
    end
  end
  
  defp split_vectors_into_subvectors(vectors, subvector_count, subvector_dim) do
    0..(subvector_count - 1)
    |> Enum.map(fn subvector_idx ->
      start_idx = subvector_idx * subvector_dim
      end_idx = start_idx + subvector_dim - 1
      
      # Extract subvector at this position from all training vectors
      Enum.map(vectors, fn vector ->
        Enum.slice(vector, start_idx..end_idx)
      end)
    end)
  end
  
  defp train_codebooks(subvector_groups, codebook_size) do
    codebooks = Enum.map(subvector_groups, fn subvectors ->
      # Use k-means to create codebook for this subvector position
      case train_single_codebook(subvectors, codebook_size) do
        {:ok, codebook} -> codebook
        {:error, _reason} -> 
          # Fallback to random codebook if k-means fails
          create_random_codebook(subvectors, codebook_size)
      end
    end)
    
    {:ok, codebooks}
  end
  
  defp train_single_codebook(subvectors, codebook_size) do
    # Prepare data for k-means clustering
    vectors_with_ids = subvectors
    |> Enum.with_index()
    |> Enum.map(fn {subvector, idx} -> {to_string(idx), subvector, %{}} end)
    
    case KMeans.cluster(VSMVectorStore.Indexing.KMeans, vectors_with_ids, codebook_size) do
      {:ok, clusters} ->
        # Extract centroids as codebook
        codebook = Enum.map(clusters, fn cluster -> cluster.centroid end)
        {:ok, codebook}
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  defp create_random_codebook(subvectors, codebook_size) do
    case subvectors do
      [] -> []
      [first_subvector | _] ->
        subvector_dim = length(first_subvector)
        
        # Calculate range of values in subvectors
        {min_vals, max_vals} = calculate_subvector_ranges(subvectors, subvector_dim)
        
        # Generate random codebook entries
        1..codebook_size
        |> Enum.map(fn _ ->
          0..(subvector_dim - 1)
          |> Enum.map(fn dim ->
            min_val = Enum.at(min_vals, dim)
            max_val = Enum.at(max_vals, dim)
            min_val + :rand.uniform() * (max_val - min_val)
          end)
        end)
    end
  end
  
  defp calculate_subvector_ranges(subvectors, subvector_dim) do
    0..(subvector_dim - 1)
    |> Enum.map(fn dim ->
      values = Enum.map(subvectors, fn subvector -> Enum.at(subvector, dim) end)
      {Enum.min(values), Enum.max(values)}
    end)
    |> Enum.unzip()
  end
  
  defp encode_vector(vector, state) do
    if length(vector) != state.dimension do
      {:error, :dimension_mismatch}
    else
      subvector_dim = div(state.dimension, state.subvector_count)
      
      codes = 0..(state.subvector_count - 1)
      |> Enum.map(fn subvector_idx ->
        start_idx = subvector_idx * subvector_dim
        end_idx = start_idx + subvector_dim - 1
        
        subvector = Enum.slice(vector, start_idx..end_idx)
        codebook = Enum.at(state.codebooks, subvector_idx)
        
        find_nearest_codeword(subvector, codebook)
      end)
      
      {:ok, codes}
    end
  end
  
  defp find_nearest_codeword(subvector, codebook) do
    codebook
    |> Enum.with_index()
    |> Enum.map(fn {codeword, idx} ->
      distance = VSMVectorStore.Storage.VectorOps.euclidean_distance(subvector, codeword)
      {distance, idx}
    end)
    |> Enum.min_by(fn {distance, _idx} -> distance end)
    |> elem(1)
  end
  
  defp decode_codes(codes, state) do
    if length(codes) != state.subvector_count do
      {:error, :invalid_codes}
    else
      decoded_subvectors = codes
      |> Enum.with_index()
      |> Enum.map(fn {code, subvector_idx} ->
        codebook = Enum.at(state.codebooks, subvector_idx)
        Enum.at(codebook, code)
      end)
      
      # Concatenate subvectors to form full vector
      decoded_vector = List.flatten(decoded_subvectors)
      {:ok, decoded_vector}
    end
  end
  
  defp encode_vectors_batch(vectors, state) do
    results = Enum.map(vectors, fn vector ->
      encode_vector(vector, state)
    end)
    
    # Check if any encoding failed
    case Enum.find(results, fn result -> match?({:error, _}, result) end) do
      nil ->
        codes_list = Enum.map(results, fn {:ok, codes} -> codes end)
        {:ok, codes_list}
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  defp compute_query_distance_table(query_vector, state) do
    if length(query_vector) != state.dimension do
      {:error, :dimension_mismatch}
    else
      subvector_dim = div(state.dimension, state.subvector_count)
      
      distance_table = 0..(state.subvector_count - 1)
      |> Enum.map(fn subvector_idx ->
        start_idx = subvector_idx * subvector_dim
        end_idx = start_idx + subvector_dim - 1
        
        query_subvector = Enum.slice(query_vector, start_idx..end_idx)
        codebook = Enum.at(state.codebooks, subvector_idx)
        
        # Compute distances from query subvector to all codewords in this codebook
        Enum.map(codebook, fn codeword ->
          VectorOps.euclidean_distance(query_subvector, codeword)
        end)
      end)
      
      {:ok, distance_table}
    end
  end
  
  defp bytes_per_code(codebook_size) do
    cond do
      codebook_size <= 256 -> 1
      codebook_size <= 65536 -> 2
      true -> 4
    end
  end
end