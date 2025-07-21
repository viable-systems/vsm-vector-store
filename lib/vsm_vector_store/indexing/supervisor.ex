defmodule VSMVectorStore.Indexing.Supervisor do
  @moduledoc """
  Indexing subsystem supervisor - manages K-means clustering and quantization.
  """
  
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    children = [
      # K-means clustering service
      VSMVectorStore.Indexing.KMeans,
      
      # Vector quantization service
      VSMVectorStore.Indexing.Quantization
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end