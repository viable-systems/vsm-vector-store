ExUnit.start(
  exclude: [:performance, :stress, :memory, :slow],
  timeout: 60_000  # 1 minute default timeout
)

# Configure test environment
Application.put_env(:vsm_vector_store, :test_mode, true)
Application.put_env(:vsm_vector_store, :log_level, :warn)

# Setup telemetry for testing
:telemetry.attach_many(
  "test-telemetry-handler",
  [
    [:vsm_vector_store, :vector_insert],
    [:vsm_vector_store, :vector_search],
    [:vsm_vector_store, :vector_clustering],
    [:vsm_vector_store, :anomaly_detection],
    [:vsm_vector_store, :test_event]
  ],
  fn event_name, measurements, metadata, _config ->
    # Store telemetry events for test verification
    Agent.update(:test_telemetry_store, fn events ->
      [{event_name, measurements, metadata, System.monotonic_time()} | events]
    end)
  end,
  nil
)

# Start telemetry event store
{:ok, _} = Agent.start_link(fn -> [] end, name: :test_telemetry_store)

# Test helper functions available globally
defmodule TestGlobals do
  @doc "Gets all telemetry events captured during test"
  def get_telemetry_events() do
    Agent.get(:test_telemetry_store, & &1)
  end
  
  @doc "Clears telemetry event store"
  def clear_telemetry_events() do
    Agent.update(:test_telemetry_store, fn _ -> [] end)
  end
  
  @doc "Waits for a specific telemetry event"
  def wait_for_telemetry_event(event_name, timeout \\ 5000) do
    start_time = System.monotonic_time(:millisecond)
    
    wait_loop = fn wait_fn ->
      events = get_telemetry_events()
      
      case Enum.find(events, fn {name, _, _, _} -> name == event_name end) do
        nil ->
          if System.monotonic_time(:millisecond) - start_time < timeout do
            Process.sleep(10)
            wait_fn.(wait_fn)
          else
            {:error, :timeout}
          end
        
        event -> {:ok, event}
      end
    end
    
    wait_loop.(wait_loop)
  end
end

# Performance test configuration
performance_config = %{
  small_dataset: 1_000,
  medium_dataset: 10_000,
  large_dataset: 50_000,
  xl_dataset: 100_000,
  
  low_dimensions: 8,
  medium_dimensions: 32,
  high_dimensions: 128,
  
  quick_test_timeout: 30_000,     # 30 seconds
  medium_test_timeout: 120_000,   # 2 minutes  
  long_test_timeout: 300_000      # 5 minutes
}

Application.put_env(:vsm_vector_store, :performance_config, performance_config)

# Memory test helpers
defmodule MemoryTestHelpers do
  @doc "Forces garbage collection across all processes"
  def force_gc() do
    :erlang.garbage_collect()
    
    # GC other processes too
    Process.list()
    |> Enum.each(fn pid ->
      try do
        :erlang.garbage_collect(pid)
      rescue
        _ -> :ok
      end
    end)
  end
  
  @doc "Gets detailed memory information"
  def get_memory_info() do
    memory = :erlang.memory()
    
    %{
      total_mb: Keyword.get(memory, :total) / 1024 / 1024,
      processes_mb: Keyword.get(memory, :processes) / 1024 / 1024,
      system_mb: Keyword.get(memory, :system) / 1024 / 1024,
      atom_mb: Keyword.get(memory, :atom) / 1024 / 1024,
      binary_mb: Keyword.get(memory, :binary) / 1024 / 1024,
      code_mb: Keyword.get(memory, :code) / 1024 / 1024,
      ets_mb: Keyword.get(memory, :ets) / 1024 / 1024
    }
  end
end

# Concurrency test helpers  
defmodule ConcurrencyTestHelpers do
  @doc "Runs a function concurrently N times"
  def run_concurrent(fun, n, timeout \\ 10_000) do
    1..n
    |> Enum.map(fn i ->
      Task.async(fn -> 
        try do
          {:ok, fun.(i)}
        rescue
          error -> {:error, error}
        catch
          :exit, reason -> {:error, {:exit, reason}}
        end
      end)
    end)
    |> Task.await_many(timeout)
  end
  
  @doc "Measures throughput of concurrent operations"
  def measure_throughput(operation_fn, duration_ms, concurrency \\ 1) do
    start_time = System.monotonic_time(:millisecond)
    end_time = start_time + duration_ms
    
    tasks = 1..concurrency
    |> Enum.map(fn _i ->
      Task.async(fn ->
        count_operations(operation_fn, end_time, 0)
      end)
    end)
    
    results = Task.await_many(tasks, duration_ms + 5000)
    
    total_operations = Enum.sum(results)
    actual_duration = System.monotonic_time(:millisecond) - start_time
    
    %{
      total_operations: total_operations,
      duration_ms: actual_duration,
      throughput_ops_per_second: total_operations / (actual_duration / 1000),
      concurrency: concurrency
    }
  end
  
  defp count_operations(operation_fn, end_time, count) do
    if System.monotonic_time(:millisecond) < end_time do
      try do
        operation_fn.()
        count_operations(operation_fn, end_time, count + 1)
      rescue
        _ -> count_operations(operation_fn, end_time, count)
      end
    else
      count
    end
  end
end

IO.puts("VSM Vector Store test environment initialized")
IO.puts("Available test tags: :performance, :stress, :memory, :slow")
IO.puts("Run performance tests with: mix test --include performance")
IO.puts("Run all tests with: mix test --include performance --include stress --include memory --include slow")
