defmodule VsmVectorStore.VSMIntegrationTest do
  @moduledoc """
  Integration tests for VSM Vector Store with the VSM ecosystem.
  
  Tests telemetry integration, algedonic channels, event bus coordination,
  and cross-subsystem communication patterns.
  """
  
  use ExUnit.Case, async: false  # VSM integration needs sequential execution
  
  alias VsmVectorStore.TestHelpers
  alias VsmVectorStore.{TelemetryReporter, EventBridge, System1, System2, System3, System4, System5}
  
  setup_all do
    # Start the VSM Vector Store application
    {:ok, _} = Application.ensure_all_started(:vsm_vector_store)
    
    # Setup test telemetry handlers
    :telemetry.attach_many(
      "test-vsm-integration",
      [
        [:vsm_vector_store, :vector_insert],
        [:vsm_vector_store, :vector_search],
        [:vsm_vector_store, :vector_clustering],
        [:vsm_vector_store, :anomaly_detection],
        [:vsm_vector_store, :algedonic_signal]
      ],
      &__MODULE__.handle_telemetry_event/4,
      %{test_pid: self()}
    )
    
    on_exit(fn ->
      :telemetry.detach("test-vsm-integration")
      Application.stop(:vsm_vector_store)
    end)
    
    :ok
  end
  
  describe "VSM Subsystem Integration" do
    test "System 1 (Operations) coordinates with other subsystems" do
      vectors = TestHelpers.generate_random_vectors(100, 8, seed: 6001)
      
      # Insert vectors through System1 operations
      {:ok, vector_ids} = System1.Operations.insert_batch(vectors, %{source: "test"})
      
      assert length(vector_ids) == length(vectors)
      
      # Verify System2 coordination is notified
      Process.sleep(100)  # Allow coordination messages to propagate
      coordination_state = System2.Coordination.get_state()
      assert coordination_state.recent_operations > 0
      
      # Verify System3 resource tracking
      resource_usage = System3.Resources.get_memory_usage()
      assert resource_usage.vector_storage > 0
      
      # System4 should have analytics data
      analytics = System4.Analytics.get_insertion_stats()
      assert analytics.total_insertions >= length(vectors)
    end
    
    test "telemetry integration reports to VSM infrastructure" do
      vectors = TestHelpers.generate_clustered_vectors(50, 4, 3, seed: 6002)
      
      # Perform operations and capture telemetry events
      {:ok, _vector_ids} = System1.Operations.insert_batch(vectors, %{})
      
      query_vector = hd(vectors)
      {:ok, _search_results} = System1.Operations.search_knn(query_vector, 5, %{})
      
      # Wait for telemetry events
      Process.sleep(200)
      
      # Verify telemetry events were emitted
      assert_received {:telemetry_event, [:vsm_vector_store, :vector_insert], _, _}
      assert_received {:telemetry_event, [:vsm_vector_store, :vector_search], _, _}
    end
    
    test "algedonic channels trigger on performance issues" do
      # Create a scenario that should trigger algedonic signals
      large_vectors = TestHelpers.generate_random_vectors(1000, 64, seed: 6003)
      
      # Simulate high load to trigger performance degradation
      tasks = 1..5
      |> Enum.map(fn i ->
        Task.async(fn ->
          query = TestHelpers.generate_random_vectors(1, 64, seed: i * 1000) |> hd()
          System1.Operations.search_knn(query, 50, %{})
        end)
      end)
      
      # Insert data while searches are running (stress scenario)
      {:ok, _vector_ids} = System1.Operations.insert_batch(large_vectors, %{})
      
      Task.await_many(tasks, 30000)
      
      # Wait for algedonic signals
      Process.sleep(500)
      
      # Should have received algedonic signals for performance issues
      algedonic_state = System3.Monitoring.get_algedonic_signals()
      assert length(algedonic_state.recent_signals) > 0
      
      # Verify signal types
      signal_types = Enum.map(algedonic_state.recent_signals, & &1.type)
      assert :performance_degradation in signal_types or :memory_pressure in signal_types
    end
    
    test "System5 policy enforcement works correctly" do
      # Setup restrictive policy
      policy_config = %{
        max_vector_dimensions: 10,
        max_batch_size: 50,
        allowed_operations: [:insert, :search]
      }
      
      System5.Policy.update_configuration(policy_config)
      
      # Test policy enforcement - should succeed
      valid_vectors = TestHelpers.generate_random_vectors(30, 8, seed: 6004)
      {:ok, _vector_ids} = System1.Operations.insert_batch(valid_vectors, %{})
      
      # Test policy violation - dimensions too high
      invalid_vectors = TestHelpers.generate_random_vectors(10, 20, seed: 6005)
      
      assert {:error, :policy_violation} = 
        System1.Operations.insert_batch(invalid_vectors, %{})
      
      # Test policy violation - batch too large
      large_batch = TestHelpers.generate_random_vectors(100, 8, seed: 6006)
      
      assert {:error, :policy_violation} = 
        System1.Operations.insert_batch(large_batch, %{})
    end
  end
  
  describe "Event Bus Integration" do
    test "vector operations publish events to VSM event bus" do
      # Setup event subscriber
      EventBridge.subscribe_to_events(self(), [:vector_events, :ml_events])
      
      vectors = TestHelpers.generate_random_vectors(25, 6, seed: 6007)
      
      # Perform operations that should generate events
      {:ok, vector_ids} = System1.Operations.insert_batch(vectors, %{source: "integration_test"})
      
      # Cluster the vectors
      {:ok, _cluster_result} = System4.ML.cluster_kmeans(3, %{})
      
      # Detect anomalies
      {:ok, _anomalies} = System4.AnomalyDetection.detect(%{method: :isolation_forest})
      
      # Wait for events to propagate
      Process.sleep(300)
      
      # Verify events were received
      assert_received {:vsm_event, :vector_insert, %{vector_ids: ^vector_ids}}
      assert_received {:vsm_event, :ml_clustering, %{method: :kmeans, k: 3}}
      assert_received {:vsm_event, :anomaly_detection, %{method: :isolation_forest}}
    end
    
    test "responds to external VSM system events" do
      # Simulate external system events
      data_ingestion_event = %{
        vectors: TestHelpers.generate_random_vectors(20, 4, seed: 6008),
        metadata: %{source: "external_system", timestamp: System.system_time(:microsecond)}
      }
      
      # Publish event to event bus
      EventBridge.handle_external_event(:data_ingestion, data_ingestion_event)
      
      # Wait for processing
      Process.sleep(200)
      
      # Verify vectors were automatically ingested
      stats = System1.Operations.get_vector_count()
      assert stats.total_vectors >= 20
      
      # Verify metadata was preserved
      metadata_entries = System1.Operations.search_by_metadata(%{source: "external_system"})
      assert length(metadata_entries) >= 20
    end
    
    test "coordinates with external VSM subsystems" do
      # Simulate S4 (Intelligence) requesting vector analytics
      analytics_request = %{
        operation: :vector_analysis,
        parameters: %{
          analysis_type: :clustering_quality,
          sample_size: 100
        }
      }
      
      # Send request through VSM coordination channel
      {:ok, response} = System2.Coordination.handle_external_request(:s4_intelligence, analytics_request)
      
      # Should receive comprehensive analytics
      assert %{
        clustering_metrics: metrics,
        vector_statistics: stats,
        performance_indicators: performance
      } = response
      
      assert is_map(metrics)
      assert is_map(stats)
      assert is_map(performance)
      
      # Verify analytics quality
      assert Map.has_key?(metrics, :silhouette_scores)
      assert Map.has_key?(stats, :dimensionality_distribution)
      assert Map.has_key?(performance, :search_latency_p95)
    end
  end
  
  describe "Cross-Subsystem Workflows" do
    test "complete ML workflow with VSM coordination" do
      # Step 1: Data ingestion (S1)
      vectors = TestHelpers.generate_clustered_vectors(150, 6, 4, seed: 6009)
      {:ok, vector_ids} = System1.Operations.insert_batch(vectors, %{workflow: "ml_pipeline"})
      
      # Step 2: Resource allocation check (S3)
      resource_check = System3.Resources.check_ml_capacity(%{
        operation: :clustering,
        data_size: length(vectors),
        estimated_complexity: :medium
      })
      
      assert resource_check.sufficient_resources == true
      
      # Step 3: ML processing coordination (S2)
      workflow_id = System2.Coordination.start_ml_workflow(%{
        vectors: vector_ids,
        operations: [:clustering, :anomaly_detection, :pattern_recognition]
      })
      
      # Step 4: Execute ML operations (S4)
      clustering_result = System4.ML.cluster_kmeans(4, %{workflow_id: workflow_id})
      anomaly_result = System4.AnomalyDetection.detect(%{workflow_id: workflow_id})
      pattern_result = System4.PatternRecognition.detect_patterns(%{workflow_id: workflow_id})
      
      # Step 5: Policy compliance check (S5)
      compliance_check = System5.Policy.validate_ml_results([
        clustering_result, anomaly_result, pattern_result
      ])
      
      assert compliance_check.compliant == true
      
      # Step 6: Workflow completion (S2)
      {:ok, workflow_summary} = System2.Coordination.complete_ml_workflow(workflow_id)
      
      # Verify complete workflow execution
      assert workflow_summary.status == :completed
      assert length(workflow_summary.operations_completed) == 3
      assert workflow_summary.total_vectors_processed == length(vectors)
    end
    
    test "handles workflow failures with proper error propagation" do
      # Create scenario that will fail
      invalid_vectors = [
        [1.0, 2.0, Float.nan()],  # NaN values
        [Float.infinity(), 2.0, 3.0],  # Infinity values
        [1.0, 2.0, 3.0]  # Valid vector
      ]
      
      # Attempt insertion - should fail validation
      {:error, validation_error} = System1.Operations.insert_batch(invalid_vectors, %{})
      
      # Verify error propagation through VSM channels
      Process.sleep(100)
      
      # S3 should log the error
      error_logs = System3.Monitoring.get_recent_errors()
      assert length(error_logs) > 0
      assert Enum.any?(error_logs, fn log -> 
        log.subsystem == :s1_operations and log.error_type == :validation_error
      end)
      
      # S2 should coordinate error response
      coordination_state = System2.Coordination.get_error_handling_state()
      assert coordination_state.active_error_responses > 0
      
      # S5 should evaluate if policy changes are needed
      policy_review = System5.Policy.get_error_triggered_reviews()
      assert length(policy_review) > 0
    end
    
    test "adaptive performance optimization across subsystems" do
      # Generate load that triggers optimization
      vectors = TestHelpers.generate_random_vectors(500, 10, seed: 6010)
      {:ok, _vector_ids} = System1.Operations.insert_batch(vectors, %{})
      
      # Simulate heavy search load
      search_tasks = 1..20
      |> Enum.map(fn i ->
        Task.async(fn ->
          query = TestHelpers.generate_random_vectors(1, 10, seed: i * 500) |> hd()
          System1.Operations.search_knn(query, 10, %{})
        end)
      end)
      
      # S3 should detect performance degradation
      Process.sleep(1000)  # Allow some searches to complete
      performance_state = System3.Monitoring.get_performance_metrics()
      
      if performance_state.search_latency_p95 > 1000 do  # > 1 second
        # S3 should trigger optimization
        optimization_result = System3.Optimization.trigger_adaptive_optimization()
        assert optimization_result.optimizations_applied > 0
        
        # S2 should coordinate load balancing
        load_balancing = System2.Coordination.activate_load_balancing()
        assert load_balancing.strategy in [:round_robin, :least_loaded, :adaptive]
        
        # S5 should evaluate if policy adjustments are needed
        policy_adjustment = System5.Policy.evaluate_performance_policies(performance_state)
        assert is_map(policy_adjustment)
      end
      
      # Wait for all searches to complete
      Task.await_many(search_tasks, 30000)
      
      # Verify system adaptation
      final_performance = System3.Monitoring.get_performance_metrics()
      
      # Performance should be stable or improved
      assert final_performance.system_stability >= :stable
    end
  end
  
  describe "VSM Recursion and Self-Organization" do
    test "vector store exhibits recursive VSM structure" do
      # Vector store should contain viable subsystems that themselves follow VSM principles
      subsystem_health = %{
        s1: System1.health_check(),
        s2: System2.health_check(),
        s3: System3.health_check(), 
        s4: System4.health_check(),
        s5: System5.health_check()
      }
      
      # Each subsystem should have its own VSM structure
      Enum.each(subsystem_health, fn {subsystem, health} ->
        assert health.has_operations_layer == true  # S1 equivalent
        assert health.has_coordination_layer == true  # S2 equivalent  
        assert health.has_control_layer == true  # S3 equivalent
        assert health.has_intelligence_layer == true  # S4 equivalent
        assert health.has_policy_layer == true  # S5 equivalent
        assert health.viable == true
      end)
    end
    
    test "demonstrates variety engineering principles" do
      # Create high variety input (diverse vector types and operations)
      diverse_vectors = [
        TestHelpers.generate_random_vectors(50, 4, seed: 7001),      # Random
        TestHelpers.generate_clustered_vectors(50, 4, 3, seed: 7002), # Clustered  
        TestHelpers.generate_sparse_vectors(50, 4, 0.1, seed: 7003),  # Sparse
        TestHelpers.generate_unit_vectors(50, 4, seed: 7004)          # Normalized
      ] |> List.flatten()
      
      # System should attenuate variety appropriately
      {:ok, vector_ids} = System1.Operations.insert_batch(diverse_vectors, %{variety: "high"})
      
      # S2 should coordinate variety management
      variety_state = System2.Coordination.get_variety_management()
      assert variety_state.input_variety > variety_state.output_variety  # Attenuation
      
      # S3 should control variety levels
      control_metrics = System3.Control.get_variety_metrics()
      assert control_metrics.variety_balance in [:attenuated, :amplified, :balanced]
      
      # S4 should analyze variety patterns
      variety_analysis = System4.Analytics.analyze_variety_patterns()
      assert Map.has_key?(variety_analysis, :entropy_measures)
      assert Map.has_key?(variety_analysis, :complexity_indicators)
      
      # Perform diverse operations (amplify variety)
      clustering_tasks = Task.async(fn -> System4.ML.cluster_kmeans(5, %{}) end)
      anomaly_tasks = Task.async(fn -> System4.AnomalyDetection.detect(%{}) end)
      pattern_tasks = Task.async(fn -> System4.PatternRecognition.detect_patterns(%{}) end)
      
      # Wait for variety amplification
      Task.await_many([clustering_tasks, anomaly_tasks, pattern_tasks], 10000)
      
      # Final variety state should demonstrate proper engineering
      final_variety_state = System2.Coordination.get_variety_management()
      assert final_variety_state.variety_engineering_effective == true
    end
    
    test "exhibits self-organization and adaptation" do
      # Initial configuration
      initial_config = System5.Policy.get_current_configuration()
      
      # Create conditions that should trigger self-organization
      # Load the system with different patterns over time
      time_series_data = 1..10
      |> Enum.map(fn iteration ->
        # Each iteration has different characteristics
        case rem(iteration, 3) do
          0 -> TestHelpers.generate_clustered_vectors(30, 6, 2, seed: iteration)
          1 -> TestHelpers.generate_random_vectors(30, 6, seed: iteration)
          2 -> TestHelpers.generate_sparse_vectors(30, 6, 0.2, seed: iteration)
        end
      end)
      
      # Process data over time to trigger adaptation
      Enum.each(time_series_data, fn vectors ->
        {:ok, _} = System1.Operations.insert_batch(vectors, %{})
        Process.sleep(200)  # Allow adaptation time
      end)
      
      # Wait for self-organization to occur
      Process.sleep(1000)
      
      # System should have self-organized
      adaptation_metrics = System2.Coordination.get_adaptation_metrics()
      assert adaptation_metrics.adaptations_made > 0
      
      # Configuration should have evolved
      final_config = System5.Policy.get_current_configuration()
      config_changes = System5.Policy.compare_configurations(initial_config, final_config)
      assert config_changes.parameters_changed > 0
      
      # Performance should be optimized for observed patterns
      performance_evolution = System3.Monitoring.get_performance_evolution()
      assert performance_evolution.trend in [:improving, :stable]
      assert performance_evolution.adaptation_effectiveness > 0.5
    end
  end
  
  describe "VSM Algedonic System Integration" do
    test "pain signals propagate correctly through VSM hierarchy" do
      # Create a pain condition (system overload)
      overload_vectors = TestHelpers.generate_random_vectors(2000, 32, seed: 8001)
      
      # Attempt massive insertion to trigger pain signals
      spawn(fn ->
        System1.Operations.insert_batch(overload_vectors, %{source: "overload_test"})
      end)
      
      # Monitor for algedonic pain signals
      Process.sleep(500)
      
      algedonic_state = System3.Monitoring.get_algedonic_state()
      pain_signals = Enum.filter(algedonic_state.signals, & &1.type == :pain)
      
      assert length(pain_signals) > 0
      
      # Pain signals should propagate up the hierarchy
      s2_algedonic = System2.Coordination.get_received_algedonic_signals()
      s5_algedonic = System5.Policy.get_received_algedonic_signals()
      
      assert length(s2_algedonic) > 0
      assert length(s5_algedonic) > 0
      
      # System should respond to pain
      pain_response = System2.Coordination.get_pain_response_actions()
      assert length(pain_response.actions_taken) > 0
      assert :load_shedding in pain_response.strategies_activated or
             :resource_throttling in pain_response.strategies_activated
    end
    
    test "pleasure signals reinforce successful patterns" do
      # Create conditions for pleasure signals (successful operations)
      well_structured_vectors = TestHelpers.generate_clustered_vectors(100, 8, 4, seed: 8002)
      
      # Perform successful operations
      {:ok, vector_ids} = System1.Operations.insert_batch(well_structured_vectors, %{})
      {:ok, cluster_result} = System4.ML.cluster_kmeans(4, %{})
      {:ok, search_results} = System1.Operations.search_knn(hd(well_structured_vectors), 10, %{})
      
      # All operations should complete successfully and efficiently
      Process.sleep(300)
      
      algedonic_state = System3.Monitoring.get_algedonic_state()
      pleasure_signals = Enum.filter(algedonic_state.signals, & &1.type == :pleasure)
      
      assert length(pleasure_signals) > 0
      
      # Pleasure should reinforce current configuration
      reinforcement_state = System5.Policy.get_reinforcement_state()
      assert reinforcement_state.positive_reinforcements > 0
      
      # System should preserve successful patterns
      pattern_preservation = System4.PatternRecognition.get_preserved_patterns()
      assert length(pattern_preservation.successful_patterns) > 0
    end
    
    test "algedonic interrupts trigger immediate attention" do
      # Simulate critical system condition
      critical_condition = %{
        type: :memory_exhaustion,
        severity: :critical,
        affected_subsystems: [:s1, :s4],
        immediate_action_required: true
      }
      
      # Trigger algedonic interrupt
      System3.Monitoring.trigger_algedonic_interrupt(critical_condition)
      
      # Should interrupt normal processing immediately
      Process.sleep(100)
      
      interrupt_state = System2.Coordination.get_interrupt_handling_state()
      assert interrupt_state.active_interrupts > 0
      assert interrupt_state.processing_suspended == true
      
      # All subsystems should receive interrupt signal
      subsystem_states = %{
        s1: System1.get_interrupt_state(),
        s2: System2.get_interrupt_state(),  
        s3: System3.get_interrupt_state(),
        s4: System4.get_interrupt_state(),
        s5: System5.get_interrupt_state()
      }
      
      Enum.each(subsystem_states, fn {subsystem, state} ->
        assert state.interrupt_received == true
        assert state.interrupt_type == :memory_exhaustion
      end)
      
      # System should implement emergency response
      emergency_response = System3.Control.get_emergency_response()
      assert emergency_response.response_activated == true
      assert :memory_cleanup in emergency_response.actions_taken
    end
  end
  
  # Telemetry event handler for testing
  def handle_telemetry_event(event_name, measurements, metadata, %{test_pid: test_pid}) do
    send(test_pid, {:telemetry_event, event_name, measurements, metadata})
  end
  
  # Helper functions for test data generation
  
  defp generate_sparse_vectors(count, dimensions, sparsity, opts \\ []) do
    seed = Keyword.get(opts, :seed)
    if seed, do: :rand.seed(:exsss, {seed, seed, seed})
    
    non_zero_count = round(dimensions * sparsity)
    
    1..count
    |> Enum.map(fn _ ->
      vector = List.duplicate(0.0, dimensions)
      indices = Enum.take_random(0..(dimensions-1), non_zero_count)
      
      Enum.reduce(indices, vector, fn idx, acc ->
        List.replace_at(acc, idx, :rand.normal() * 2.0)
      end)
    end)
  end
  
  defp generate_unit_vectors(count, dimensions, opts \\ []) do
    seed = Keyword.get(opts, :seed)
    if seed, do: :rand.seed(:exsss, {seed, seed, seed})
    
    1..count
    |> Enum.map(fn _ ->
      vector = TestHelpers.generate_random_vectors(1, dimensions, seed: nil) |> hd()
      norm = TestHelpers.vector_norm(vector)
      if norm > 0, do: Enum.map(vector, &(&1 / norm)), else: vector
    end)
  end
end