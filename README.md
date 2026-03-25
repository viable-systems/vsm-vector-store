# vsm-vector-store

In-memory vector database written in pure Elixir. Implements HNSW indexing, k-means clustering, product quantization, anomaly detection, and pattern recognition. No external ML dependencies.

## Status

Version 0.1.0. Not published to Hex. Core storage and ML modules are implemented. The API works through the `VSMVectorStore` facade module. Some operations (e.g., `optimize/1`) are stubs returning `:ok`.

## What it does

| Operation | Implementation |
|-----------|---------------|
| Nearest-neighbor search | HNSW (Hierarchical Navigable Small World) graph |
| Clustering | k-means with k-means++ initialization |
| Compression | Product quantization (configurable bit width) |
| Anomaly detection | Isolation forest, LOF, statistical (z-score/IQR) |
| Pattern recognition | Density peaks, similarity graph, temporal patterns |

All algorithms are implemented from scratch in Elixir. There are no NIFs, no Python interop, and no calls to external services.

## Repository structure

| Path | Contents |
|------|----------|
| `lib/vsm_vector_store.ex` | Public API facade |
| `lib/vsm_vector_store/storage/` | HNSW index, vector ops, space management, storage manager (5 files) |
| `lib/vsm_vector_store/indexing/` | k-means, product quantization (3 files) |
| `lib/vsm_vector_store/ml/` | Anomaly detection, pattern recognition (3 files) |
| `lib/vsm_vector_store/application.ex` | OTP application with supervision tree |
| `lib/vsm_vector_store/core.ex` | Shared types and helpers |
| `lib/vsm_vector_store/telemetry_reporter.ex` | Metrics reporter |
| `test/unit/` | HNSW, k-means, quantization, anomaly detection tests |
| `test/integration/` | VSM integration test |
| `test/performance/` | Benchmark tests |
| `test/property/` | Property-based tests |

## Module count

14 `.ex` files under `lib/`. 10 test files under `test/`.

## Dependencies

| Category | Packages |
|----------|----------|
| Telemetry | telemetry, telemetry_metrics |
| Serialization | jason |
| Dev/test | ex_doc, excoveralls, benchee, stream_data, dialyxir, credo |

Requires Elixir >= 1.17. No runtime dependencies beyond telemetry and jason.

## Quick start

```bash
git clone https://github.com/viable-systems/vsm-vector-store.git
cd vsm-vector-store
mix deps.get
mix test
```

```elixir
{:ok, _} = VSMVectorStore.start()

{:ok, space_id} = VSMVectorStore.create_space("embeddings", 384)

vectors = [
  List.duplicate(0.1, 384),
  List.duplicate(0.4, 384),
  List.duplicate(0.7, 384)
]
{:ok, ids} = VSMVectorStore.insert(space_id, vectors)

query = List.duplicate(0.15, 384)
{:ok, results} = VSMVectorStore.search(space_id, query, k: 10)

{:ok, clusters} = VSMVectorStore.cluster(space_id, k: 3)
{:ok, anomalies} = VSMVectorStore.detect_anomalies(space_id)
```

## Configuration

```elixir
config :vsm_vector_store,
  hnsw: [
    m: 16,
    ef_construction: 200
  ],
  kmeans: [
    default_init: :kmeans_plus_plus,
    tolerance: 1.0e-4,
    max_iterations: 100
  ],
  quantization: [
    default_bits: 8,
    subvector_size: 8
  ],
  performance: [
    batch_size: 1000,
    cache_size: 10_000,
    gc_interval: 60_000
  ]
```

## OTP supervision tree

```
VSMVectorStore.Application
  MainSupervisor
    Registry
    DynamicSupervisor
    Storage.Supervisor  (HNSW, VectorOps, Space, Manager)
    Indexing.Supervisor  (KMeans, Quantization)
    ML.Supervisor        (AnomalyDetection, PatternRecognition)
    TelemetryReporter
```

## Known limitations

- All data is stored in-memory (ETS/process state). No disk persistence.
- `optimize/1` and `compact/1` are partial stubs.
- The HNSW implementation is a teaching/prototype quality -- not benchmarked against FAISS, Annoy, or ScaNN.
- No SIMD or NIF acceleration; large-dimension / large-dataset performance will be significantly slower than C/Rust implementations.
- Not published to Hex.
- The repo contains several top-level demo/test scripts (`demo.exs`, `test_100_percent.exs`, etc.) that are not part of the library.

## Related packages

- [vsm-pattern-engine](https://github.com/viable-systems/vsm-pattern-engine) -- uses vsm-vector-store for pattern persistence
- [vsm-event-bus](https://github.com/viable-systems/vsm-event-bus) -- event coordination
- [vsm-mcp](https://github.com/viable-systems/vsm-mcp) -- top-level VSM orchestrator
- [vsm-external-interfaces](https://github.com/viable-systems/vsm-external-interfaces) -- HTTP/WS/gRPC adapters

## License

MIT
