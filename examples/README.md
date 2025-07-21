# VSM Vector Store Examples

This directory contains examples demonstrating the functionality of the VSM Vector Store.

## Running Examples

All examples can be run directly with Elixir:

```bash
elixir examples/basic_usage.exs
```

Or made executable and run directly:

```bash
./examples/basic_usage.exs
```

## Available Examples

### basic_usage.exs
Demonstrates the core functionality:
- Creating vector spaces
- Inserting vectors with metadata
- Searching for similar vectors
- Retrieving specific vectors
- Updating metadata
- Getting space statistics
- Managing vector spaces

### More Examples Coming Soon
- Advanced search with different distance metrics
- Batch operations for bulk processing
- Clustering and anomaly detection
- Performance benchmarking
- Multi-space operations
- Concurrent usage patterns

## Example Output

```
=== VSM Vector Store Basic Usage Example ===

1. Creating vector space...
   Created space: space_a1b2c3d4e5f6

2. Inserting vectors...
   Inserted user_1 (Alice)
   Inserted user_2 (Bob)
   Inserted user_3 (Charlie)
   Inserted user_4 (David)
   Inserted user_5 (Eve)

3. Searching for similar vectors...
   Top 3 most similar vectors:
   - user_3 (Charlie): distance = 3.4521
   - user_1 (Alice): distance = 3.8734
   - user_5 (Eve): distance = 4.2156

...
```