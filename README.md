# babyvector

Extremely light per-directory **1-bit** quantized vector index. One index is one folder on disk; search never reads another index’s files.

This crate does **not** bundle an embedding model. Implement [`Embedder`](src/embed.rs) (or use [`HashEmbedder`](src/embed.rs) in tests) to turn text into `f32` vectors before sign quantization.

## Layout

```
{root}/{index_id}/
  meta.json      # dim, bits=1, count, generation, embedder id
  ids.bin        # 16-byte opaque ids
  vectors.bin    # packed 1-bit rows (row-major, 8-byte aligned)
  LOCK           # advisory lock during publish
  .staging/      # removed on open; publish writes then swaps in
```

## Example

```rust
use babyvector::{Engine, HashEmbedder, IndexConfig};

let engine = Engine::new("./data/indexes")?;
let embedder = HashEmbedder::new("my-model", 128);
let cfg = IndexConfig { dim: 128 };

let mut index = engine.create("docs", cfg)?;
let id = [0u8; 16];
index.add_texts(&embedder, &[id], &["hello world"])?;
index.publish()?;

let hits = index.search_text(&embedder, "hello", 10)?;
```

## Search

v1 brute-force scans packed rows. Each dim is a sign bit. Score is Hamming similarity `dim - 2 * hamming`.

## Non-goals

- 2-bit or higher quantization
- Hosted multi-tenant vector database
- ANN graphs (HNSW / IVF)
- Crawling, passage storage, or ONNX/GGUF embedding runtimes
- Hosted embedding runtimes (Go embeds via HTTP, then `bv_add_vectors`)
