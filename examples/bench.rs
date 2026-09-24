use std::env;
use std::time::Instant;

use babyvector::{Engine, IndexConfig};

fn id_for(i: usize) -> [u8; 16] {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&(i as u64).to_le_bytes());
    id
}

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn fill_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        out.push(lcg(&mut state));
    }
    out
}

fn percentile(sorted_ns: &[u128], p: f64) -> u128 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted_ns.len() - 1) as f64).round() as usize;
    sorted_ns[idx]
}

fn bench(n: usize, dim: usize, queries: usize) {
    let root = env::temp_dir().join(format!("babyvector-bench-{n}-{dim}"));
    let _ = std::fs::remove_dir_all(&root);
    let engine = Engine::new(&root).expect("engine");
    let mut index = engine
        .create("bench", IndexConfig { dim })
        .expect("create");

    let t_build = Instant::now();
    const BATCH: usize = 4096;
    let mut offset = 0usize;
    while offset < n {
        let batch_n = (n - offset).min(BATCH);
        let ids: Vec<[u8; 16]> = (offset..offset + batch_n).map(id_for).collect();
        let floats = fill_vectors(batch_n, dim, 0x9e3779b97f4a7c15 ^ offset as u64);
        index.add_vectors(&ids, &floats).expect("add");
        offset += batch_n;
    }
    index.publish().expect("publish");
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

    let query = fill_vectors(1, dim, 0xdeadbeefcafebabe);
    index.search_vector(&query, 10).expect("warmup");

    let mut samples = Vec::with_capacity(queries);
    for i in 0..queries {
        let q = fill_vectors(1, dim, 0x1234_5678_9abc_def0 ^ i as u64);
        let t = Instant::now();
        let hits = index.search_vector(&q, 10).expect("search");
        samples.push(t.elapsed().as_nanos());
        assert_eq!(hits.len(), 10.min(n));
    }
    samples.sort_unstable();
    let p50 = percentile(&samples, 50.0) as f64 / 1_000.0;
    let p95 = percentile(&samples, 95.0) as f64 / 1_000.0;
    let mean = samples.iter().sum::<u128>() as f64 / samples.len() as f64 / 1_000.0;
    let qps = 1_000_000.0 / mean;
    let vecs_per_sec = n as f64 / (mean / 1_000_000.0);

    let stride = ((dim + 7) / 8 + 7) / 8 * 8;
    let mb = (n * stride) as f64 / (1024.0 * 1024.0);

    println!(
        "n={n:>8}  dim={dim:>4}  packed={mb:>6.2} MiB  build={build_ms:>8.1} ms  \
         search p50={p50:>8.1} µs  p95={p95:>8.1} µs  ~{qps:.0} qps  ~{:.1} M vec/s",
        vecs_per_sec / 1_000_000.0
    );

    engine.drop_index("bench").ok();
    let _ = std::fs::remove_dir_all(&root);
}

fn main() {
    println!("babyvector 1-bit brute-force search (release, k=10)");
    println!("machine scan of mmap'd vectors.bin — not ANN\n");
    for &(n, dim, queries) in &[
        (10_000, 256, 80),
        (10_000, 768, 80),
        (10_000, 1024, 80),
        (100_000, 256, 40),
        (100_000, 768, 40),
        (100_000, 1024, 40),
        (1_000_000, 768, 12),
        (1_000_000, 1024, 12),
    ] {
        bench(n, dim, queries);
    }
}
