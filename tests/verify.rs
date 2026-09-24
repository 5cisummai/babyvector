use std::fs;

use babyvector::store::{IDS_FILE, META_FILE, VECTORS_FILE};
use babyvector::{Embedder, Engine, Error, HashEmbedder, IndexConfig};

fn id(byte: u8) -> [u8; 16] {
    [byte; 16]
}

fn stride_bytes(dim: usize) -> usize {
    let bytes = (dim + 7) / 8;
    ((bytes + 7) / 8) * 8
}

#[test]
fn on_disk_layout_matches_meta() {
    let root = tempfile::tempdir().unwrap();
    let engine = Engine::new(root.path()).unwrap();
    let embedder = HashEmbedder::new("verify-model", 32);
    let mut index = engine.create("layout", IndexConfig { dim: 32 }).unwrap();

    index
        .add_texts(
            &embedder,
            &[id(1), id(2), id(3)],
            &["alpha", "beta", "gamma"],
        )
        .unwrap();
    index.publish().unwrap();

    let dir = root.path().join("layout");
    let opened = engine.open("layout").unwrap();
    let meta = opened.meta();
    assert_eq!(meta.version, 1);
    assert_eq!(meta.dim, 32);
    assert_eq!(meta.bits, 1);
    assert_eq!(meta.count, 3);
    assert_eq!(meta.generation, 1);
    assert_eq!(meta.id_width, 16);
    assert_eq!(meta.embedder_id, "verify-model");
    assert_eq!(meta.stride_bytes, stride_bytes(32));

    let ids = fs::metadata(dir.join(IDS_FILE)).unwrap().len();
    let vectors = fs::metadata(dir.join(VECTORS_FILE)).unwrap().len();
    assert_eq!(ids, (meta.count * meta.id_width) as u64);
    assert_eq!(vectors, (meta.count * meta.stride_bytes) as u64);
    assert!(!dir.join("scales.bin").exists());
}

#[test]
fn self_query_is_top_hit() {
    let root = tempfile::tempdir().unwrap();
    let engine = Engine::new(root.path()).unwrap();
    let dim = 8;
    let mut index = engine.create("self", IndexConfig { dim }).unwrap();
    let mut vectors = vec![-1.0f32; dim * 3];
    vectors[0] = 1.0;
    vectors[dim + 1] = 1.0;
    vectors[2 * dim + 2] = 1.0;
    index
        .add_vectors(&[id(10), id(11), id(12)], &vectors)
        .unwrap();
    index.publish().unwrap();

    let hits = index.search_vector(&vectors[dim..2 * dim], 3).unwrap();
    assert_eq!(hits[0].id, id(11), "scores={:?}", hits);
    assert_eq!(hits.len(), 3);
}

#[test]
fn identical_one_bit_codes_score_plus_dim() {
    let dim = 16;
    let embedder = HashEmbedder::new("verify-model", dim);
    let vector = embedder.embed(&["same"]).unwrap().remove(0);
    let root = tempfile::tempdir().unwrap();
    let engine = Engine::new(root.path()).unwrap();
    let mut index = engine.create("score", IndexConfig { dim }).unwrap();
    index.add_vectors(&[id(1)], &vector).unwrap();
    index.publish().unwrap();

    let hits = index.search_vector(&vector, 1).unwrap();
    assert_eq!(hits[0].id, id(1));
    assert!(
        (hits[0].score - dim as f32).abs() < 1e-5,
        "score={}",
        hits[0].score
    );
}

#[test]
fn drop_index_removes_only_that_directory() {
    let root = tempfile::tempdir().unwrap();
    let engine = Engine::new(root.path()).unwrap();
    let embedder = HashEmbedder::new("verify-model", 8);
    let cfg = IndexConfig { dim: 8 };
    let mut keep = engine.create("keep", cfg).unwrap();
    keep.add_texts(&embedder, &[id(1)], &["keep me"]).unwrap();
    keep.publish().unwrap();
    let mut gone = engine.create("gone", cfg).unwrap();
    gone.add_texts(&embedder, &[id(2)], &["delete me"]).unwrap();
    gone.publish().unwrap();

    engine.drop_index("gone").unwrap();
    assert!(root.path().join("keep").join(META_FILE).exists());
    assert!(!root.path().join("gone").exists());
    match engine.open("gone") {
        Err(Error::IndexNotFound(_)) => {}
        Err(err) => panic!("expected IndexNotFound, got {err}"),
        Ok(_) => panic!("expected IndexNotFound, opened gone"),
    }
    let keep = engine.open("keep").unwrap();
    assert_eq!(
        keep.search_text(&embedder, "keep me", 1).unwrap()[0].id,
        id(1)
    );
}

#[test]
fn rejects_path_escape_index_ids() {
    let root = tempfile::tempdir().unwrap();
    let engine = Engine::new(root.path()).unwrap();
    let cfg = IndexConfig { dim: 8 };
    for bad in ["../escape", "a/b", "a\\b", "foo..bar"] {
        match engine.create(bad, cfg) {
            Err(Error::InvalidIndexId(_)) => {}
            Err(err) => panic!("expected InvalidIndexId for {bad}, got {err}"),
            Ok(_) => panic!("expected InvalidIndexId for {bad}"),
        }
    }
}

#[test]
fn dim_mismatch_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let engine = Engine::new(root.path()).unwrap();
    let mut index = engine.create("dims", IndexConfig { dim: 8 }).unwrap();
    let err = index.add_vectors(&[id(1)], &[1.0, 2.0]).unwrap_err();
    assert!(matches!(err, Error::DimMismatch { expected: 8, got: 2 }));
}

#[test]
fn unpublished_rows_are_not_searchable() {
    let root = tempfile::tempdir().unwrap();
    let engine = Engine::new(root.path()).unwrap();
    let embedder = HashEmbedder::new("verify-model", 8);
    let mut index = engine.create("pending", IndexConfig { dim: 8 }).unwrap();
    index.add_texts(&embedder, &[id(1)], &["secret"]).unwrap();
    assert_eq!(index.len(), 1);
    let hits = index.search_text(&embedder, "secret", 5).unwrap();
    assert!(hits.is_empty());
}
