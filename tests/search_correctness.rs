use babyvector::{Engine, HashEmbedder, IndexConfig};

fn id(n: u8) -> [u8; 16] {
    [n; 16]
}

#[test]
fn embedder_mismatch_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let engine = Engine::new(root.path()).unwrap();
    let cfg = IndexConfig { dim: 4 };
    let e1 = HashEmbedder::new("model-a", 4);
    let e2 = HashEmbedder::new("model-b", 4);

    let mut index = engine.create("mismatch", cfg).unwrap();
    index.add_texts(&e1, &[id(1)], &["hello"]).unwrap();
    index.publish().unwrap();

    let index = engine.open("mismatch").unwrap();
    let err = index.search_text(&e2, "hello", 1).unwrap_err();
    assert!(matches!(err, babyvector::Error::EmbedderMismatch { .. }));
}
