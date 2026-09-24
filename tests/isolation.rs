use babyvector::{Engine, HashEmbedder, IndexConfig};

fn id(byte: u8) -> [u8; 16] {
    [byte; 16]
}

#[test]
fn indexes_do_not_share_vectors() {
    let root = tempfile::tempdir().unwrap();
    let engine = Engine::new(root.path()).unwrap();
    let embedder = HashEmbedder::new("test", 8);
    let cfg = IndexConfig { dim: 8 };

    let mut a = engine.create("alpha", cfg).unwrap();
    a.add_texts(&embedder, &[id(1)], &["alpha only"]).unwrap();
    a.publish().unwrap();

    let mut b = engine.create("beta", cfg).unwrap();
    b.add_texts(&embedder, &[id(2)], &["beta only"]).unwrap();
    b.publish().unwrap();

    let a = engine.open("alpha").unwrap();
    let b = engine.open("beta").unwrap();

    let hits_a = a.search_text(&embedder, "alpha only", 5).unwrap();
    let hits_b = b.search_text(&embedder, "beta only", 5).unwrap();

    assert_eq!(hits_a.len(), 1);
    assert_eq!(hits_b.len(), 1);
    assert_eq!(hits_a[0].id, id(1));
    assert_eq!(hits_b[0].id, id(2));
}
