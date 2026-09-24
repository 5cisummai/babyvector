use std::fs;

use babyvector::store::{META_FILE, STAGING_DIR, VECTORS_FILE};
use babyvector::{Engine, HashEmbedder, IndexConfig};

fn id(byte: u8) -> [u8; 16] {
    [byte; 16]
}

#[test]
fn leftover_staging_does_not_affect_published_search() {
    let root = tempfile::tempdir().unwrap();
    let engine = Engine::new(root.path()).unwrap();
    let embedder = HashEmbedder::new("test", 16);
    let cfg = IndexConfig { dim: 16 };

    let mut index = engine.create("main", cfg).unwrap();
    index
        .add_texts(&embedder, &[id(9)], &["published document"])
        .unwrap();
    index.publish().unwrap();

    let baseline = index.search_text(&embedder, "published document", 1).unwrap();
    assert_eq!(baseline.len(), 1);
    assert_eq!(baseline[0].id, id(9));

    let index_dir = root.path().join("main");
    let bogus = index_dir.join(STAGING_DIR).join("999");
    fs::create_dir_all(&bogus).unwrap();
    fs::write(bogus.join(META_FILE), b"{not valid json").unwrap();
    fs::write(bogus.join(VECTORS_FILE), b"garbage").unwrap();

    let reopened = engine.open("main").unwrap();
    let hits = reopened
        .search_text(&embedder, "published document", 1)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, id(9));
}

#[test]
fn publish_appends_to_existing_snapshot() {
    let root = tempfile::tempdir().unwrap();
    let engine = Engine::new(root.path()).unwrap();
    let embedder = HashEmbedder::new("test", 8);
    let cfg = IndexConfig { dim: 8 };

    let mut index = engine.create("grow", cfg).unwrap();
    index.add_texts(&embedder, &[id(1)], &["first"]).unwrap();
    index.publish().unwrap();
    index.add_texts(&embedder, &[id(2)], &["second"]).unwrap();
    index.publish().unwrap();

    let index = engine.open("grow").unwrap();
    assert_eq!(index.len(), 2);
    let hits = index.search_text(&embedder, "second", 2).unwrap();
    assert_eq!(hits.len(), 2);
}
