use std::path::PathBuf;

use crate::embed::{check_embedder, Embedder};
use crate::error::{Error, Result};
use crate::meta::{IndexConfig, Meta};
use crate::pack::{pack_1bit_query, stride_bytes};
use crate::search::{top_k, Hit};
use crate::store::{
    index_dir, publish_staging, remove_leftover_staging, write_staging, PublishBatch,
    PublishedSnapshot,
};

pub struct Index {
    pub(crate) dir: PathBuf,
    meta: Meta,
    snapshot: Option<PublishedSnapshot>,
    pending_ids: Vec<[u8; 16]>,
    pending_vectors: Vec<u8>,
}

impl Index {
    pub(crate) fn create(dir: PathBuf, cfg: IndexConfig) -> Result<Self> {
        remove_leftover_staging(&dir);
        let meta = Meta::new(cfg.dim, "");
        let batch = PublishBatch {
            ids: Vec::new(),
            vectors: Vec::new(),
            meta: meta.clone(),
        };
        let staging = write_staging(&dir, 0, &batch)?;
        publish_staging(&dir, &staging)?;
        let snapshot = PublishedSnapshot::load(&dir)?;
        Ok(Self {
            dir,
            meta,
            snapshot,
            pending_ids: Vec::new(),
            pending_vectors: Vec::new(),
        })
    }

    pub(crate) fn open(dir: PathBuf) -> Result<Self> {
        remove_leftover_staging(&dir);
        let snapshot = PublishedSnapshot::load(&dir)?;
        let meta = snapshot
            .as_ref()
            .map(|s| s.meta.clone())
            .ok_or_else(|| Error::IndexNotFound(dir.to_string_lossy().to_string()))?;
        Ok(Self {
            dir,
            meta,
            snapshot,
            pending_ids: Vec::new(),
            pending_vectors: Vec::new(),
        })
    }

    pub fn dir(&self) -> &PathBuf {
        &self.dir
    }

    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    pub fn len(&self) -> usize {
        self.meta.count + self.pending_ids.len()
    }

    pub fn add_texts(
        &mut self,
        embedder: &dyn Embedder,
        ids: &[[u8; 16]],
        texts: &[&str],
    ) -> Result<()> {
        if ids.len() != texts.len() {
            return Err(Error::InvalidArgument(
                "ids and texts length mismatch".into(),
            ));
        }
        self.ensure_embedder(embedder)?;
        let vectors = embedder.embed(texts)?;
        let flat: Vec<f32> = vectors.into_iter().flatten().collect();
        self.add_vectors(ids, &flat)?;
        Ok(())
    }

    pub fn add_vectors(&mut self, ids: &[[u8; 16]], vectors: &[f32]) -> Result<()> {
        let n = ids.len();
        if n == 0 {
            return Ok(());
        }
        if vectors.len() != n * self.meta.dim {
            return Err(Error::DimMismatch {
                expected: self.meta.dim,
                got: if n > 0 { vectors.len() / n } else { 0 },
            });
        }
        let stride = self.meta.stride_bytes;
        for i in 0..n {
            let start = i * self.meta.dim;
            let row_vec = &vectors[start..start + self.meta.dim];
            let mut row = vec![0u8; stride];
            crate::quant::quantize_row(self.meta.dim, row_vec, &mut row);
            self.pending_ids.push(ids[i]);
            self.pending_vectors.extend_from_slice(&row);
        }
        Ok(())
    }

    pub fn publish(&mut self) -> Result<()> {
        if self.pending_ids.is_empty() {
            return Err(Error::NothingToPublish);
        }
        let mut all_ids: Vec<[u8; 16]> = Vec::new();
        let mut all_vectors: Vec<u8> = Vec::new();

        if let Some(snap) = &self.snapshot {
            if snap.meta.count > 0 {
                all_ids.extend_from_slice(snap.ids());
                all_vectors.extend_from_slice(snap.vectors());
            }
        }

        all_ids.extend_from_slice(&self.pending_ids);
        all_vectors.extend_from_slice(&self.pending_vectors);

        let generation = self.meta.generation + 1;
        let count = all_ids.len();
        let mut meta = self.meta.clone();
        meta.count = count;
        meta.generation = generation;
        meta.stride_bytes = stride_bytes(meta.dim);

        let batch = PublishBatch {
            ids: all_ids,
            vectors: all_vectors,
            meta: meta.clone(),
        };

        let staging = write_staging(&self.dir, generation, &batch)?;
        publish_staging(&self.dir, &staging)?;

        self.meta = meta;
        self.snapshot = PublishedSnapshot::load(&self.dir)?;
        self.pending_ids.clear();
        self.pending_vectors.clear();
        Ok(())
    }

    pub fn search_text(&self, embedder: &dyn Embedder, query: &str, k: usize) -> Result<Vec<Hit>> {
        check_embedder(embedder, self.meta.dim, &self.meta.embedder_id)?;
        let vectors = embedder.embed(&[query])?;
        let q = &vectors[0];
        self.search_vector(q, k)
    }

    pub fn search_vector(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        let snap = self.snapshot.as_ref().ok_or(Error::EmptyIndex)?;
        if snap.meta.count == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.meta.dim {
            return Err(Error::DimMismatch {
                expected: self.meta.dim,
                got: query.len(),
            });
        }
        let stride = snap.meta.stride_bytes;
        let mut query_packed = vec![0u8; stride];
        pack_1bit_query(snap.meta.dim, query, &mut query_packed);
        Ok(top_k(
            snap.meta.dim,
            stride,
            snap.meta.count,
            snap.ids(),
            snap.vectors(),
            &query_packed,
            k,
        ))
    }

    fn ensure_embedder(&mut self, embedder: &dyn Embedder) -> Result<()> {
        if embedder.dim() != self.meta.dim {
            return Err(Error::DimMismatch {
                expected: self.meta.dim,
                got: embedder.dim(),
            });
        }
        if self.meta.embedder_id.is_empty() {
            self.meta.embedder_id = embedder.id().to_string();
            return Ok(());
        }
        check_embedder(embedder, self.meta.dim, &self.meta.embedder_id)
    }
}

pub fn create_index_at(root: &std::path::Path, index_id: &str, cfg: IndexConfig) -> Result<Index> {
    let dir = index_dir(root, index_id)?;
    if dir.exists() {
        return Err(Error::IndexExists(index_id.to_string()));
    }
    std::fs::create_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
    Index::create(dir, cfg)
}

pub fn open_index_at(root: &std::path::Path, index_id: &str) -> Result<Index> {
    let dir = index_dir(root, index_id)?;
    if !dir.join(crate::store::META_FILE).exists() {
        return Err(Error::IndexNotFound(index_id.to_string()));
    }
    Index::open(dir)
}
