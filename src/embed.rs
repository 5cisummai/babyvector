use crate::error::{Error, Result};

pub trait Embedder: Send + Sync {
    fn id(&self) -> &str;
    fn dim(&self) -> usize;
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

/// Deterministic fake embedder for tests: hashes text into a fixed-dim vector.
pub struct HashEmbedder {
    id: String,
    dim: usize,
}

impl HashEmbedder {
    pub fn new(id: impl Into<String>, dim: usize) -> Self {
        Self {
            id: id.into(),
            dim,
        }
    }
}

impl Embedder for HashEmbedder {
    fn id(&self) -> &str {
        &self.id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            let mut v = vec![0.0f32; self.dim];
            let mut hash = 0xcbf29ce484222325u64;
            for byte in text.as_bytes() {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
                let idx = (hash as usize) % self.dim;
                v[idx] += 1.0;
                hash = hash.rotate_left(13);
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            out.push(v);
        }
        Ok(out)
    }
}

pub fn check_embedder(embedder: &dyn Embedder, meta_dim: usize, meta_embedder_id: &str) -> Result<()> {
    if embedder.dim() != meta_dim {
        return Err(Error::DimMismatch {
            expected: meta_dim,
            got: embedder.dim(),
        });
    }
    if embedder.id() != meta_embedder_id {
        return Err(Error::EmbedderMismatch {
            expected: meta_embedder_id.to_string(),
            got: embedder.id().to_string(),
        });
    }
    Ok(())
}
