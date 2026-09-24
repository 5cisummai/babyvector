use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid index id: {0}")]
    InvalidIndexId(String),
    #[error("index already exists: {0}")]
    IndexExists(String),
    #[error("index not found: {0}")]
    IndexNotFound(String),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("embedder mismatch: expected {expected}, got {got}")]
    EmbedderMismatch { expected: String, got: String },
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("index is empty")]
    EmptyIndex,
    #[error("nothing to publish")]
    NothingToPublish,
    #[error("index is locked")]
    Locked,
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("embedder error: {0}")]
    Embedder(String),
}

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
