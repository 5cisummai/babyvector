use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::index::{create_index_at, open_index_at, Index};
use crate::meta::IndexConfig;
use crate::store::index_dir;

pub struct Engine {
    root: PathBuf,
}

impl Engine {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root).map_err(|e| Error::io(&root, e))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn create(&self, index_id: &str, cfg: IndexConfig) -> Result<Index> {
        create_index_at(&self.root, index_id, cfg)
    }

    pub fn open(&self, index_id: &str) -> Result<Index> {
        open_index_at(&self.root, index_id)
    }

    pub fn drop_index(&self, index_id: &str) -> Result<()> {
        let dir = index_dir(&self.root, index_id)?;
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
        }
        Ok(())
    }
}
