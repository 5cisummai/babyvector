use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

use fs4::fs_std::FileExt;
use memmap2::{Mmap, MmapOptions};

use crate::error::{Error, Result};
use crate::meta::{Meta, ID_WIDTH};

pub const META_FILE: &str = "meta.json";
pub const IDS_FILE: &str = "ids.bin";
pub const VECTORS_FILE: &str = "vectors.bin";
pub const LOCK_FILE: &str = "LOCK";
pub const STAGING_DIR: &str = ".staging";

pub fn validate_index_id(index_id: &str) -> Result<()> {
    if index_id.is_empty() || index_id.len() > 128 {
        return Err(Error::InvalidIndexId(index_id.to_string()));
    }
    if index_id.contains('/') || index_id.contains('\\') || index_id.contains("..") {
        return Err(Error::InvalidIndexId(index_id.to_string()));
    }
    Ok(())
}

pub fn index_dir(root: &Path, index_id: &str) -> Result<PathBuf> {
    validate_index_id(index_id)?;
    Ok(root.join(index_id))
}

pub struct PublishedSnapshot {
    pub meta: Meta,
    pub ids_mmap: Option<Mmap>,
    pub vectors_mmap: Option<Mmap>,
}

impl PublishedSnapshot {
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let meta_path = dir.join(META_FILE);
        if !meta_path.exists() {
            return Ok(None);
        }
        let meta_bytes = fs::read(&meta_path).map_err(|e| Error::io(&meta_path, e))?;
        let meta: Meta = serde_json::from_slice(&meta_bytes)?;
        meta.validate()?;
        if meta.count == 0 {
            return Ok(Some(Self {
                meta,
                ids_mmap: None,
                vectors_mmap: None,
            }));
        }
        let ids_path = dir.join(IDS_FILE);
        let vectors_path = dir.join(VECTORS_FILE);
        let ids_file = File::open(&ids_path).map_err(|e| Error::io(&ids_path, e))?;
        let vectors_file = File::open(&vectors_path).map_err(|e| Error::io(&vectors_path, e))?;
        let ids_mmap = unsafe {
            MmapOptions::new()
                .map(&ids_file)
                .map_err(|e| Error::io(&ids_path, e))?
        };
        let vectors_mmap = unsafe {
            MmapOptions::new()
                .map(&vectors_file)
                .map_err(|e| Error::io(&vectors_path, e))?
        };
        let expected_ids = meta.count * ID_WIDTH;
        let expected_vectors = meta.count * meta.stride_bytes;
        if ids_mmap.len() != expected_ids {
            return Err(Error::InvalidArgument(format!(
                "ids.bin size mismatch: {} vs {}",
                ids_mmap.len(),
                expected_ids
            )));
        }
        if vectors_mmap.len() != expected_vectors {
            return Err(Error::InvalidArgument(format!(
                "vectors.bin size mismatch: {} vs {}",
                vectors_mmap.len(),
                expected_vectors
            )));
        }
        Ok(Some(Self {
            meta,
            ids_mmap: Some(ids_mmap),
            vectors_mmap: Some(vectors_mmap),
        }))
    }

    pub fn ids(&self) -> &[[u8; 16]] {
        if self.meta.count == 0 {
            return &[];
        }
        let bytes = &self.ids_mmap.as_ref().expect("ids mmap")[..];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const [u8; 16], self.meta.count) }
    }

    pub fn vectors(&self) -> &[u8] {
        self.vectors_mmap
            .as_ref()
            .map(|m| m.as_ref())
            .unwrap_or(&[])
    }
}

pub struct PublishBatch {
    pub ids: Vec<[u8; 16]>,
    pub vectors: Vec<u8>,
    pub meta: Meta,
}

pub fn write_staging(dir: &Path, generation: u64, batch: &PublishBatch) -> Result<PathBuf> {
    let staging = dir.join(STAGING_DIR).join(generation.to_string());
    fs::create_dir_all(&staging).map_err(|e| Error::io(&staging, e))?;
    let meta_path = staging.join(META_FILE);
    let ids_path = staging.join(IDS_FILE);
    let vectors_path = staging.join(VECTORS_FILE);
    let meta_json = serde_json::to_vec_pretty(&batch.meta)?;
    fs::write(&meta_path, meta_json).map_err(|e| Error::io(&meta_path, e))?;
    fs::write(&ids_path, ids_as_bytes(&batch.ids)).map_err(|e| Error::io(&ids_path, e))?;
    fs::write(&vectors_path, &batch.vectors).map_err(|e| Error::io(&vectors_path, e))?;
    sync_dir(&staging)?;
    Ok(staging)
}

pub fn publish_staging(index_dir: &Path, staging: &Path) -> Result<()> {
    let lock_path = index_dir.join(LOCK_FILE);
    fs::create_dir_all(index_dir).map_err(|e| Error::io(index_dir, e))?;
    let lock_file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(&lock_path)
        .map_err(|e| Error::io(&lock_path, e))?;
    lock_file
        .lock_exclusive()
        .map_err(|e| Error::io(&lock_path, e))?;

    let result: Result<()> = (|| {
        for name in [IDS_FILE, VECTORS_FILE, META_FILE] {
            let src = staging.join(name);
            if !src.exists() {
                continue;
            }
            let dst = index_dir.join(name);
            let tmp = index_dir.join(format!("{}.tmp", name));
            fs::copy(&src, &tmp).map_err(|e| Error::io(&tmp, e))?;
            sync_file(&tmp)?;
            fs::rename(&tmp, &dst).map_err(|e| Error::io(&dst, e))?;
            sync_dir(index_dir)?;
        }
        Ok(())
    })();

    lock_file.unlock().map_err(|e| Error::io(&lock_path, e))?;
    result?;
    if staging.exists() {
        fs::remove_dir_all(staging).map_err(|e| Error::io(staging, e))?;
    }
    Ok(())
}

pub fn remove_leftover_staging(index_dir: &Path) {
    let staging_root = index_dir.join(STAGING_DIR);
    if staging_root.exists() {
        let _ = fs::remove_dir_all(&staging_root);
    }
}

fn ids_as_bytes(ids: &[[u8; 16]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * ID_WIDTH);
    for id in ids {
        out.extend_from_slice(id);
    }
    out
}

fn sync_file(path: &Path) -> Result<()> {
    let file = File::open(path).map_err(|e| Error::io(path, e))?;
    file.sync_all().map_err(|e| Error::io(path, e))?;
    Ok(())
}

fn sync_dir(_path: &Path) -> Result<()> {
    Ok(())
}
