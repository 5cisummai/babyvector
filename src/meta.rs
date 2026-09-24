use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

pub const FORMAT_VERSION: u32 = 1;
pub const ID_WIDTH: usize = 16;
pub const BITS: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub version: u32,
    pub dim: usize,
    pub bits: u8,
    pub count: usize,
    pub generation: u64,
    pub stride_bytes: usize,
    pub id_width: usize,
    pub embedder_id: String,
}

impl Meta {
    pub fn new(dim: usize, embedder_id: &str) -> Self {
        Self {
            version: FORMAT_VERSION,
            dim,
            bits: BITS,
            count: 0,
            generation: 0,
            stride_bytes: crate::pack::stride_bytes(dim),
            id_width: ID_WIDTH,
            embedder_id: embedder_id.to_string(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != FORMAT_VERSION {
            return Err(Error::InvalidArgument(format!(
                "unsupported format version: {}",
                self.version
            )));
        }
        if self.bits != BITS {
            return Err(Error::InvalidArgument(format!(
                "unsupported bits: {} (only 1-bit indexes are supported)",
                self.bits
            )));
        }
        if self.id_width != ID_WIDTH {
            return Err(Error::InvalidArgument(format!(
                "unsupported id width: {}",
                self.id_width
            )));
        }
        if self.stride_bytes != crate::pack::stride_bytes(self.dim) {
            return Err(Error::InvalidArgument(
                "stride_bytes does not match dim".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IndexConfig {
    pub dim: usize,
}
