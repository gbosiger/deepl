use anyhow::Result;
use std::path::Path;

pub trait SavePgmImage {
    fn save_file(&self, idx: usize, path: &Path) -> Result<()>;
}
