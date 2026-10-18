use crate::dataset::RawMnistDataItems;
use anyhow::{bail, Context, Result};
use burn::data::dataset::Dataset;
use std::{
    fs::{create_dir_all, File},
    io::Write,
    path::Path,
};

pub trait SavePgmImage {
    fn save_file(&self, idx: usize, path: &Path) -> Result<()>;
}

impl SavePgmImage for RawMnistDataItems {
    // PGM supports grayscale pixels directly an no image library is needed
    fn save_file(&self, idx: usize, path: &Path) -> Result<()> {
        let item = self.get(idx).context(format!("Getting item {}", idx))?;

        let label_str = label_code_to_str(item.label).unwrap();
        let filename = format!("{}_{}.pgm", item.idx, label_str);

        create_dir_all(path).context(format!("Creating directory {}", path.display()))?;

        let mut file = File::create(path.join(&filename))
            .context(format!("Creating pgm export file in {}", path.display()))?;
        write!(file, "P5\n{} {}\n255\n", self.cols, self.rows)?;

        file.write_all(&self.images[item.pixels])?;

        Ok(())
    }
}

fn label_code_to_str(code: u8) -> Result<&'static str> {
    match code {
        0 => Ok("tshirttop"),
        1 => Ok("trouser"),
        2 => Ok("pullover"),
        3 => Ok("dress"),
        4 => Ok("coat"),
        5 => Ok("sandal"),
        6 => Ok("shirt"),
        7 => Ok("sneaker"),
        8 => Ok("bag"),
        9 => Ok("ankleboot"),
        _ => bail!("Cannot name a label from code {}", code),
    }
}
