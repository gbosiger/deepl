use crate::pgm::SavePgmImage;
use anyhow::{bail, ensure, Context, Result};
use burn::data::dataset::Dataset;
use flate2::read::GzDecoder;
use std::{
    fs::{create_dir_all, File},
    io::{Read, Write},
    ops::Range,
    path::Path,
};

#[derive(Debug)]
pub struct RawMnistDataItems {
    image_archive: String,
    label_archive: String,
    count: usize,
    rows: usize,
    cols: usize,
    images: Vec<u8>,
    labels: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct RawMnistDataItem {
    pixels: Range<usize>,
    idx: usize,
    label: u8,
}

impl RawMnistDataItems {
    pub const IMAGE_HEADER_LEN: usize = 16;
    pub const LABEL_HEADER_LEN: usize = 8;

    pub fn new(image_archive: &Path, label_archive: &Path) -> Result<Self> {
        let images = read_gzip(image_archive)
            .context(format!("Loading image archive {}", image_archive.display()))?;
        let (im_count, rows, cols) = process_raw_image_data(&images)
            .context(format!("Processing image data {}", image_archive.display()))?;

        let labels = read_gzip(label_archive)
            .context(format!("Loading label archive {}", label_archive.display()))?;
        let lb_count = process_raw_label_data(&labels)
            .context(format!("Processing label data{}", label_archive.display()))?;

        ensure!(im_count == lb_count, "Image and label count mismatch");

        Ok(Self {
            image_archive: image_archive.display().to_string(),
            label_archive: label_archive.display().to_string(),
            count: im_count,
            rows,
            cols,
            images,
            labels,
        })
    }
}

impl Dataset<RawMnistDataItem> for RawMnistDataItems {
    fn get(&self, index: usize) -> Option<RawMnistDataItem> {
        if index >= self.count {
            return None;
        }

        let pixel_start = RawMnistDataItems::IMAGE_HEADER_LEN + index * self.rows * self.cols;
        let pixel_end = pixel_start + self.rows * self.cols;

        Some(RawMnistDataItem {
            pixels: pixel_start..pixel_end,
            idx: index,
            label: self.labels[RawMnistDataItems::LABEL_HEADER_LEN + index],
        })
    }

    fn len(&self) -> usize {
        self.count
    }
}

impl std::fmt::Display for RawMnistDataItems {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "RawMnistDataItems:")?;
        writeln!(f, "image_archive: {}", self.image_archive)?;
        writeln!(f, "label_archive: {}", self.label_archive)?;
        writeln!(f, "count: {}", self.count)?;
        writeln!(f, "rows: {}", self.rows)?;
        writeln!(f, "cols: {}", self.cols)?;
        writeln!(f, "pixels (size): {}", self.images.len())?;
        writeln!(f, "labels (size): {}", self.labels.len())
    }
}

fn read_gzip(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path).context("Opening a gzip file")?;
    let mut decoder = GzDecoder::new(file);
    let mut buffer = Vec::new();

    decoder
        .read_to_end(&mut buffer)
        .context("Decoding gzip file")?;

    Ok(buffer)
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

fn process_raw_image_data(data: &[u8]) -> Result<(usize, usize, usize)> {
    ensure!(
        data.len() > RawMnistDataItems::IMAGE_HEADER_LEN,
        "Image data should contain more than the header"
    );

    let magic = u32::from_be_bytes(data[0..4].try_into().unwrap());
    if magic != 2051 {
        bail!("Invalid image magic number: expected 2051, got {magic}");
    }

    let count = usize::try_from(u32::from_be_bytes(data[4..8].try_into().unwrap()))?;
    if count == 0 {
        bail!("Image count must be greater than zero");
    }

    let rows = usize::try_from(u32::from_be_bytes(data[8..12].try_into().unwrap()))?;
    let cols = usize::try_from(u32::from_be_bytes(data[12..16].try_into().unwrap()))?;

    ensure!(
        data.len() == RawMnistDataItems::IMAGE_HEADER_LEN + (count * rows * cols),
        "Image data not consistent"
    );

    Ok((count, rows, cols))
}

fn process_raw_label_data(data: &[u8]) -> Result<usize> {
    ensure!(
        data.len() > RawMnistDataItems::LABEL_HEADER_LEN,
        "Label data should contain more than the header"
    );

    let magic = u32::from_be_bytes(data[0..4].try_into().unwrap());
    if magic != 2049 {
        bail!("Invalid label magic number: expected 2049, got {magic}");
    }

    let count = usize::try_from(u32::from_be_bytes(data[4..8].try_into().unwrap()))?;
    if count == 0 {
        bail!("Label count must be greater than zero");
    }

    ensure!(
        data.len() == RawMnistDataItems::LABEL_HEADER_LEN + count,
        "Label data not consistent"
    );

    Ok(count)
}
