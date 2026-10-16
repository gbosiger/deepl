use anyhow::{bail, Context, Result};
use core::fmt;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;

#[derive(Debug)]
pub struct RawMnistImages {
    pub path: String,
    pub count: usize,
    pub rows: usize,
    pub cols: usize,
    pub pixels: Vec<u8>,
}

impl fmt::Display for RawMnistImages {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MnistImages:")?;
        writeln!(f, "path: {}", self.path)?;
        writeln!(f, "count: {}", self.count)?;
        writeln!(f, "rows: {}", self.rows)?;
        writeln!(f, "cols: {}", self.cols)?;
        writeln!(f, "pixels (count): {}", self.pixels.len())
    }
}

pub fn load_raw_image_data(path: &Path) -> Result<RawMnistImages> {
    let mut file =
        File::open(path).with_context(|| format!("Opening image file {}", path.display()))?;

    let mut header = [0u8; 16];
    file.read_exact(&mut header)
        .with_context(|| format!("Reading image header from {}", path.display()))?;

    let magic = u32::from_be_bytes(header[0..4].try_into().unwrap());
    if magic != 2051 {
        bail!("Invalid image magic number: expected 2051, got {magic}");
    }

    let count = usize::try_from(u32::from_be_bytes(header[4..8].try_into().unwrap()))?;
    if count == 0 {
        bail!("Image count must be greater than zero");
    }

    let rows = usize::try_from(u32::from_be_bytes(header[8..12].try_into().unwrap()))?;
    let cols = usize::try_from(u32::from_be_bytes(header[12..16].try_into().unwrap()))?;
    if rows != 28 || cols != 28 {
        bail!("Expected 28 × 28 images, got {rows} × {cols}");
    }

    let expected_len = rows
        .checked_mul(cols)
        .and_then(|grid| grid.checked_mul(count))
        .context("Image payload size overflows usize")?;

    let mut pixels = Vec::new();
    file.read_to_end(&mut pixels)
        .with_context(|| format!("Reading image pixels from {}", path.display()))?;
    if pixels.len() != expected_len {
        bail!(
            "Invalid image payload length: expected {expected_len} bytes, got {}",
            pixels.len()
        );
    }

    Ok(RawMnistImages {
        path: path.display().to_string(),
        count,
        rows,
        cols,
        pixels,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct RawMnistLabelItem(u8);

impl RawMnistLabelItem {
    pub fn id(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for RawMnistLabelItem {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        if value > 9 {
            bail!("Invalid Fashion-MNIST label: {value}");
        }
        Ok(Self(value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawMnistLabels {
    pub path: String,
    pub count: usize,
    labels: Vec<u8>,
}

impl RawMnistLabels {
    pub fn label(&self, index: usize) -> Option<RawMnistLabelItem> {
        self.labels.get(index).copied().map(RawMnistLabelItem)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = RawMnistLabelItem> + '_ {
        self.labels.iter().copied().map(RawMnistLabelItem)
    }
}

impl fmt::Display for RawMnistLabels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MnistLabels:")?;
        writeln!(f, "path: {}", self.path)?;
        writeln!(f, "labels (count): {}", self.count)
    }
}

pub fn load_raw_label_data(path: &Path) -> Result<RawMnistLabels> {
    let mut file =
        File::open(path).with_context(|| format!("Opening label file {}", path.display()))?;

    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .with_context(|| format!("Reading labels header from {}", path.display()))?;

    let magic = u32::from_be_bytes(header[0..4].try_into().unwrap());
    if magic != 2049 {
        bail!("Invalid label magic number: expected 2049, got {magic}");
    }

    let count = usize::try_from(u32::from_be_bytes(header[4..8].try_into().unwrap()))?;
    if count == 0 {
        bail!("Label count must be greater than zero");
    }
    let mut labels = Vec::new();
    file.read_to_end(&mut labels)
        .with_context(|| format!("Reading labels from {}", path.display()))?;
    if labels.len() != count {
        bail!(
            "Invalid labels payload length: expected {count} bytes, got {}",
            labels.len()
        );
    }

    for (index, &value) in labels.iter().enumerate() {
        RawMnistLabelItem::try_from(value)
            .with_context(|| format!("Invalid label at index {index} in {}", path.display()))?;
    }

    Ok(RawMnistLabels {
        path: path.display().to_string(),
        count,
        labels,
    })
}

pub fn label_code_to_str(code: u8) -> Result<&'static str> {
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

// PGM supports grayscale pixels directly an no image library is needed
pub fn export_pgm_image(
    path: &Path,
    idx: usize,
    images: &RawMnistImages,
    labels: &RawMnistLabels,
) -> Result<String> {
    anyhow::ensure!(images.count == labels.count);
    anyhow::ensure!(idx < images.count, "Index invalid during image export");

    let label = label_code_to_str(labels.labels[idx]).unwrap();
    let filename = format!("{}_{}.pgm", idx, label);

    let mut file = File::create(path.join(&filename))
        .with_context(|| format!("Creating pgm export file in {}", path.display()))?;
    write!(file, "P5\n{} {}\n255\n", images.cols, images.rows)?;

    let image_size = images.rows * images.cols;
    let start = idx * image_size;
    file.write_all(&images.pixels[start..start + image_size])?;

    Ok(filename)
}
