use anyhow::Result;
use core::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
pub struct MnistImages {
    pub count: usize,
    pub rows: usize,
    pub cols: usize,
    pub pixels: Vec<u8>,
}

impl fmt::Display for MnistImages {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MinstImages:")?;
        writeln!(f, "count: {}", self.count)?;
        writeln!(f, "rows: {}", self.rows)?;
        writeln!(f, "cols: {}", self.cols)?;
        write!(f, "pixels (count): {}", self.pixels.len())
    }
}

pub fn load_raw_image_data(path: &Path) -> Result<MnistImages> {
    let mut file = File::open(path).expect("Error opening raw image file");

    let mut header = [0u8; 16];
    file.read_exact(&mut header)
        .expect("Error reading image file header");

    let magic = u32::from_be_bytes(header[0..4].try_into().unwrap());
    assert_eq!(magic, 2051);

    let count = u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_be_bytes(header[8..12].try_into().unwrap()) as usize;
    let cols = u32::from_be_bytes(header[12..16].try_into().unwrap()) as usize;

    assert_eq!(rows, 28);
    assert_eq!(cols, 28);

    let mut pixels = Vec::new();
    file.read_to_end(&mut pixels)?;

    assert_eq!(pixels.len(), rows * cols * count);

    println!("Header: {header:?}");
    println!("Payload length: {}", pixels.len());

    Ok(MnistImages {
        count,
        rows,
        cols,
        pixels,
    })
}
