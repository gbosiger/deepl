use lecture3::dataset::{RawMnistDataItem, RawMnistDataItems};
use lecture3::pgm::SavePgmFile;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    let raw_items = RawMnistDataItems::new(
        &data_path.join("train-images-idx3-ubyte.gz"),
        &data_path.join("train-labels-idx1-ubyte.gz"),
    )?;

    println!("{raw_items}");

    Ok(())
}
