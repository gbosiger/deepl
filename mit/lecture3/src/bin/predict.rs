use lecture3::data::*;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    let images = load_raw_image_data(&data_path.clone().join("t10k-images-idx3-ubyte"))?;
    println!("{images}");

    Ok(())
}
