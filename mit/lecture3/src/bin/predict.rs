use lecture3::data::*;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    let images = load_raw_image_data(&data_path.join("t10k-images-idx3-ubyte"))?;
    println!("{images}");

    let labels = load_raw_label_data(&data_path.join("t10k-labels-idx1-ubyte"))?;
    println!("{labels}");
    anyhow::ensure!(
        images.count == labels.count,
        "Image and label counts differ"
    );

    Ok(())
}
