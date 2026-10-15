use std::path::Path;

use lecture3::data::{export_pgm_image, load_raw_image_data, load_raw_label_data};

fn main() -> anyhow::Result<()> {
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    let images = load_raw_image_data(&data_path.join("train-images-idx3-ubyte"))?;
    println!("{images}");

    let labels = load_raw_label_data(&data_path.join("train-labels-idx1-ubyte"))?;
    println!("{labels}");
    anyhow::ensure!(
        images.count == labels.count,
        "Image and label counts differ"
    );

    // Save couple of images with labels in filename for a test
    for idx in [4, 56, 680, 4343] {
        let filename = export_pgm_image(&data_path, idx, &images, &labels)?;
        println!("Saved image {filename}");
    }

    Ok(())
}
