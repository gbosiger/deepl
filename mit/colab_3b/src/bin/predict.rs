use colab_3b::dataset::RawMnistDataItems;
use colab_3b::pgm::SavePgmImage;
use std::path::Path;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");

    let raw_items = Arc::new(RawMnistDataItems::new(
        &data_path.join("train-images-idx3-ubyte.gz"),
        &data_path.join("train-labels-idx1-ubyte.gz"),
    )?);

    // Save some images to disk for inspection
    for img in [30, 245, 3534, 5634] {
        raw_items.save_file(img, &data_path.join("artifacts"))?;
    }

    println!("{raw_items}");

    Ok(())
}
