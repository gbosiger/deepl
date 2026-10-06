use lecture2::data::read_heart_data;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // TODO: Decide how this program accepts dataset paths and training settings.
    // TODO: Call the training workflow you build in the shared library.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("heart.csv");

    let data = read_heart_data(&path)?;
    println!("Loaded {} rows", data.len());

    for row in data.iter().take(3) {
        println!("{row:?}");
    }
    Ok(())
}
