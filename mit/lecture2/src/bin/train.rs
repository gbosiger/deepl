use lecture2::data::*;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // TODO: Decide how this program accepts dataset paths and training settings.
    // TODO: Call the training workflow you build in the shared library.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("heart.csv");

    // Load data
    let data = read_heart_data(&path)?;
    println!("Loaded {} rows", data.len());

    // Print couple of entries
    for row in data.iter().take(3) {
        println!("{row:?}");
    }

    // Split training and test data
    let (training_samples, testing_samples) = shuffle_and_split_heart_data(&data);

    // Print sample sizes
    println!("Training samples: {}", training_samples.len());
    println!("Test samples: {}", testing_samples.len());

    let normalization = fit_normalization(&training_samples);
    let training_samples_normalized = normalize_heart_data(&training_samples, &normalization);
    let testing_samples_normalized = normalize_heart_data(&testing_samples, &normalization);

    // Print couple of normalized training entries
    for row in training_samples_normalized.iter().take(3) {
        println!("{row:?}");
    }

    Ok(())
}
