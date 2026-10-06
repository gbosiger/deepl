use lecture2::data::*;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // TODO: Decide how this program accepts dataset paths and training settings.
    // TODO: Call the training workflow you build in the shared library.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("heart.csv");

    // Load data
    let raw_data = read_heart_data(&path)?;
    println!("Loaded {} rows", raw_data.len());

    // Print couple of entries
    for row in raw_data.iter().take(3) {
        println!("{row:?}");
    }

    // Split training and test data
    let (mut training_samples, testing_samples) = shuffle_and_split_heart_data(&raw_data);
    // Further split training samples for validation
    let validation_samples = training_samples.split_off(training_samples.len() * 80 / 100);

    // Calculate mean and std deviation for the input parameters
    let normalization = fit_normalization(&training_samples);

    let training_samples_normalized = normalize_heart_data(&training_samples, &normalization);
    let validation_samples_normalized = normalize_heart_data(&validation_samples, &normalization);
    let testing_samples_normalized = normalize_heart_data(&testing_samples, &normalization);

    // Print couple of normalized training entries
    for row in training_samples_normalized.iter().take(3) {
        println!("{row:?}");
    }

    // Print sample sizes
    println!("Training samples: {}", training_samples_normalized.len());
    println!(
        "Validation samples: {}",
        validation_samples_normalized.len()
    );
    println!("Test samples: {}", testing_samples_normalized.len());

    let encoded_training_data: Vec<EncodedData> = training_samples
        .into_iter()
        .zip(training_samples_normalized.iter())
        .map(|(raw, norm)| encode_data(raw, norm))
        .collect();

    let encoded_validation_data: Vec<EncodedData> = validation_samples
        .into_iter()
        .zip(validation_samples_normalized.iter())
        .map(|(raw, norm)| encode_data(raw, norm))
        .collect();

    let encoded_testing_data: Vec<EncodedData> = testing_samples
        .into_iter()
        .zip(testing_samples_normalized.iter())
        .map(|(raw, norm)| encode_data(raw, norm))
        .collect();

    // Print sample sizes - hopefully they match the ones above
    println!("Training samples: {}", encoded_training_data.len());
    println!("Validation samples: {}", encoded_validation_data.len());
    println!("Test samples: {}", encoded_testing_data.len());

    // We need to store normalization values
    let path = Path::new("normalization.csv");
    write_normalization_params(path, &normalization)
        .expect("Storing of normalization params failed");

    Ok(())
}
