//! Shared CSV preparation for both training approaches; no Burn types here.
use crate::common::data::*;
use num_traits::FromPrimitive;
use std::path::Path;

pub struct PreparedData<F, I> {
    pub training: EncodedData<F, I>,
    pub validation: EncodedData<F, I>,
    pub testing: EncodedData<F, I>,
    pub normalization: Normalization,
}

pub fn prepare_data<F, I>(path: &Path) -> anyhow::Result<PreparedData<F, I>>
where
    F: FromPrimitive + From<u8>,
    I: From<u8>,
{
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

    // Normalize samples
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

    // Encode each split into flat buffers, use types from the Backend (for direct Tensor creation
    // later)
    let encoded_training_data =
        encode_samples::<F, I>(training_samples, training_samples_normalized);
    let encoded_validation_data =
        encode_samples::<F, I>(validation_samples, validation_samples_normalized);
    let encoded_testing_data = encode_samples::<F, I>(testing_samples, testing_samples_normalized);

    // Print sample sizes - hopefully they match the ones above
    println!("Training samples: {}", encoded_training_data.targets.len());
    println!(
        "Validation samples: {}",
        encoded_validation_data.targets.len()
    );
    println!("Test samples: {}", encoded_testing_data.targets.len());

    Ok(PreparedData {
        training: encoded_training_data,
        validation: encoded_validation_data,
        testing: encoded_testing_data,
        normalization,
    })
}
