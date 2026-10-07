use burn::{module::Module, tensor::Tensor};
use lecture2::data::*;
use lecture2::model::ModelConfig;
use std::path::Path;

type Backend = burn::backend::Flex;

fn main() -> anyhow::Result<()> {
    // Load data
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("heart.csv");

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

    // Encode data
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

    // We need to store normalization values to be used with the model later
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("generated")
        .join("normalization.csv");
    write_normalization_params(&path, &normalization)
        .expect("Storing of normalization params failed");

    // Create and validate model
    let device = Default::default();
    let model_config = ModelConfig::new(29, 16);
    let model = model_config.init::<Backend>(&device);

    let input = Tensor::<Backend, 2>::zeros([2, 29], &device);
    let output = model.forward(input);

    assert_eq!(output.dims(), [2, 1]);
    assert_eq!(model.num_params(), 497);

    // The model has random weights, so these probabilities aren't useful predictions yet.
    let probabilities = output.into_data().to_vec::<f32>()?;
    assert!(probabilities.iter().all(|p| (0.0..=1.0).contains(p)));

    // Print probabilities from random weights
    println!("Probabilities: {:?}", probabilities);

    Ok(())
}
