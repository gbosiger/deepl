use crate::{batch::HeartBatch, data::*, model::ModelConfig};
use burn::{
    nn::loss::BinaryCrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::backend::BackendTypes,
};
use std::path::Path;

type Backend = burn::backend::Autodiff<burn::backend::Flex>;
type FloatElem = <Backend as BackendTypes>::FloatElem;
type IntElem = <Backend as BackendTypes>::IntElem;

pub fn run() -> anyhow::Result<()> {
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

    // Encode each split into flat buffers, use types from the Backend (for direct Tensor creation
    // later)
    let encoded_training_data =
        encode_samples::<FloatElem, IntElem>(training_samples, training_samples_normalized);
    let encoded_validation_data =
        encode_samples::<FloatElem, IntElem>(validation_samples, validation_samples_normalized);
    let encoded_testing_data =
        encode_samples::<FloatElem, IntElem>(testing_samples, testing_samples_normalized);

    // Print sample sizes - hopefully they match the ones above
    println!("Training samples: {}", encoded_training_data.targets.len());
    println!(
        "Validation samples: {}",
        encoded_validation_data.targets.len()
    );
    println!("Test samples: {}", encoded_testing_data.targets.len());

    // We need to store normalization values to be used with the model later
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("generated")
        .join("normalization_manual.csv");
    write_normalization_params(&path, &normalization)
        .expect("Storing of normalization params failed");

    // Create device, model and an optimizer
    let device = Default::default();
    let mut model = ModelConfig::new(29, 16).init::<Backend>(&device);
    // This optimizer remembers previous steps, this is why it is created outside the training loop
    let mut optimizer = AdamConfig::new().init::<Backend, crate::model::Model<Backend>>();

    // We calculate by exposing logits as precision is better than after sigmoid
    let loss_function = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init::<Backend>(&device);

    let batch = HeartBatch::<Backend>::from_encoded(encoded_training_data, &device)?;

    let logits = model.logits(batch.inputs.clone());
    let loss = loss_function.forward(logits, batch.targets.clone());

    println!("Before: {}", loss.clone().into_scalar());

    let gradients = GradientsParams::from_grads(loss.backward(), &model);
    model = optimizer.step(0.001, model, gradients);

    let logits = model.logits(batch.inputs);
    let loss_after = loss_function.forward(logits, batch.targets);

    println!("After: {}", loss_after.into_scalar());

    Ok(())
}
