use crate::{batch::HeartBatch, data::*, model::ModelConfig};
use burn::{
    module::AutodiffModule,
    nn::loss::BinaryCrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::backend::BackendTypes,
};
use std::path::Path;

type InferenceBackend = burn::backend::Flex;
// When training we need to have gradient tracking to build the graph, needed for backpropagation
type TrainingBackend = burn::backend::Autodiff<InferenceBackend>;
type FloatElem = <TrainingBackend as BackendTypes>::FloatElem;
type IntElem = <TrainingBackend as BackendTypes>::IntElem;

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

    // Create device, model and an optimizer for the training
    let device = Default::default();
    let mut model = ModelConfig::new(29, 16).init::<TrainingBackend>(&device);
    // This optimizer remembers previous steps, this is why it is created outside the training loop
    let mut optimizer =
        AdamConfig::new().init::<TrainingBackend, crate::model::Model<TrainingBackend>>();

    // Prepare training batch and loss function
    let training_batch =
        HeartBatch::<TrainingBackend>::from_encoded(encoded_training_data, &device)?;
    // We calculate by exposing logits as precision is better than after sigmoid
    let loss_function = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init::<TrainingBackend>(&device);

    // Prepare validation batch and loss function
    let validation_batch =
        HeartBatch::<InferenceBackend>::from_encoded(encoded_validation_data, &device)?;
    let validation_loss_function = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init::<InferenceBackend>(&device);

    for epoch in 0..20 {
        for (i, batch) in training_batch.batches(32).enumerate() {
            // Prepare input and calculate the loss based on the training samples
            let logits = model.logits(batch.inputs);
            let loss = loss_function.forward(logits, batch.targets);

            let loss_value = loss.clone().into_scalar();
            println!(
                "Epoch: {}, batch: {}: training loss {}",
                epoch + 1,
                i + 1,
                loss_value
            );

            // Calculate gradients
            let gradients = GradientsParams::from_grads(loss.backward(), &model);

            // Update the model based on calculated gradients
            model = optimizer.step(0.001, model, gradients);
        }

        // Now we use the updated model to check what we get with validation batch, but we do not
        // update the model
        // This allows us to compare how loss changes between the training and validation batches
        // If training and validation loss both fall, great, if training loss falls while
        // validation loss rises, we could be overfitting
        // valid() gives you the current model without gradient tracking
        let validation_model = model.valid();

        // Prepare input and calculate the loss based on the training samples
        let logits = validation_model.logits(validation_batch.inputs.clone());
        let loss = validation_loss_function.forward(logits, validation_batch.targets.clone());

        let loss_value = loss.clone().into_scalar();
        println!("Epoch: {}: validation loss {}", epoch + 1, loss_value);
    }

    Ok(())
}
