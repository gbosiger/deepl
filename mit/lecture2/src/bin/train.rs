use burn::{nn::loss::BinaryCrossEntropyLossConfig, tensor::backend::BackendTypes};
use lecture2::model::ModelConfig;
use lecture2::{batch::HeartBatch, data::*};
use std::path::Path;

type Backend = burn::backend::Flex;
type FloatElem = <Backend as BackendTypes>::FloatElem;
type IntElem = <Backend as BackendTypes>::IntElem;

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
        .join("normalization.csv");
    write_normalization_params(&path, &normalization)
        .expect("Storing of normalization params failed");

    // Create and validate model
    let device = Default::default();
    let model_config = ModelConfig::new(29, 16);
    let model = model_config.init::<Backend>(&device);

    // Prepare the training data batch and a loss function config
    let batch = HeartBatch::<Backend>::from_encoded(encoded_training_data, &device)?;

    // We calculate by exposing logits as precision is better than after sigmoid
    let loss_function = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init::<Backend>(&device);
    let logits = model.logits(batch.inputs);
    let loss = loss_function.forward(logits, batch.targets);

    println!("Loss: {}", loss.into_scalar());

    /*
    Notes: what training looks like by hand.
    If I try this, replace the model/batch/loss code above. The encoded data
    needs to still be available, since creating a batch consumes its buffers.

    Repeat: predict -> calculate loss -> backward -> update weights.
    Burn's trainer can handle the loop, progress display, and checkpoints later.

    use burn::optim::{AdamConfig, GradientsParams, Optimizer};
    use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

    type TrainingBackend = burn::backend::Autodiff<Backend>;
    let device = Default::default();
    let mut model = ModelConfig::new(29, 16).init::<TrainingBackend>(&device);
    let mut optimizer = AdamConfig::new().init();
    let loss_function = BinaryCrossEntropyLossConfig::new()
        .with_logits(true)
        .init::<TrainingBackend>(&device);

    let mut indices: Vec<usize> = (0..encoded_training_data.targets.len()).collect();
    let mut rng = StdRng::seed_from_u64(42);
    let batch_size = 32;
    let learning_rate = 0.001;

    for epoch in 0..300 {
        // One epoch = go through all training patients. Shuffle again each time.
        indices.shuffle(&mut rng);
        let mut total_loss = 0.0;

        for rows in indices.chunks(batch_size) {
            // Pick the rows for this batch. Each patient has 29 values in the flat buffer.
            let mut encoded = EncodedData {
                inputs: Vec::with_capacity(rows.len() * 29),
                targets: Vec::with_capacity(rows.len()),
            };
            for &i in rows {
                encoded.inputs.extend_from_slice(
                    &encoded_training_data.inputs[i * 29..(i + 1) * 29],
                );
                encoded.targets.push(encoded_training_data.targets[i]);
            }
            let batch = HeartBatch::<TrainingBackend>::from_encoded(encoded, &device)?;

            // Get logits, then compare with the actual answers.
            let logits = model.logits(batch.inputs);
            let loss = loss_function.forward(logits, batch.targets);
            // Clone here: printing the scalar consumes it, but I still need backward.
            total_loss += f64::from(loss.clone().into_scalar()) * rows.len() as f64;

            // Find how the weights affect the loss.
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            // Adam adjusts the weights and gives back the updated model.
            model = optimizer.step(learning_rate, model, gradients);
        }

        // Average over patients, not batches: the last batch is smaller.
        println!("Epoch {}: training loss {}", epoch + 1,
            total_loss / indices.len() as f64);
        // Check validation here, but do not update weights.
        // Once training is done: check test data and save the model.
    }
    */

    Ok(())
}
