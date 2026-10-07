use burn::{
    module::{AutodiffModule, Module},
    nn::loss::BinaryCrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::DefaultRecorder,
    tensor::backend::BackendTypes,
};
use lecture2::{
    batch::HeartBatch,
    data::*,
    model::ModelConfig,
    prep::{PreparedData, prepare_data},
};
use std::path::Path;

type InferenceBackend = burn::backend::Flex;
// When training we need to have gradient tracking to build the graph, needed for backpropagation
type TrainingBackend = burn::backend::Autodiff<InferenceBackend>;
type FloatElem = <TrainingBackend as BackendTypes>::FloatElem;
type IntElem = <TrainingBackend as BackendTypes>::IntElem;

fn main() -> anyhow::Result<()> {
    let main_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let PreparedData {
        training: encoded_training_data,
        validation: encoded_validation_data,
        testing: encoded_testing_data,
        normalization,
    } = prepare_data::<FloatElem, IntElem>(&main_path.join("data/heart.csv"))?;

    // Keep manual files separate from TUI files.
    let path_to_generated = main_path.join("generated/manual");
    std::fs::create_dir_all(&path_to_generated)?;
    // Prediction must use these same training means and deviations.
    write_normalization_params(&path_to_generated.join("normalization.csv"), &normalization)?;

    // Create device, model and an optimizer for the training
    let device = Default::default();
    let mut model = ModelConfig::new(29, 16).init::<TrainingBackend>(&device);
    // This optimizer remembers previous steps, this is why it is created outside the training loop
    let mut optimizer =
        AdamConfig::new().init::<TrainingBackend, lecture2::model::Model<TrainingBackend>>();

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

    // Train and calculate validation loss
    for epoch in 0..20 {
        for (i, batch) in training_batch.batches(32).enumerate() {
            // Prepare input and calculate the loss based on the training samples
            let logits = model.logits(batch.inputs);
            let training_loss = loss_function.forward(logits, batch.targets);

            let loss_value = training_loss.clone().into_scalar();
            println!(
                "Epoch: {}, batch: {}: training loss {}",
                epoch + 1,
                i + 1,
                loss_value
            );

            // Calculate gradients
            let gradients = GradientsParams::from_grads(training_loss.backward(), &model);

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

    // Now we use the third, testing, batch and take the trained model and calculate the loss
    let testing_batch =
        HeartBatch::<InferenceBackend>::from_encoded(encoded_testing_data, &device)?;
    let testing_model = model.valid();
    let logits = testing_model.logits(testing_batch.inputs);
    let test_loss = validation_loss_function.forward(logits, testing_batch.targets);

    println!("Final test loss: {}", test_loss.into_scalar());

    // Finally we save the trained weights
    let model_path = path_to_generated.clone().join("model");

    model
        .valid()
        .save_file(model_path, &DefaultRecorder::new())?;

    Ok(())
}
