use burn::{data::dataloader::DataLoaderBuilder, tensor::backend::BackendTypes};
use lecture2::{
    common::{data::write_normalization_params, prep::prepare_data},
    trainer::batcher::HeartBatcher,
};
use std::{path::Path, sync::Arc};

type InferenceBackend = burn::backend::Flex;
type TrainingBackend = burn::backend::Autodiff<InferenceBackend>;
type FloatElem = <TrainingBackend as BackendTypes>::FloatElem;
type IntElem = <TrainingBackend as BackendTypes>::IntElem;

fn main() -> anyhow::Result<()> {
    let main_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let prepared = prepare_data::<FloatElem, IntElem>(&main_path.join("data/heart.csv"))?;
    let output = main_path.join("generated/tui");
    std::fs::create_dir_all(&output)?;
    write_normalization_params(&output.join("normalization.csv"), &prepared.normalization)?;

    let training = Arc::new(prepared.training);
    let validation = Arc::new(prepared.validation);
    // Keep prepared.testing for final evaluation when the trainer is connected.

    // Share the same buffers between each dataset and its batcher.
    let training_loader =
        DataLoaderBuilder::new(HeartBatcher::<TrainingBackend>::new(Arc::clone(&training)))
            .batch_size(32)
            .shuffle(42)
            .num_workers(0)
            .build(training);

    let validation_loader = DataLoaderBuilder::new(HeartBatcher::<InferenceBackend>::new(
        Arc::clone(&validation),
    ))
    .batch_size(32)
    .num_workers(0)
    .build(validation);

    // Check the loader output before connecting the trainer.
    for batch in training_loader.iter() {
        println!("Training batch: {:?}", batch.inputs.dims());
    }
    for batch in validation_loader.iter() {
        println!("Validation batch: {:?}", batch.inputs.dims());
    }
    Ok(())
}
