use crate::{batch::HeartBatch, data::*};
use burn::{
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    tensor::backend::{Backend, BackendTypes},
};
use std::{path::Path, sync::Arc};

type InferenceBackend = burn::backend::Flex;
type TrainingBackend = burn::backend::Autodiff<InferenceBackend>;
type FloatElem = <TrainingBackend as BackendTypes>::FloatElem;
type IntElem = <TrainingBackend as BackendTypes>::IntElem;

impl<F: Send + Sync, I: Send + Sync> Dataset<usize> for EncodedData<F, I> {
    fn get(&self, index: usize) -> Option<usize> {
        if index < self.targets.len() {
            Some(index)
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.targets.len()
    }
}

#[derive(Clone)]
struct HeartBatcher<B: Backend> {
    data: Arc<EncodedData<B::FloatElem, B::IntElem>>,
}

impl<B: Backend> Batcher<B, usize, HeartBatch<B>> for HeartBatcher<B> {
    fn batch(&self, indices: Vec<usize>, device: &B::Device) -> HeartBatch<B> {
        let mut encoded = EncodedData {
            inputs: Vec::with_capacity(indices.len() * 29),
            targets: Vec::with_capacity(indices.len()),
        };
        for i in indices {
            encoded
                .inputs
                .extend_from_slice(&self.data.inputs[i * 29..(i + 1) * 29]);
            encoded.targets.push(self.data.targets[i]);
        }
        HeartBatch::from_encoded(encoded, device)
            .expect("Batch must contain valid encoded patients")
    }
}

pub fn run() -> anyhow::Result<()> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("heart.csv");
    let raw = read_heart_data(&path)?;
    let (mut training, _testing) = shuffle_and_split_heart_data(&raw);
    let validation = training.split_off(training.len() * 80 / 100);

    let normalization = fit_normalization(&training);
    let training_normalized = normalize_heart_data(&training, &normalization);
    let validation_normalized = normalize_heart_data(&validation, &normalization);
    let training = Arc::new(encode_samples::<FloatElem, IntElem>(
        training,
        training_normalized,
    ));
    let validation = Arc::new(encode_samples::<FloatElem, IntElem>(
        validation,
        validation_normalized,
    ));

    // Share the same buffers between each dataset and its batcher.
    let training_loader = DataLoaderBuilder::new(HeartBatcher::<TrainingBackend> {
        data: Arc::clone(&training),
    })
    .batch_size(32)
    .shuffle(42)
    .num_workers(0)
    .build(training);

    let validation_loader = DataLoaderBuilder::new(HeartBatcher::<InferenceBackend> {
        data: Arc::clone(&validation),
    })
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
