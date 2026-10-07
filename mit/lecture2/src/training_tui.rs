use crate::{batch::HeartBatch, data::EncodedData};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::backend::Backend,
};
use std::sync::Arc;

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
    println!("TUI training setup");
    Ok(())
}
