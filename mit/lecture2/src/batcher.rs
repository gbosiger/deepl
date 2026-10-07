// Here we implement a batcher as defined by Burn
use crate::{batch::HeartBatch, data::EncodedData};
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use std::sync::Arc;

#[derive(Clone)]
pub struct HeartBatcher<B: Backend> {
    data: Arc<EncodedData<B::FloatElem, B::IntElem>>,
}

impl<B: Backend> HeartBatcher<B> {
    pub fn new(data: std::sync::Arc<EncodedData<B::FloatElem, B::IntElem>>) -> Self {
        Self { data }
    }
}

impl<B: Backend> Batcher<B, usize, HeartBatch<B>> for HeartBatcher<B> {
    fn batch(&self, indices: Vec<usize>, device: &B::Device) -> HeartBatch<B> {
        let mut encoded = EncodedData {
            inputs: Vec::with_capacity(indices.len() * 29),
            targets: Vec::with_capacity(indices.len()),
        };
        // Burn can shuffle indices so we extend_from_slice per 29 items (one input)
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
