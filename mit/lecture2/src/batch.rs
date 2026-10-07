//! Bridge between prepared data and Burn tensors.
use crate::data::EncodedData;
use anyhow::{Result, ensure};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

#[derive(Debug)]
pub struct HeartBatch<B: Backend> {
    /// Shape: [batch size, 29].
    pub inputs: Tensor<B, 2>,
    /// Shape: [batch size, 1]; integer labels for binary cross-entropy.
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> HeartBatch<B> {
    pub fn from_samples(samples: &[EncodedData], device: &B::Device) -> Result<Self> {
        ensure!(!samples.is_empty(), "Cannot create an empty batch");
        let mut inputs = Vec::with_capacity(samples.len() * 29);
        let mut targets = Vec::with_capacity(samples.len());
        for sample in samples {
            // Preprocessing uses f64; model inputs use f32.
            for value in sample.input {
                let value = value as f32;
                ensure!(value.is_finite(), "Batch inputs must be finite f32 values");
                inputs.push(value);
            }
            ensure!(
                sample.target == 0.0 || sample.target == 1.0,
                "Targets must be 0 or 1"
            );
            targets.push(sample.target as i32);
        }
        Ok(Self {
            inputs: Tensor::from_data(TensorData::new(inputs, [samples.len(), 29]), device),
            targets: Tensor::from_data(TensorData::new(targets, [samples.len(), 1]), device),
        })
    }
}
