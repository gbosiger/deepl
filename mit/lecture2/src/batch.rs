//! Bridge between prepared data and Burn tensors.
use crate::data::EncodedData;
use anyhow::{Result, ensure};
use burn::tensor::{ElementConversion, Int, Tensor, TensorData, backend::Backend};

#[derive(Debug, Clone)]
pub struct HeartBatch<B: Backend> {
    /// Shape: [batch size, 29].
    pub inputs: Tensor<B, 2>,
    /// Shape: [batch size, 1]; integer labels for binary cross-entropy.
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> HeartBatch<B> {
    /// Slice patient rows into mini-batches; the last batch may be smaller.
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = Self> + '_ {
        assert!(batch_size > 0);
        let [count, features] = self.inputs.dims();
        assert_eq!(self.targets.dims(), [count, 1]);

        (0..count).step_by(batch_size).map(move |start| {
            let end = start.saturating_add(batch_size).min(count);
            Self {
                inputs: self.inputs.clone().slice([start..end, 0..features]),
                targets: self.targets.clone().slice([start..end, 0..1]),
            }
        })
    }

    /// Consume flat buffers to avoid cloning or flattening them again.
    pub fn from_encoded(
        data: EncodedData<B::FloatElem, B::IntElem>,
        device: &B::Device,
    ) -> Result<Self> {
        let count = data.targets.len();
        ensure!(count > 0, "Cannot create an empty batch");
        ensure!(
            data.inputs.len() == count * 29,
            "Each sample must have 29 inputs"
        );
        ensure!(
            data.inputs.iter().all(|x| x.elem::<f64>().is_finite()),
            "Inputs must be finite"
        );
        ensure!(
            data.targets
                .iter()
                .all(|x| matches!(x.elem::<f64>(), 0.0 | 1.0)),
            "Targets must be 0 or 1"
        );
        Ok(Self {
            inputs: Tensor::from_data(TensorData::new(data.inputs, [count, 29]), device),
            targets: Tensor::from_data(TensorData::new(data.targets, [count, 1]), device),
        })
    }
}
