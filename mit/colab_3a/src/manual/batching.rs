// Our manual batching approach: slice tensors that are already on the device.
use crate::common::batch::HeartBatch;
use burn::tensor::backend::Backend;

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
}
