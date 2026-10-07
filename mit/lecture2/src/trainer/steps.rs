use crate::common::{batch::HeartBatch, model::Model};
use burn::{
    nn::loss::BinaryCrossEntropyLossConfig,
    tensor::backend::{AutodiffBackend, Backend},
    train::{InferenceStep, MultiLabelClassificationOutput, TrainOutput, TrainStep},
};

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = HeartBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let logits = self.logits(batch.inputs);

        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init::<B>(&logits.device())
            .forward(logits.clone(), batch.targets.clone());

        let gradients = loss.clone().backward();

        // Report probabilities and loss alongside the gradients.
        let probabilities = burn::tensor::activation::sigmoid(logits);
        let item = MultiLabelClassificationOutput::new(loss, probabilities, batch.targets);

        TrainOutput::new(self, gradients, item)
    }
}

// Same prediction and loss calculation, but no backward or weight update.
impl<B: Backend> InferenceStep for Model<B> {
    type Input = HeartBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let logits = self.logits(batch.inputs);
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init::<B>(&logits.device())
            .forward(logits.clone(), batch.targets.clone());

        let probabilities = burn::tensor::activation::sigmoid(logits);
        MultiLabelClassificationOutput::new(loss, probabilities, batch.targets)
    }
}
