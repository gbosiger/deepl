//! TODO: implement the Burn model yourself.
//
// TODO 1: Choose a backend and create a device; start with CPU execution.
// TODO 2: Learn how Burn represents tensors and their dimensions.
// TODO 3: Define a model with a 29 -> 16 hidden layer and a 16 -> 1 output layer.
// TODO 4: Initialize the layers, including their biases.
// TODO 5: Implement the forward pass with ReLU and a final sigmoid.
//
// TODO 6: Check that a batch of inputs produces one probability per input.
// TODO 7: Verify the model has 497 trainable parameters.
// TODO 8: Decide how to expose logits for a numerically stable training loss.

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu, Sigmoid},
    tensor::backend::Backend,
    tensor::Tensor,
};

#[derive(Config, Debug)]
pub struct ModelConfig {
    input_size: usize,
    hidden_size: usize,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    hidden: Linear<B>,
    linear_output: Linear<B>,
    activation: Relu,
    end: Sigmoid,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            hidden: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            linear_output: LinearConfig::new(self.hidden_size, 1).init(device),
            activation: Relu::new(),
            end: Sigmoid::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    // rank 2 represents [batch size, feature count]
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // hidden -> relu -> output -> sigmoid
        let x = self.hidden.forward(input);
        let x = self.activation.forward(x);
        let x = self.linear_output.forward(x);

        self.end.forward(x)
    }
}
