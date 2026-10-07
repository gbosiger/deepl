//! TODO: implement the Burn model yourself.
//
// TODO 1: Choose a backend and create a device; start with CPU execution.
// TODO 2: Learn how Burn represents tensors and their dimensions.
// TODO 3: Define a model with a 29 -> 16 hidden layer and a 16 -> 1 output layer.
// TODO 4: Initialize the layers, including their biases.
// TODO 5: Implement the forward pass with ReLU and a final sigmoid.
// TODO 6: Check that a batch of inputs produces one probability per input.
// TODO 7: Verify the model has 497 trainable parameters.
// TODO 8: Decide how to expose logits for a numerically stable training loss.

use burn::{
    module::Module,
    nn::{Linear, Relu, Sigmoid},
    tensor::Tensor,
};

#[derive(Module, Debug)]
pub struct Model {
    linear_input: Linear,
    hidden: Linear,
    linear_output: Linear,
    activation: Relu,
    end: Sigmoid,
}
