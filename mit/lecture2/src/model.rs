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
    pub fn logits(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // hidden -> relu -> output
        let x = self.hidden.forward(input);
        let x = self.activation.forward(x);
        self.linear_output.forward(x)
    }

    // rank 2 represents [batch size, feature count]
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // hidden -> relu -> output -> sigmoid
        self.end.forward(self.logits(input))
    }
}
