use burn::{module::Module, record::DefaultRecorder};
use lecture2::{data::read_normalization_params, model::ModelConfig};
use std::path::Path;

type InferenceBackend = burn::backend::Flex;

fn main() -> anyhow::Result<()> {
    let generated = Path::new(env!("CARGO_MANIFEST_DIR")).join("generated");

    let normalization = read_normalization_params(&generated.join("normalization_manual.csv"))?;

    let device = Default::default();
    let model = ModelConfig::new(29, 16)
        .init::<InferenceBackend>(&device)
        .load_file(
            generated.join("model_manual"),
            &DefaultRecorder::new(),
            &device,
        )?;

    assert_eq!(model.num_params(), 497);
    println!("Loaded model and normalization");
    println!("{normalization:?}");

    Ok(())
}
