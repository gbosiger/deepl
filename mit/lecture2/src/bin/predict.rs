use burn::{module::Module, record::DefaultRecorder, tensor::backend::BackendTypes};
use lecture2::{batch::HeartBatch, data::*, model::ModelConfig};
use std::path::Path;

type InferenceBackend = burn::backend::Flex;

fn main() -> anyhow::Result<()> {
    let generated = Path::new(env!("CARGO_MANIFEST_DIR")).join("generated");

    // New runs use their own directory; older saved manual models still load.
    let manual = generated.join("manual");
    let (normalization_path, model_path) = if manual.join("model.mpk").exists() {
        (manual.join("normalization.csv"), manual.join("model"))
    } else {
        (
            generated.join("normalization_manual.csv"),
            generated.join("model_manual"),
        )
    };
    let normalization = read_normalization_params(&normalization_path)?;

    let device = Default::default();
    let model = ModelConfig::new(29, 16)
        .init::<InferenceBackend>(&device)
        .load_file(model_path, &DefaultRecorder::new(), &device)?;

    assert_eq!(model.num_params(), 497);
    println!("Loaded model and normalization");
    println!("{normalization:?}");

    // Use the first CSV patient to check the saved-model prediction pipeline.
    // Its target is not passed into the model; this is not a held-out evaluation.
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("heart.csv");
    let patients = read_heart_data(&path)?;
    let patient = patients
        .first()
        .ok_or_else(|| anyhow::anyhow!("Patient CSV is empty"))?;
    let sample = vec![patient];
    let normalized = normalize_heart_data(&sample, &normalization);
    let encoded = encode_samples::<
        <InferenceBackend as BackendTypes>::FloatElem,
        <InferenceBackend as BackendTypes>::IntElem,
    >(sample, normalized);
    let batch = HeartBatch::<InferenceBackend>::from_encoded(encoded, &device)?;
    let probability = model.forward(batch.inputs).into_scalar();
    println!("First patient: predicted probability = {probability}");

    Ok(())
}
