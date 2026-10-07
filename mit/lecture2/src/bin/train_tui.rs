use burn::{
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::DefaultRecorder,
    tensor::backend::BackendTypes,
    train::{
        Learner, SupervisedTraining, metric::LossMetric, renderer::tui::TuiMetricsRendererWrapper,
    },
};
use lecture2::{
    common::{data::write_normalization_params, model::ModelConfig, prep::prepare_data},
    trainer::batcher::HeartBatcher,
};
use std::{io::IsTerminal, path::Path, sync::Arc};

type InferenceBackend = burn::backend::Flex;
type TrainingBackend = burn::backend::Autodiff<InferenceBackend>;
type FloatElem = <TrainingBackend as BackendTypes>::FloatElem;
type IntElem = <TrainingBackend as BackendTypes>::IntElem;

fn main() -> anyhow::Result<()> {
    let main_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let prepared = prepare_data::<FloatElem, IntElem>(&main_path.join("data/heart.csv"))?;
    let output = main_path.join("generated/tui");
    std::fs::create_dir_all(&output)?;
    write_normalization_params(&output.join("normalization.csv"), &prepared.normalization)?;

    let training = Arc::new(prepared.training);
    let validation = Arc::new(prepared.validation);
    // Keep prepared.testing for final evaluation when the trainer is connected.

    // Share the same buffers between each dataset and its batcher.
    let training_loader =
        DataLoaderBuilder::new(HeartBatcher::<TrainingBackend>::new(Arc::clone(&training)))
            .batch_size(32)
            .shuffle(42)
            .num_workers(0)
            .build(training);

    let validation_loader = DataLoaderBuilder::new(HeartBatcher::<InferenceBackend>::new(
        Arc::clone(&validation),
    ))
    .batch_size(32)
    .num_workers(0)
    .build(validation);

    let device = Default::default();
    let model = ModelConfig::new(29, 16).init::<TrainingBackend>(&device);
    let optimizer = AdamConfig::new().init();

    let training = SupervisedTraining::new(&output, training_loader, validation_loader)
        .metrics((LossMetric::<InferenceBackend>::new(),))
        .num_epochs(20);

    // Keep the finished graph open until I close it; redirected output stays plain.
    let training = if std::io::stdout().is_terminal() {
        let renderer = TuiMetricsRendererWrapper::new(training.interrupter(), None).persistent();
        training.renderer(renderer)
    } else {
        training
    };

    let result = training.launch(Learner::new(model, optimizer, 0.001));

    result
        .model
        .save_file(output.join("model"), &DefaultRecorder::new())?;

    Ok(())
}
