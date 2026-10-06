# Lecture 2: heart-disease binary classification

[Original notebook](https://colab.research.google.com/drive/1flLafeFpy8JjLN4H_ertcs5wJE3--TdQ?usp=sharing)

## What is provided

Only the Cargo setup, module files, two minimal executable entry points, and TODOs.
The existing `data/heart.csv` is unchanged. There are no prepared data structures,
preprocessing functions, backend aliases, tensor examples, or model implementations.

## Where to work

- `src/data.rs`: your row representation, CSV loading, splitting, and preprocessing.
- `src/model.rs`: your backend/device choices, tensors, layers, and forward pass.
- `src/training.rs`: your batching, loss, optimizer, training, evaluation, and saving.
- `src/bin/train.rs`: your training entry point and any command-line interface.
- `src/bin/predict.rs`: your input handling, artifact loading, and prediction.
- `src/lib.rs`: exposes the shared modules to both executables.

One package can have a library and multiple executables. Both executables can use
code from `lecture2::data`, `lecture2::model`, and `lecture2::training` as you add it.
You choose the types and function signatures.

## Learning sequence

1. Inspect the CSV and choose how to represent a row. Load it and print a few rows.
2. Work through splitting, normalization, and categorical encoding yourself.
3. Build and inspect the Burn model: 29 inputs, 16 ReLU units, one sigmoid output.
4. Add training and evaluation.
5. Save the model and preprocessing information, then implement prediction.

Use the notebook as the reference for the details. Start with one TODO at a time;
there is no need to design the entire pipeline before loading the first row.

From the workspace root, `cargo check --workspace --all-targets` checks your work.
`cargo run -p lecture2 --bin train` and `cargo run -p lecture2 --bin predict` run
the two entry points. No arguments or backend demo have been implemented.
