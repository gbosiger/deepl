# Lecture 2: heart-disease binary classification

[Original notebook](https://colab.research.google.com/drive/1flLafeFpy8JjLN4H_ertcs5wJE3--TdQ?usp=sharing)
[Lecture](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/video_galleries/lecture-videos/)

## Learning sequence

1. Inspect the CSV and choose how to represent a row. Load it and print a few rows.
2. Work through splitting, normalization, and categorical encoding yourself.
3. Build and inspect the Burn model: 29 inputs, 16 ReLU units, one sigmoid output.
4. Add training and evaluation.
5. Save the model and preprocessing information, then implement prediction.

## General notes

Used the notebook as the reference for the details. 
Data manipulation is coded in pure Rust (with csv and serde for example). This is
intentional in order to avoid learning some new framework (like Polars) in this stage.
Also there are two executables for training, one manually coded and the second with Burn
TUI.
This helps to compare how much data preparation code is needed for a simple example
and to understand implementation details needed for that.

## Running the example

From the workspace root, `cargo check --workspace --all-targets` checks your work.
`cargo run -p lecture2 --bin train` and `cargo run -p lecture2 --bin predict` run
the two entry points. No arguments or backend demo have been implemented.
