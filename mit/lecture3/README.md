# Lecture 3 — Fashion-MNIST image classification

Build this yourself, one milestone at a time. Start with a correct, reproducible
baseline; add complexity only after it works. This README is a plan, not an implementation.

## References and scope

- [Assignment notebook](https://colab.research.google.com/drive/1IDS6F8nkfszgqm54EvvnCPaUBXGgsZzf?usp=sharing#scrollTo=A-9nymirkCOe)
- [MIT lecture 3](https://ocw.mit.edu/courses/15-773-hands-on-deep-learning-spring-2024/resources/15773-sp24-lecture-3-version-2_mp4/)
- [Fashion-MNIST dataset and labels](https://github.com/zalandoresearch/fashion-mnist)

The lecture resource is titled “Deep Learning for Computer Vision — Building
Convolutional Neural Networks from Scratch.” The linked notebook actually starts
with a **fully connected image classifier**, then proposes a second dense hidden
layer. Reproduce that first. A convolutional model is a later extension, not part
of the baseline architecture below.

## What changes from lecture 2

- Images replace CSV patient records: grayscale pixels, 28 × 28 per image.
- There are ten mutually exclusive classes, rather than one binary target.
- Training uses ten logits and multiclass cross-entropy with integer class labels.
- Prediction uses softmax to produce ten probabilities that sum to one.
- Labels remain integers 0–9; they do not need one-hot encoding for this loss.
- Pixel preprocessing divides by 255. This is a fixed scale, so there are no fitted
  means/deviations and no normalization-statistics CSV to carry around.
- Fixed pixel scaling can happen before splitting without leaking information.
  Any later preprocessing that learns statistics must be fitted on training only.
- Use Burn’s trainer/TUI as the main path. Keep the manual training lesson in
  lecture 2 instead of maintaining another duplicate loop here.
- Polars remains a future tabular-data lesson. It is not needed for these images.

## Notebook baseline to reproduce

- Data: 60,000 development images and 10,000 official test images.
- Each image: 28 × 28 grayscale pixels with byte values 0–255.
- Validation: 20% of the 60,000 development examples, giving 48,000 training,
  12,000 validation, and 10,000 test examples.
- Input: flatten each image to 784 values, preserving pixel order.
- Network: 784 → 256 with ReLU → 10 logits; softmax for probabilities.
- Parameters with biases: (784 × 256 + 256) + (256 × 10 + 10) = **203,530**.
- Optimizer: Adam. Explicitly choose and record a learning rate; 0.001 is a
  reasonable starting point corresponding to the usual Keras Adam default.
- Batch size: 64. Epochs: 20. Seed: 42.
- Metrics: multiclass cross-entropy loss and top-1 accuracy.
- Optional notebook experiment: 784 → 256 → 256 → 10, ReLU after each hidden
  layer, **269,322** parameters.

Keras reserves the last 12,000 development examples before its training shuffle.
Choose either this exact split or an explicitly documented seeded/stratified split.
A Rust shuffle with seed 42 does not reproduce Keras’s random initialization or
shuffle sequence. Aim for comparable behavior, not identical metrics.

## Boundaries to keep clear

- Raw data loading owns acquisition, decoding, pixel bytes, labels, and validation.
- Preparation owns split membership and the fixed pixel transformation.
- The model accepts tensors; it does not download data or read files.
- Batching/data-loader adapters own tensor layout and device placement.
- Trainer steps own task-specific loss and reporting; Burn manages the loop.
- Executables own paths, configuration, launch, evaluation, and artifact selection.
- Share only what training and prediction both need. Keep trainer-only adapters
  out of common code. Start with modules in one crate, not a new framework.
- Reuse ideas from lecture 2. Extract cross-lecture helper crates only when actual
  duplicate code warrants them; do not make lecture 3 depend on patient-specific code.

Possible module responsibilities when you get there: common data/preparation/model,
trainer dataset/batcher/steps, and train/predict executable entry points. Create
files as each milestone needs them; no empty hierarchy is required upfront.

## Dataset setup

Use the official Fashion-MNIST compressed IDX files from
[Zalando's repository](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion).
Burn 0.21.0's built-in `MnistDataset` loads handwritten digits, not Fashion-MNIST.

The download task in `Makefile.toml` uses **cargo-make**, an external Cargo task
runner, not a Rust library dependency. Install it once:

```sh
cargo install cargo-make --locked
```

The task also requires `bash`, `curl`, and `gzip` on your PATH. From the workspace
root, run:

```sh
cargo make --cwd mit/lecture3 --no-workspace download-data
```

`--cwd` selects lecture 3's task file and data directory. `--no-workspace`
prevents cargo-make from running the task across all workspace members (lecture 2
does not define this task). If already inside `mit/lecture3`, use
`cargo make --no-workspace download-data` instead.

It downloads these four files into lecture-local `data/`:

- `train-images-idx3-ubyte.gz` — 60,000 development images.
- `train-labels-idx1-ubyte.gz` — matching development labels.
- `t10k-images-idx3-ubyte.gz` — 10,000 official test images.
- `t10k-labels-idx1-ubyte.gz` — matching test labels.

The task decompresses each file with `gzip -dk`, keeping the original `.gz` file
alongside its uncompressed IDX file. Existing downloads and uncompressed files
are skipped. The entire `mit/lecture3/data/` directory is ignored by Git.

This is an explicit setup step, not a build dependency: compiling, training, and
prediction should not silently download data. The Rust loader will read the
uncompressed IDX files using standard-library file I/O; no gzip dependency is
needed in the crate. IDX headers and payloads still need decoding and validation.

The current task does not verify checksums. Compare the compressed files against
the checksums in the official dataset README before trusting them; merely finding
an existing file does not validate its contents.

## Learning checklist

### 1. Workspace and experiment contract

- [x] Add lecture 3 to the workspace when its first executable exists.
- [x] Review the existing manifest: it currently calls the training binary
      `train_manual` but points at `src/bin/train.rs`. Choose a consistent name
      for the trainer path before writing it.
- [x] Decide initial backend (CPU for correctness), executable names, dataset
      source, and output location. Record Burn/Rust versions.
- [x] Write down input layout, output layout, label mapping, and preprocessing.
- [x] Keep existing lecture 2 code and artifacts separate.

Done when: the package and naming decisions are clear, and the first tiny executable
can be built without introducing unrelated abstractions.

### 2. Acquire and inspect the data

- [x] Choose Burn’s Fashion-MNIST loader if supported by the installed version,
      or the official compressed IDX files. Verify that you are loading
      **Fashion-MNIST**, not the similarly shaped handwritten-digit MNIST data.
- [x] Keep original data unchanged. Record source, file checksums where available,
      and how cached data is found; avoid hidden downloads during prediction.
- [x] Inspect dimensions, image/label counts, class counts, and label range.
- [x] If decoding IDX yourself, validate its header, dimensions, payload length,
      and image/label alignment. IDX is a binary format, not CSV.
- [ ] Display a small labeled image grid to check orientation and class meanings.

Done when: all 70,000 examples are accounted for and a few images look correct.
Do not download or implement a loader until you have chosen the source.

### 3. Fix split membership

- [ ] Preserve the official 10,000-image test split.
- [ ] Divide the 60,000 development examples into 48,000 training and 12,000 validation.
- [ ] Store split indices or a precise reproducible split specification.
- [ ] Check disjoint membership and matching image/label order.
- [ ] Use training data for weight updates, validation for decisions, and test
      data for final evaluation. Do not repeatedly tune against test results.

Done when: membership is reproducible and there is no overlap across groups.

### 4. Prepare values and tensors

- [ ] Convert byte pixels to the model’s float element type and divide by 255.
- [ ] Keep cached pixels as bytes if useful; avoid a second full converted copy
      unless measurements justify it. Convert/gather batches at the boundary.
- [ ] For the dense baseline, produce inputs [batch, 784] and integer targets
      [batch]. Ensure flattening keeps rows and columns in a consistent order.
- [ ] Check values are finite and in [0, 1]; class labels are 0–9.
- [ ] Verify selected sample values and labels survive batching unchanged.
- [ ] Support a smaller final batch; shuffle training samples between epochs.
- [ ] Share the same transformation with prediction, including orientation,
      grayscale interpretation, image size, and flattening order.

Done when: one batch is correct in shape, values, and label alignment.

### 5. Define and smoke-check the dense model

- [ ] Initialize the two biased linear layers and ReLU from a model configuration.
- [ ] Expose logits for loss and probabilities for prediction; avoid applying
      softmax before a loss that already expects logits.
- [ ] Confirm 203,530 parameters and output shape [batch, 10].
- [ ] Confirm each probability vector is finite, lies in [0, 1], and sums to
      approximately one. Top-1 prediction is the index of its largest value.
- [ ] Check one batch with a multiclass loss, using integer class targets.
- [ ] Make model initialization reproducible as well as dataset shuffling.

Done when: forward pass and loss work; untrained outputs are not presented as learned results.

### 6. Connect Burn’s trainer and establish a baseline

- [ ] Implement the multiclass training/validation result and step adapters.
- [ ] Configure Adam, batch size 64, 20 epochs, loss, and accuracy in the TUI.
- [ ] Run a brief smoke experiment; confirm gradients/updates work and memory
      use is sensible before committing to a full run.
- [ ] Use a tiny-subset overfit check if the pipeline is suspect: the model should
      be able to learn a small fixed group of examples.
- [ ] Run the baseline and retain epoch metrics, not just screenshots.
- [ ] Measure elapsed training time separately from data preparation and the
      persistent TUI wait. Record hardware and backend.
- [ ] Compare accuracy with uniform random guessing (10%) and a majority-class
      baseline. Uniform class probabilities give cross-entropy ln(10), about 2.303.

Done when: the baseline has recorded training and validation curves and a reproducible configuration.

### 7. Choose a model using validation

- [ ] Inspect both loss and accuracy curves for overfitting.
- [ ] Select the epoch/checkpoint using validation, with the selection rule recorded.
- [ ] Keep separate run records for changes to learning rate, seed, architecture,
      or epoch count; do not silently overwrite the baseline.
- [ ] Try the notebook’s second hidden layer only after the first baseline works.
- [ ] Compare it against the baseline using the same split and explicit settings;
      more parameters do not guarantee better validation performance.

Done when: you can explain which model was selected and why, without using test performance to select it.

### 8. Final evaluation and saved artifacts

- [ ] Evaluate the selected model on the official test set.
- [ ] Report loss, top-1 accuracy, and a confusion matrix/per-class results.
- [ ] Inspect some mistakes alongside their images and probabilities.
- [ ] Save model configuration, weights, class names, and preprocessing description.
- [ ] Save the split specification, run configuration, seeds, versions, metrics,
      timing, and dataset provenance beside the model.
- [ ] Confirm reloading reproduces the same predictions within numeric tolerance.

Done when: a fresh process can load the selected model and reproduce its evaluation.

### 9. Prediction as a separate executable

- [ ] First predict a selected Fashion-MNIST example by index, without refitting
      preprocessing or relying on its label to generate inputs.
- [ ] Return the top class name and ten probabilities; optionally show the top three.
- [ ] Then support an external image only after defining how grayscale conversion,
      resizing, foreground/background, and orientation match training data.
- [ ] Time inference after warm-up and multiple repetitions. Wait for the result
      before stopping a GPU timer; separate loading and preprocessing from inference.
- [ ] Handle invalid dimensions, unreadable input, and incompatible artifacts clearly.

Done when: prediction needs only the selected artifacts and an input image, not a training run.

### 10. GPU and later experiments

- [ ] Verify driver/device readiness and explicitly record the selected GPU backend.
- [ ] Repeat the same correctness checks on GPU; compare CPU/GPU outputs with tolerance.
- [ ] Compare warm-up and steady-state timings, including data movement where relevant.
- [ ] Consider a CNN next: document [batch, channels, height, width] layout and
      each convolution/pooling shape rather than reusing flattened dense inputs blindly.
- [ ] Leave augmentation, more workers, and larger experiments until the baseline
      is trustworthy. Apply stochastic augmentation to training only.

## Run files and naming

Use lecture-local raw data/cache and a separate directory per experiment under
`generated/runs/`. A run name should distinguish dense baseline, deeper dense,
and any later CNN, plus a unique run identifier. Never mix these with lecture 2.

Each completed run should contain weights, architecture/configuration,
preprocessing/class mapping, dataset/split provenance, metrics, and timing.
Keep one consistent run identifier across them. Prediction selects a run directory,
not a loose weight file guessed from a filename. Keep generated artifacts out of
Git unless intentionally publishing a small reference artifact.

## Start here

Do **milestone 1**, then acquire and inspect the dataset. Do not implement the
model, trainer, GPU path, and prediction at once. Ask for review at each “Done when”
checkpoint; code changes should stay within the step you are working on.
