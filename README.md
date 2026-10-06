# Learning Burn

A Cargo workspace with one package per exercise. Start with
[MIT lecture 2](mit/lecture2/README.md).

The root Cargo.toml is a virtual workspace: it has no executable of its own.
`[workspace.dependencies]` centralizes dependency versions. Each package opts in
with `workspace = true` and selects the Burn features it needs.

Lecture 2 uses Burn 0.21.0 with the CPU Flex backend and autodiff support enabled.
The dependency configuration is ready; using those features is part of your exercise.
Commit Cargo.lock to keep the executable dependencies reproducible.

To add an exercise, create a package directory, add it to `workspace.members`,
and inherit dependencies as lecture2 does. Share code between exercises later
when there is a clear need for it.

From this directory:

```sh
cargo check --workspace --all-targets
cargo run -p lecture2 --bin train
cargo run -p lecture2 --bin predict
```

Both executables currently print TODO messages. All data handling, Burn tensor
usage, model construction, training, prediction, and command-line handling are
left for you to implement.
