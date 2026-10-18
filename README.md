# My Hands on Deep Learning with Rust + Burn

A Cargo workspace with one package per exercise that I do for learning using
various sources. I work with agents, but not outsource main work to them in
order to learn different details.

The root Cargo.toml is a virtual workspace: it has no executable of its own.
`[workspace.dependencies]` centralizes dependency versions. Each package opts in
with `workspace = true` and selects the Burn features it needs.

To add an exercise, create a package directory, add it to `workspace.members`,
and inherit dependencies as colab_3a does. Share code between exercises later
when there is a clear need for it.

From this directory:

```sh
cargo check --workspace --all-targets
cargo run -p colab_3a --bin train
cargo run -p colab_3a --bin predict
```

