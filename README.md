# Hands on Deep Learning with Burn

A Cargo workspace with one package per exercise that I do for learning using
various sources.

The root Cargo.toml is a virtual workspace: it has no executable of its own.
`[workspace.dependencies]` centralizes dependency versions. Each package opts in
with `workspace = true` and selects the Burn features it needs.

To add an exercise, create a package directory, add it to `workspace.members`,
and inherit dependencies as lecture2 does. Share code between exercises later
when there is a clear need for it.

From this directory:

```sh
cargo check --workspace --all-targets
cargo run -p lecture2 --bin train
cargo run -p lecture2 --bin predict
```

