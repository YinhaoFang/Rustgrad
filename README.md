# RustGrad

RustGrad is a Rust course project that implements the core pieces of a small
deep learning framework. It is designed to show how tensor operations,
automatic differentiation, neural network layers, losses, optimizers, datasets,
training loops, and experiment reports fit together inside one inspectable
project.

The project is not a wrapper around an existing deep learning framework. Core
tensor operations, gradient propagation, model layers, optimizers, training
loops, and CLI examples are implemented in this repository so the design can be
inspected, tested, and explained clearly in a course report.

## Implemented Features

| Area | Functionality |
| --- | --- |
| Tensor | Shape validation, indexing, reshape, arithmetic, reductions, transpose, matrix multiplication |
| Autograd | Computation graph nodes, dependency ordering, backward traversal, gradient accumulation |
| Neural networks | `Linear`, `Sequential`, ReLU, Sigmoid, Tanh, Softmax |
| Losses | Mean squared error and cross entropy |
| Optimizers | SGD, Momentum, Adam |
| Data | Deterministic linear regression, XOR, and spiral datasets |
| Training | Reusable training configs, metrics, histories, and convergence examples |
| CLI | `train-linear`, `train-xor`, `train-spiral`, `inspect` |
| Reports | Markdown summaries and CSV loss curves through `--output DIR` |
| Quality | Unit tests, CLI integration tests, formatting, Clippy, and GitHub Actions CI |

## Project Structure

```text
src/
  autograd/   dynamic computation graph and backward rules
  data/       synthetic datasets used by examples and tests
  loss/       MSE and cross entropy losses
  nn/         layers, activations, and module abstractions
  optim/      SGD, Momentum, and Adam optimizers
  report/     Markdown and CSV training report export
  tensor/     dense tensor shape, indexing, math, and reductions
  train/      training loops and metrics
  main.rs     command-line interface
tests/
  cli.rs      end-to-end CLI integration tests
```

## Documentation

- [Autograd design](docs/autograd.md): computation graph, backward propagation,
  gradient accumulation, and shape handling.
- [Training workflow](docs/training.md): datasets, training loops, optimizer
  flow, CLI examples, and report export.
- [Testing strategy](docs/testing.md): unit tests, CLI tests, integration
  coverage, reproducibility, and Windows notes.
- [Technical report](docs/experiment-report.md): bilingual course report that
  connects the architecture, implementation, validation, and limits.
- [Changelog](CHANGELOG.md): project milestone summary and known limits.

## Quick Start

Build and run the tests:

```bash
cargo build
cargo test
```

Run a linear regression example:

```bash
cargo run -- train-linear --epochs 120 --learning-rate 0.12 --samples 31
```

Run the XOR MLP example:

```bash
cargo run -- train-xor --epochs 160 --learning-rate 0.4
```

Run the spiral softmax classifier and export report files:

```bash
cargo run -- train-spiral --epochs 160 --learning-rate 0.7 --samples-per-class 12 --classes 3 --output runs/spiral-demo
```

The `--output` directory receives:

- `summary.md`: Markdown summary with training metrics and history table.
- `history.csv`: CSV loss and accuracy curve for plotting or reports.

Inspect a compact snapshot of trained example models:

```bash
cargo run -- inspect
```

## CLI Reference

```text
rustgrad train-linear [--epochs N] [--learning-rate LR] [--samples N] [--slope V] [--intercept V] [--format text|csv|markdown] [--output DIR]
rustgrad train-xor [--epochs N] [--learning-rate LR] [--format text|csv|markdown] [--output DIR]
rustgrad train-spiral [--epochs N] [--learning-rate LR] [--samples-per-class N] [--classes N] [--format text|csv|markdown] [--output DIR]
rustgrad inspect
rustgrad --version
```

Output formats:

- `text`: human-readable summary plus useful predictions or parameter details.
- `csv`: machine-readable training history.
- `markdown` or `md`: report-ready Markdown.

## Example Output

```text
XOR MLP training
epochs=5
initial_loss=0.242958
final_loss=0.232110
best_loss=0.232110
loss_improvement=0.010848
best_accuracy=1.000000
last=epoch=5 loss=0.232110 accuracy=1.000000
probabilities=[0.205530]; [0.791871]; [0.791871]; [0.206772]
classes=[0.000000]; [1.000000]; [1.000000]; [0.000000]
```

## Development

Run the standard quality checks before each commit:

```bash
cargo fmt --check
cargo test
cargo clippy -- -D warnings
```

The repository includes a GitHub Actions workflow that runs formatting, build,
tests, and Clippy checks on push and pull request events.

This project works directly on Windows with stable Rust and Cargo. WSL2 is not
required for the current CPU-only implementation.

## Experiment Scoring Alignment

- **Code scale and complexity:** the framework is split into tensor, autograd,
  neural network, optimizer, data, training, report, and CLI modules.
- **Testing and quality:** the project includes broad unit coverage plus
  end-to-end CLI integration tests for success and error cases.
- **Engineering standards:** CI, conventional commit-friendly history, clear
  module boundaries, and report export are part of the repository.
- **Project acceptance:** examples show linear regression fitting, XOR
  classification, and non-linear spiral classification through a softmax
  classifier with feature mapping.
- **Experiment report:** `--output DIR` generates Markdown and CSV artifacts
  that can be attached to or copied into the final course report.

## License

This project is licensed under the MIT license.
