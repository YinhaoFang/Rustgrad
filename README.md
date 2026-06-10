# RustGrad

RustGrad is a teaching-oriented deep learning framework written in Rust. It is
designed for learning how tensor operations, automatic differentiation, neural
network layers, losses, optimizers, and training loops fit together inside a
small but complete framework.

The project is intentionally not a wrapper around an existing deep learning
library. Core tensor operations, gradient propagation, model layers, and
optimizers are implemented in this repository so the design can be inspected,
tested, and explained clearly.

## Goals

- Provide a compact `Tensor` type for one-dimensional and two-dimensional data.
- Implement common tensor operations such as elementwise arithmetic, matrix
  multiplication, transpose, sum, and mean.
- Build a dynamic automatic differentiation engine with `backward()` and
  gradient storage.
- Provide basic neural network building blocks: activations, linear layers, and
  sequential models.
- Implement supervised learning losses and optimizers.
- Include small training examples for linear regression, XOR classification, and
  spiral classification.
- Export training logs that can be used in experiment reports.

## Planned Features

The project will be developed in small, reviewable commits.

| Area | Planned functionality |
| --- | --- |
| Tensor | Shape validation, indexing, reshape, arithmetic, reductions, matmul |
| Autograd | Computation graph nodes, backward traversal, gradient accumulation |
| Neural networks | ReLU, Sigmoid, Tanh, Softmax, Linear, Sequential |
| Losses | Mean squared error and cross entropy |
| Optimizers | SGD, Momentum, Adam |
| Training | Reusable loops, metrics, deterministic synthetic datasets |
| CLI | `train-linear`, `train-xor`, `train-spiral`, `inspect` |
| Reports | CSV and Markdown loss summaries |

## Example CLI

The final command-line interface is planned to look like this:

```text
rustgrad train-linear --epochs 200 --learning-rate 0.05
rustgrad train-xor --epochs 1000 --hidden-size 8
rustgrad train-spiral --epochs 500 --classes 3
rustgrad inspect runs/latest
```

These commands will be added as the framework grows.

## Development

Run the standard quality checks before each commit:

```bash
cargo fmt --check
cargo test
cargo clippy -- -D warnings
```

The repository also contains a GitHub Actions workflow that runs formatting,
build, tests, and Clippy checks on push and pull request events.

## Experiment Scoring Alignment

This project is structured to match the Rust course experiment requirements:

- **Code scale and complexity:** the framework is split into tensor, autograd,
  neural network, optimizer, data, training, and report modules.
- **Testing and quality:** core behavior will be covered by unit and integration
  tests, including normal paths, boundary cases, and error handling.
- **Engineering standards:** CI, conventional commits, documentation, and a
  changelog are planned from the beginning.
- **Project acceptance:** training examples will demonstrate that the framework
  can fit simple datasets and produce reproducible results.
- **Experiment report:** generated logs and documentation will support a clear
  GitHub Issue report.

## License

This project is licensed under the MIT license.
