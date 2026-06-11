//! Neural network modules and layers.

use crate::tensor::Tensor;
use crate::{Result, RustGradError};

/// Common activation functions used by neural networks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Rectified linear unit.
    Relu,
    /// Logistic sigmoid.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Softmax normalization.
    Softmax,
}

impl Activation {
    /// Applies the activation function to a tensor.
    pub fn apply(self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Relu => relu(input),
            Self::Sigmoid => sigmoid(input),
            Self::Tanh => tanh(input),
            Self::Softmax => softmax(input),
        }
    }

    /// Returns the activation name used in reports and debug output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::Softmax => "softmax",
        }
    }
}

/// Applies ReLU element by element.
pub fn relu(input: &Tensor) -> Result<Tensor> {
    map_values(input, |value| value.max(0.0))
}

/// Returns the elementwise derivative of ReLU with respect to its input.
pub fn relu_derivative(input: &Tensor) -> Result<Tensor> {
    map_values(input, |value| if value > 0.0 { 1.0 } else { 0.0 })
}

/// Applies the logistic sigmoid element by element.
pub fn sigmoid(input: &Tensor) -> Result<Tensor> {
    map_values(input, stable_sigmoid)
}

/// Returns the elementwise derivative of sigmoid from sigmoid output values.
pub fn sigmoid_derivative_from_output(output: &Tensor) -> Result<Tensor> {
    map_values(output, |value| value * (1.0 - value))
}

/// Applies hyperbolic tangent element by element.
pub fn tanh(input: &Tensor) -> Result<Tensor> {
    map_values(input, f64::tanh)
}

/// Returns the elementwise derivative of tanh from tanh output values.
pub fn tanh_derivative_from_output(output: &Tensor) -> Result<Tensor> {
    map_values(output, |value| 1.0 - value * value)
}

/// Applies numerically stable softmax.
///
/// Vectors are normalized as one distribution. Matrices are normalized row by
/// row, which matches the common batched-classification case.
pub fn softmax(input: &Tensor) -> Result<Tensor> {
    match input.rank() {
        1 => Tensor::vector(softmax_slice(input.data())),
        2 => softmax_matrix(input),
        rank => Err(RustGradError::InvalidArgument {
            name: "rank",
            reason: format!("softmax supports rank 1 or 2, got rank {rank}"),
        }),
    }
}

fn map_values(input: &Tensor, apply: impl Fn(f64) -> f64) -> Result<Tensor> {
    let data = input.data().iter().map(|&value| apply(value)).collect();
    Tensor::from_vec(input.shape().to_vec(), data)
}

fn stable_sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn softmax_matrix(input: &Tensor) -> Result<Tensor> {
    let rows = input.rows().expect("rank 2 tensors always have rows");
    let cols = input.cols().expect("rank 2 tensors always have columns");
    let mut data = Vec::with_capacity(input.len());

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        data.extend(softmax_slice(&input.data()[start..end]));
    }

    Tensor::matrix(rows, cols, data)
}

fn softmax_slice(values: &[f64]) -> Vec<f64> {
    let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = values
        .iter()
        .map(|value| (value - max_value).exp())
        .collect();
    let sum: f64 = exp_values.iter().sum();

    exp_values.iter().map(|value| value / sum).collect()
}
