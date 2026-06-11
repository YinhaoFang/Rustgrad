//! Neural network modules and layers.

use crate::tensor::Tensor;
use crate::{Result, RustGradError};

/// Common interface for neural network components.
pub trait Module {
    /// Computes the module output for an input tensor.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Returns immutable trainable parameters.
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    /// Returns mutable trainable parameters.
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }

    /// Returns a stable module name for reports and debugging.
    fn name(&self) -> &str;
}

/// Fully connected layer with row-major weights.
///
/// The layer computes `output = input @ weights + bias`. Weights have shape
/// `[input_size, output_size]`, and bias has shape `[output_size]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    input_size: usize,
    output_size: usize,
    weights: Tensor,
    bias: Tensor,
}

impl Linear {
    /// Creates a linear layer with deterministic small weights and zero bias.
    pub fn new(input_size: usize, output_size: usize) -> Result<Self> {
        Self::validate_size("input_size", input_size)?;
        Self::validate_size("output_size", output_size)?;

        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let weights = (0..input_size * output_size)
            .map(|index| {
                let pattern = (index % output_size) as f64 - (output_size as f64 - 1.0) / 2.0;
                pattern * scale
            })
            .collect();

        Self::with_parameters(
            Tensor::matrix(input_size, output_size, weights)?,
            Tensor::zeros(vec![output_size])?,
        )
    }

    /// Creates a linear layer from explicit weights and bias tensors.
    pub fn with_parameters(weights: Tensor, bias: Tensor) -> Result<Self> {
        let input_size = weights
            .rows()
            .ok_or_else(|| RustGradError::InvalidArgument {
                name: "weights",
                reason: format!("linear weights must be rank 2, got rank {}", weights.rank()),
            })?;
        let output_size = weights.cols().expect("rank 2 tensors always have columns");

        if !bias.shape().is_vector() || bias.len() != output_size {
            return Err(RustGradError::ShapeMismatch {
                op: "linear bias",
                left: vec![output_size],
                right: bias.shape().to_vec(),
            });
        }

        Ok(Self {
            input_size,
            output_size,
            weights,
            bias,
        })
    }

    /// Returns the expected number of input features.
    #[must_use]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Returns the number of output features.
    #[must_use]
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Returns the weight matrix.
    #[must_use]
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Returns mutable weights.
    #[must_use]
    pub fn weights_mut(&mut self) -> &mut Tensor {
        &mut self.weights
    }

    /// Returns the bias vector.
    #[must_use]
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// Returns mutable bias values.
    #[must_use]
    pub fn bias_mut(&mut self) -> &mut Tensor {
        &mut self.bias
    }

    fn validate_size(name: &'static str, value: usize) -> Result<()> {
        if value == 0 {
            Err(RustGradError::InvalidArgument {
                name,
                reason: "size must be greater than zero".to_string(),
            })
        } else {
            Ok(())
        }
    }

    fn forward_vector(&self, input: &Tensor) -> Result<Tensor> {
        if input.len() != self.input_size {
            return Err(RustGradError::ShapeMismatch {
                op: "linear input",
                left: vec![self.input_size],
                right: input.shape().to_vec(),
            });
        }

        let matrix = input.reshape(vec![1, self.input_size])?;
        let output = self.forward_matrix(&matrix)?;
        output.flatten()
    }

    fn forward_matrix(&self, input: &Tensor) -> Result<Tensor> {
        if input.cols() != Some(self.input_size) {
            return Err(RustGradError::ShapeMismatch {
                op: "linear input",
                left: vec![self.input_size],
                right: input.shape().to_vec(),
            });
        }

        let output = input.matmul(&self.weights)?;
        self.add_bias(&output)
    }

    fn add_bias(&self, output: &Tensor) -> Result<Tensor> {
        let rows = output.rows().expect("linear output must be rank 2");
        let cols = output.cols().expect("linear output must be rank 2");
        let mut data = Vec::with_capacity(output.len());

        for row in 0..rows {
            for col in 0..cols {
                data.push(output.get(&[row, col])? + self.bias.get_flat(col)?);
            }
        }

        Tensor::matrix(rows, cols, data)
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match input.rank() {
            1 => self.forward_vector(input),
            2 => self.forward_matrix(input),
            rank => Err(RustGradError::InvalidArgument {
                name: "input",
                reason: format!("linear input must be rank 1 or 2, got rank {rank}"),
            }),
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.bias]
    }

    fn name(&self) -> &str {
        "linear"
    }
}

/// Module wrapper around a stateless activation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationLayer {
    activation: Activation,
}

impl ActivationLayer {
    /// Creates a new activation layer.
    #[must_use]
    pub fn new(activation: Activation) -> Self {
        Self { activation }
    }

    /// Returns the wrapped activation.
    #[must_use]
    pub fn activation(&self) -> Activation {
        self.activation
    }
}

impl Module for ActivationLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.activation.apply(input)
    }

    fn name(&self) -> &str {
        self.activation.name()
    }
}

/// A simple sequential model that applies modules in insertion order.
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Creates an empty sequential container.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a sequential container from existing boxed layers.
    #[must_use]
    pub fn from_layers(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    /// Adds a layer to the end of the sequence.
    pub fn push<M>(&mut self, layer: M)
    where
        M: Module + 'static,
    {
        self.layers.push(Box::new(layer));
    }

    /// Returns the number of layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true when the sequence has no layers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns layer names in forward order.
    #[must_use]
    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.iter().map(|layer| layer.name()).collect()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut parameters = Vec::new();
        for layer in &self.layers {
            parameters.extend(layer.parameters());
        }
        parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut parameters = Vec::new();
        for layer in &mut self.layers {
            parameters.extend(layer.parameters_mut());
        }
        parameters
    }

    fn name(&self) -> &str {
        "sequential"
    }
}

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

#[cfg(test)]
mod tests {
    use super::{
        relu, relu_derivative, sigmoid, sigmoid_derivative_from_output, softmax, tanh,
        tanh_derivative_from_output, Activation,
    };
    use crate::tensor::Tensor;
    use crate::RustGradError;

    const EPSILON: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_slice_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn activation_names_are_stable() {
        assert_eq!(Activation::Relu.name(), "relu");
        assert_eq!(Activation::Sigmoid.name(), "sigmoid");
        assert_eq!(Activation::Tanh.name(), "tanh");
        assert_eq!(Activation::Softmax.name(), "softmax");
    }

    #[test]
    fn activation_enum_dispatches_to_relu() {
        let input = Tensor::vector(vec![-1.0, 0.0, 2.0]).expect("valid vector");

        let output = Activation::Relu.apply(&input).expect("relu should succeed");

        assert_eq!(output.data(), &[0.0, 0.0, 2.0]);
    }

    #[test]
    fn relu_clamps_negative_values() {
        let input = Tensor::matrix(2, 2, vec![-2.0, -0.0, 3.0, 4.0]).expect("valid matrix");

        let output = relu(&input).expect("relu should succeed");

        assert_eq!(output.dims(), &[2, 2]);
        assert_eq!(output.data(), &[0.0, -0.0, 3.0, 4.0]);
    }

    #[test]
    fn relu_derivative_marks_positive_inputs() {
        let input = Tensor::vector(vec![-2.0, 0.0, 3.0]).expect("valid vector");

        let output = relu_derivative(&input).expect("relu derivative should succeed");

        assert_eq!(output.data(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn sigmoid_matches_known_values() {
        let input = Tensor::vector(vec![0.0, 2.0]).expect("valid vector");

        let output = sigmoid(&input).expect("sigmoid should succeed");

        assert_slice_close(output.data(), &[0.5, 1.0 / (1.0 + (-2.0_f64).exp())]);
    }

    #[test]
    fn sigmoid_is_stable_for_large_negative_values() {
        let input = Tensor::vector(vec![-1000.0]).expect("valid vector");

        let output = sigmoid(&input).expect("sigmoid should succeed");

        assert!(output.data()[0].is_finite());
        assert_close(output.data()[0], 0.0);
    }

    #[test]
    fn sigmoid_derivative_uses_output_values() {
        let output = Tensor::vector(vec![0.25, 0.5, 0.75]).expect("valid vector");

        let derivative =
            sigmoid_derivative_from_output(&output).expect("sigmoid derivative should succeed");

        assert_slice_close(derivative.data(), &[0.1875, 0.25, 0.1875]);
    }

    #[test]
    fn tanh_matches_standard_library_values() {
        let input = Tensor::vector(vec![-1.0, 0.0, 1.0]).expect("valid vector");

        let output = tanh(&input).expect("tanh should succeed");

        assert_slice_close(output.data(), &[-1.0_f64.tanh(), 0.0, 1.0_f64.tanh()]);
    }

    #[test]
    fn tanh_derivative_uses_output_values() {
        let output = Tensor::vector(vec![-0.5, 0.0, 0.5]).expect("valid vector");

        let derivative = tanh_derivative_from_output(&output).expect("tanh derivative succeeds");

        assert_slice_close(derivative.data(), &[0.75, 1.0, 0.75]);
    }

    #[test]
    fn softmax_vector_outputs_probability_distribution() {
        let input = Tensor::vector(vec![1.0, 2.0, 3.0]).expect("valid vector");

        let output = softmax(&input).expect("softmax should succeed");
        let sum: f64 = output.data().iter().sum();

        assert_eq!(output.dims(), &[3]);
        assert_close(sum, 1.0);
        assert!(output.data()[2] > output.data()[1]);
        assert!(output.data()[1] > output.data()[0]);
    }

    #[test]
    fn softmax_matrix_normalizes_each_row() {
        let input = Tensor::matrix(2, 2, vec![1.0, 1.0, 1.0, 3.0]).expect("valid matrix");

        let output = softmax(&input).expect("softmax should succeed");
        let first_row_sum = output.data()[0] + output.data()[1];
        let second_row_sum = output.data()[2] + output.data()[3];

        assert_eq!(output.dims(), &[2, 2]);
        assert_close(first_row_sum, 1.0);
        assert_close(second_row_sum, 1.0);
        assert_slice_close(&output.data()[0..2], &[0.5, 0.5]);
        assert!(output.data()[3] > output.data()[2]);
    }

    #[test]
    fn softmax_is_stable_for_large_values() {
        let input = Tensor::vector(vec![1000.0, 1000.0]).expect("valid vector");

        let output = softmax(&input).expect("softmax should succeed");

        assert_slice_close(output.data(), &[0.5, 0.5]);
    }

    #[test]
    fn softmax_rejects_rank_three_tensor() {
        let input = Tensor::from_vec(vec![1, 1, 2], vec![1.0, 2.0]).expect("valid rank 3 tensor");

        assert_eq!(
            softmax(&input).expect_err("rank three should fail"),
            RustGradError::InvalidArgument {
                name: "rank",
                reason: "softmax supports rank 1 or 2, got rank 3".to_string(),
            }
        );
    }
}
