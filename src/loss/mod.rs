//! Loss functions for supervised learning.

use crate::tensor::Tensor;
use crate::{Result, RustGradError};

/// Common interface for supervised learning losses.
pub trait Loss {
    /// Computes a scalar-like loss tensor.
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;

    /// Returns a stable loss name for reports and debugging.
    fn name(&self) -> &str;
}

/// Mean squared error loss.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MSELoss;

impl MSELoss {
    /// Creates a mean squared error loss.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Loss for MSELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        ensure_same_shape("mse", predictions, targets)?;

        let total: f64 = predictions
            .data()
            .iter()
            .zip(targets.data().iter())
            .map(|(&prediction, &target)| {
                let diff = prediction - target;
                diff * diff
            })
            .sum();

        Tensor::scalar(total / predictions.len() as f64)
    }

    fn name(&self) -> &str {
        "mse"
    }
}

/// Cross entropy loss for one-hot targets.
///
/// Predictions are interpreted as logits. Targets must have the same shape as
/// predictions and are expected to be one-hot or probability distributions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossEntropyLoss {
    epsilon: f64,
}

impl CrossEntropyLoss {
    /// Creates a cross entropy loss with a small numerical epsilon.
    #[must_use]
    pub fn new() -> Self {
        Self { epsilon: 1e-12 }
    }

    /// Creates a cross entropy loss with a caller-provided epsilon.
    pub fn with_epsilon(epsilon: f64) -> Result<Self> {
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(RustGradError::InvalidArgument {
                name: "epsilon",
                reason: "epsilon must be finite and greater than zero".to_string(),
            });
        }

        Ok(Self { epsilon })
    }

    /// Returns the numerical epsilon used inside logarithms.
    #[must_use]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        ensure_same_shape("cross entropy", predictions, targets)?;

        match predictions.rank() {
            1 => {
                let loss = cross_entropy_row(predictions.data(), targets.data(), self.epsilon)?;
                Tensor::scalar(loss)
            }
            2 => {
                let rows = predictions.rows().expect("rank 2 tensors always have rows");
                let cols = predictions
                    .cols()
                    .expect("rank 2 tensors always have columns");
                let mut total = 0.0;

                for row in 0..rows {
                    let start = row * cols;
                    let end = start + cols;
                    total += cross_entropy_row(
                        &predictions.data()[start..end],
                        &targets.data()[start..end],
                        self.epsilon,
                    )?;
                }

                Tensor::scalar(total / rows as f64)
            }
            rank => Err(RustGradError::InvalidArgument {
                name: "rank",
                reason: format!("cross entropy supports rank 1 or 2, got rank {rank}"),
            }),
        }
    }

    fn name(&self) -> &str {
        "cross_entropy"
    }
}

fn ensure_same_shape(op: &'static str, predictions: &Tensor, targets: &Tensor) -> Result<()> {
    if predictions.dims() == targets.dims() {
        Ok(())
    } else {
        Err(RustGradError::ShapeMismatch {
            op,
            left: predictions.shape().to_vec(),
            right: targets.shape().to_vec(),
        })
    }
}

fn cross_entropy_row(logits: &[f64], targets: &[f64], epsilon: f64) -> Result<f64> {
    validate_target_distribution(targets)?;

    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = logits.iter().map(|logit| (logit - max_logit).exp()).sum();
    let log_sum_exp = max_logit + exp_sum.ln();
    let loss = logits
        .iter()
        .zip(targets.iter())
        .map(|(&logit, &target)| {
            let probability = (logit - log_sum_exp).exp().max(epsilon);
            -target * probability.ln()
        })
        .sum();

    Ok(loss)
}

fn validate_target_distribution(targets: &[f64]) -> Result<()> {
    if targets
        .iter()
        .any(|target| *target < 0.0 || !target.is_finite())
    {
        return Err(RustGradError::InvalidArgument {
            name: "targets",
            reason: "targets must be finite and non-negative".to_string(),
        });
    }

    let total: f64 = targets.iter().sum();
    if (total - 1.0).abs() > 1e-9 {
        return Err(RustGradError::InvalidArgument {
            name: "targets",
            reason: format!("targets must sum to 1.0, got {total}"),
        });
    }

    Ok(())
}
