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

#[cfg(test)]
mod tests {
    use super::{CrossEntropyLoss, Loss, MSELoss};
    use crate::tensor::Tensor;
    use crate::RustGradError;

    const EPSILON: f64 = 1e-10;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    fn scalar_value(tensor: &Tensor) -> f64 {
        assert_eq!(tensor.dims(), &[1]);
        tensor.data()[0]
    }

    #[test]
    fn mse_loss_name_is_stable() {
        assert_eq!(MSELoss::new().name(), "mse");
    }

    #[test]
    fn mse_loss_returns_mean_squared_error_for_vectors() {
        let predictions = Tensor::vector(vec![1.0, 2.0, 4.0]).expect("valid predictions");
        let targets = Tensor::vector(vec![1.0, 1.0, 1.0]).expect("valid targets");

        let loss = MSELoss::new()
            .forward(&predictions, &targets)
            .expect("loss should compute");

        assert_close(scalar_value(&loss), 10.0 / 3.0);
    }

    #[test]
    fn mse_loss_returns_mean_squared_error_for_matrices() {
        let predictions =
            Tensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid predictions");
        let targets = Tensor::matrix(2, 2, vec![1.0, 1.0, 5.0, 4.0]).expect("valid targets");

        let loss = MSELoss::new()
            .forward(&predictions, &targets)
            .expect("loss should compute");

        assert_close(scalar_value(&loss), 1.25);
    }

    #[test]
    fn mse_loss_rejects_shape_mismatch() {
        let predictions = Tensor::vector(vec![1.0, 2.0]).expect("valid predictions");
        let targets = Tensor::matrix(1, 2, vec![1.0, 2.0]).expect("valid targets");

        let error = MSELoss::new()
            .forward(&predictions, &targets)
            .expect_err("shape mismatch should fail");

        assert_eq!(
            error,
            RustGradError::ShapeMismatch {
                op: "mse",
                left: vec![2],
                right: vec![1, 2],
            }
        );
    }

    #[test]
    fn cross_entropy_loss_name_and_default_epsilon_are_stable() {
        let loss = CrossEntropyLoss::new();

        assert_eq!(loss.name(), "cross_entropy");
        assert_close(loss.epsilon(), 1e-12);
        assert_eq!(CrossEntropyLoss::default(), loss);
    }

    #[test]
    fn cross_entropy_rejects_invalid_epsilon() {
        let error = CrossEntropyLoss::with_epsilon(0.0).expect_err("zero epsilon should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "epsilon",
                reason: "epsilon must be finite and greater than zero".to_string(),
            }
        );
    }

    #[test]
    fn cross_entropy_vector_accepts_one_hot_targets() {
        let predictions = Tensor::vector(vec![1.0, 2.0, 3.0]).expect("valid predictions");
        let targets = Tensor::vector(vec![0.0, 0.0, 1.0]).expect("valid targets");

        let loss = CrossEntropyLoss::new()
            .forward(&predictions, &targets)
            .expect("loss should compute");

        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln() - 3.0;
        assert_close(scalar_value(&loss), expected);
    }

    #[test]
    fn cross_entropy_vector_accepts_soft_targets() {
        let predictions = Tensor::vector(vec![0.0, 0.0]).expect("valid predictions");
        let targets = Tensor::vector(vec![0.25, 0.75]).expect("valid targets");

        let loss = CrossEntropyLoss::new()
            .forward(&predictions, &targets)
            .expect("loss should compute");

        assert_close(scalar_value(&loss), 2.0_f64.ln());
    }

    #[test]
    fn cross_entropy_matrix_averages_rows() {
        let predictions =
            Tensor::matrix(2, 2, vec![0.0, 0.0, 0.0, 2.0]).expect("valid predictions");
        let targets = Tensor::matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("valid targets");

        let loss = CrossEntropyLoss::new()
            .forward(&predictions, &targets)
            .expect("loss should compute");

        let first_row = 2.0_f64.ln();
        let second_row = (1.0 + (-2.0_f64).exp()).ln();
        assert_close(scalar_value(&loss), (first_row + second_row) / 2.0);
    }

    #[test]
    fn cross_entropy_is_stable_for_large_logits() {
        let predictions = Tensor::vector(vec![1000.0, 1001.0]).expect("valid predictions");
        let targets = Tensor::vector(vec![0.0, 1.0]).expect("valid targets");

        let loss = CrossEntropyLoss::new()
            .forward(&predictions, &targets)
            .expect("loss should compute");

        let actual = scalar_value(&loss);
        assert!(actual.is_finite());
        assert_close(actual, (1.0 + (-1.0_f64).exp()).ln());
    }

    #[test]
    fn cross_entropy_rejects_shape_mismatch() {
        let predictions = Tensor::vector(vec![1.0, 2.0, 3.0]).expect("valid predictions");
        let targets = Tensor::vector(vec![0.0, 1.0]).expect("valid targets");

        let error = CrossEntropyLoss::new()
            .forward(&predictions, &targets)
            .expect_err("shape mismatch should fail");

        assert_eq!(
            error,
            RustGradError::ShapeMismatch {
                op: "cross entropy",
                left: vec![3],
                right: vec![2],
            }
        );
    }

    #[test]
    fn cross_entropy_rejects_rank_three_predictions() {
        let predictions =
            Tensor::from_vec(vec![1, 1, 2], vec![0.0, 1.0]).expect("valid predictions");
        let targets = Tensor::from_vec(vec![1, 1, 2], vec![0.0, 1.0]).expect("valid targets");

        let error = CrossEntropyLoss::new()
            .forward(&predictions, &targets)
            .expect_err("rank three should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "rank",
                reason: "cross entropy supports rank 1 or 2, got rank 3".to_string(),
            }
        );
    }

    #[test]
    fn cross_entropy_rejects_negative_targets() {
        let predictions = Tensor::vector(vec![0.0, 1.0]).expect("valid predictions");
        let targets = Tensor::vector(vec![-0.1, 1.1]).expect("valid targets");

        let error = CrossEntropyLoss::new()
            .forward(&predictions, &targets)
            .expect_err("negative target should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "targets",
                reason: "targets must be finite and non-negative".to_string(),
            }
        );
    }

    #[test]
    fn cross_entropy_rejects_targets_that_do_not_sum_to_one() {
        let predictions = Tensor::vector(vec![0.0, 1.0]).expect("valid predictions");
        let targets = Tensor::vector(vec![0.2, 0.2]).expect("valid targets");

        let error = CrossEntropyLoss::new()
            .forward(&predictions, &targets)
            .expect_err("invalid target sum should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "targets",
                reason: "targets must sum to 1.0, got 0.4".to_string(),
            }
        );
    }
}
