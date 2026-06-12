//! Reusable training loops and training metrics.

use crate::data::Dataset;
use crate::nn::{sigmoid, Linear, Module};
use crate::optim::{GradientSet, Optimizer, SGD};
use crate::tensor::Tensor;
use crate::{Result, RustGradError};

/// Shared configuration for small training examples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingConfig {
    epochs: usize,
    learning_rate: f64,
    log_every: usize,
}

impl TrainingConfig {
    /// Creates a training configuration.
    pub fn new(epochs: usize, learning_rate: f64) -> Result<Self> {
        validate_positive("epochs", epochs)?;
        validate_positive_f64("learning_rate", learning_rate)?;

        Ok(Self {
            epochs,
            learning_rate,
            log_every: 1,
        })
    }

    /// Sets the logging interval in epochs.
    pub fn with_log_every(mut self, log_every: usize) -> Result<Self> {
        validate_positive("log_every", log_every)?;
        self.log_every = log_every;
        Ok(self)
    }

    /// Returns the number of training epochs.
    #[must_use]
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    /// Returns the optimizer learning rate used by the example.
    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the epoch interval used for progress logs.
    #[must_use]
    pub fn log_every(&self) -> usize {
        self.log_every
    }

    /// Returns true when a given epoch should be logged.
    #[must_use]
    pub fn should_log(&self, epoch: usize) -> bool {
        epoch == 1 || epoch == self.epochs || epoch.is_multiple_of(self.log_every)
    }
}

/// Training metrics captured for one epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingRecord {
    epoch: usize,
    loss: f64,
    accuracy: Option<f64>,
}

impl TrainingRecord {
    /// Creates a training record.
    pub fn new(epoch: usize, loss: f64, accuracy: Option<f64>) -> Result<Self> {
        validate_positive("epoch", epoch)?;
        validate_metric("loss", loss)?;
        if let Some(value) = accuracy {
            validate_accuracy(value)?;
        }

        Ok(Self {
            epoch,
            loss,
            accuracy,
        })
    }

    /// Returns the epoch index.
    #[must_use]
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Returns the recorded loss.
    #[must_use]
    pub fn loss(&self) -> f64 {
        self.loss
    }

    /// Returns the optional recorded accuracy.
    #[must_use]
    pub fn accuracy(&self) -> Option<f64> {
        self.accuracy
    }
}

/// Append-only history of training records.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainingHistory {
    records: Vec<TrainingRecord>,
}

impl TrainingHistory {
    /// Creates an empty training history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a training history from records.
    #[must_use]
    pub fn from_records(records: Vec<TrainingRecord>) -> Self {
        Self { records }
    }

    /// Appends one record.
    pub fn push(&mut self, record: TrainingRecord) {
        self.records.push(record);
    }

    /// Returns all records.
    #[must_use]
    pub fn records(&self) -> &[TrainingRecord] {
        &self.records
    }

    /// Returns the number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true when no records are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns the last record.
    #[must_use]
    pub fn last(&self) -> Option<&TrainingRecord> {
        self.records.last()
    }

    /// Returns the first recorded loss.
    #[must_use]
    pub fn initial_loss(&self) -> Option<f64> {
        self.records.first().map(TrainingRecord::loss)
    }

    /// Returns the final recorded loss.
    #[must_use]
    pub fn final_loss(&self) -> Option<f64> {
        self.records.last().map(TrainingRecord::loss)
    }

    /// Returns the lowest recorded loss.
    #[must_use]
    pub fn best_loss(&self) -> Option<f64> {
        self.records
            .iter()
            .map(TrainingRecord::loss)
            .min_by(f64::total_cmp)
    }

    /// Returns the highest recorded accuracy.
    #[must_use]
    pub fn best_accuracy(&self) -> Option<f64> {
        self.records
            .iter()
            .filter_map(TrainingRecord::accuracy)
            .max_by(f64::total_cmp)
    }

    /// Returns true when the final loss is lower than the initial loss.
    #[must_use]
    pub fn loss_decreased(&self) -> bool {
        match (self.initial_loss(), self.final_loss()) {
            (Some(initial), Some(final_loss)) => final_loss < initial,
            _ => false,
        }
    }

    /// Returns all loss values in epoch order.
    #[must_use]
    pub fn losses(&self) -> Vec<f64> {
        self.records.iter().map(TrainingRecord::loss).collect()
    }
}

/// Result produced by a small supervised training run.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearRegressionResult {
    model: Linear,
    history: TrainingHistory,
}

impl LinearRegressionResult {
    /// Creates a result from a trained model and its history.
    #[must_use]
    pub fn new(model: Linear, history: TrainingHistory) -> Self {
        Self { model, history }
    }

    /// Returns the trained linear model.
    #[must_use]
    pub fn model(&self) -> &Linear {
        &self.model
    }

    /// Returns the training history.
    #[must_use]
    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }

    /// Runs prediction with the trained model.
    pub fn predict(&self, features: &Tensor) -> Result<Tensor> {
        self.model.forward(features)
    }
}

/// Result produced by a binary classification training run.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryClassificationResult {
    model: Linear,
    history: TrainingHistory,
    threshold: f64,
}

impl BinaryClassificationResult {
    /// Creates a result from a trained model, its history, and class threshold.
    #[must_use]
    pub fn new(model: Linear, history: TrainingHistory, threshold: f64) -> Self {
        Self {
            model,
            history,
            threshold,
        }
    }

    /// Returns the trained linear model.
    #[must_use]
    pub fn model(&self) -> &Linear {
        &self.model
    }

    /// Returns the training history.
    #[must_use]
    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }

    /// Returns the probability threshold used for class prediction.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Predicts positive-class probabilities.
    pub fn predict_proba(&self, features: &Tensor) -> Result<Tensor> {
        sigmoid(&self.model.forward(features)?)
    }

    /// Predicts binary classes as `0.0` or `1.0`.
    pub fn predict_classes(&self, features: &Tensor) -> Result<Tensor> {
        let probabilities = self.predict_proba(features)?;
        threshold_probabilities(&probabilities, self.threshold)
    }
}

/// Trains a linear layer on a supervised regression dataset using MSE and SGD.
///
/// This loop intentionally keeps the gradient formula explicit so the example
/// remains easy to inspect in reports: it computes the MSE gradient for
/// `output = input @ weights + bias`, then delegates the parameter update to
/// the optimizer module.
pub fn train_linear_regression(
    dataset: &Dataset,
    config: TrainingConfig,
) -> Result<LinearRegressionResult> {
    let mut model = Linear::new(dataset.input_size(), dataset.target_size())?;
    let mut optimizer = SGD::new(config.learning_rate())?;
    let mut history = TrainingHistory::new();

    for epoch in 1..=config.epochs() {
        let predictions = model.forward(dataset.features())?;
        let (weight_grad, bias_grad) =
            linear_mse_gradients(dataset.features(), &predictions, dataset.targets())?;
        let gradients = GradientSet::from_tensors(vec![weight_grad, bias_grad]);

        {
            let mut parameters = model.parameters_mut();
            optimizer.step(&mut parameters, &gradients)?;
        }

        let updated_predictions = model.forward(dataset.features())?;
        let loss = mean_squared_error(&updated_predictions, dataset.targets())?;
        history.push(TrainingRecord::new(epoch, loss, None)?);
    }

    Ok(LinearRegressionResult::new(model, history))
}

/// Trains a logistic regression classifier using binary cross entropy and SGD.
///
/// Targets must be a single-column matrix containing `0.0` or `1.0`. The
/// recorded accuracy is computed from probabilities after each update.
pub fn train_binary_classification(
    dataset: &Dataset,
    config: TrainingConfig,
    threshold: f64,
) -> Result<BinaryClassificationResult> {
    validate_finite("threshold", threshold)?;
    validate_binary_classification_dataset(dataset)?;

    let mut model = Linear::new(dataset.input_size(), 1)?;
    let mut optimizer = SGD::new(config.learning_rate())?;
    let mut history = TrainingHistory::new();

    for epoch in 1..=config.epochs() {
        let logits = model.forward(dataset.features())?;
        let probabilities = sigmoid(&logits)?;
        let (weight_grad, bias_grad) =
            logistic_binary_gradients(dataset.features(), &probabilities, dataset.targets())?;
        let gradients = GradientSet::from_tensors(vec![weight_grad, bias_grad]);

        {
            let mut parameters = model.parameters_mut();
            optimizer.step(&mut parameters, &gradients)?;
        }

        let updated_probabilities = sigmoid(&model.forward(dataset.features())?)?;
        let loss = binary_cross_entropy(&updated_probabilities, dataset.targets())?;
        let accuracy = binary_accuracy(&updated_probabilities, dataset.targets(), threshold)?;
        history.push(TrainingRecord::new(epoch, loss, Some(accuracy))?);
    }

    Ok(BinaryClassificationResult::new(model, history, threshold))
}

/// Computes mean squared error between two tensors.
pub fn mean_squared_error(predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    ensure_same_shape("mean squared error", predictions, targets)?;
    let total: f64 = predictions
        .data()
        .iter()
        .zip(targets.data())
        .map(|(&prediction, &target)| {
            let diff = prediction - target;
            diff * diff
        })
        .sum();

    Ok(total / predictions.len() as f64)
}

/// Computes mean absolute error between two tensors.
pub fn mean_absolute_error(predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    ensure_same_shape("mean absolute error", predictions, targets)?;
    let total: f64 = predictions
        .data()
        .iter()
        .zip(targets.data())
        .map(|(&prediction, &target)| (prediction - target).abs())
        .sum();

    Ok(total / predictions.len() as f64)
}

/// Computes binary classification accuracy after thresholding predictions.
pub fn binary_accuracy(predictions: &Tensor, targets: &Tensor, threshold: f64) -> Result<f64> {
    ensure_same_shape("binary accuracy", predictions, targets)?;
    validate_finite("threshold", threshold)?;

    let correct = predictions
        .data()
        .iter()
        .zip(targets.data())
        .filter(|&(&prediction, &target)| {
            let predicted_class = if prediction >= threshold { 1.0 } else { 0.0 };
            (predicted_class - target).abs() < f64::EPSILON
        })
        .count();

    Ok(correct as f64 / predictions.len() as f64)
}

/// Computes categorical accuracy for batched class scores and one-hot targets.
pub fn categorical_accuracy(predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    ensure_same_shape("categorical accuracy", predictions, targets)?;
    validate_rank_two("predictions", predictions)?;
    validate_rank_two("targets", targets)?;

    let rows = predictions.rows().expect("rank 2 tensors always have rows");
    let cols = predictions
        .cols()
        .expect("rank 2 tensors always have columns");
    let mut correct = 0;

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let predicted = argmax(&predictions.data()[start..end]);
        let target = argmax(&targets.data()[start..end]);
        if predicted == target {
            correct += 1;
        }
    }

    Ok(correct as f64 / rows as f64)
}

fn binary_cross_entropy(probabilities: &Tensor, targets: &Tensor) -> Result<f64> {
    ensure_same_shape("binary cross entropy", probabilities, targets)?;
    validate_binary_targets(targets)?;

    let epsilon = 1e-12;
    let total: f64 = probabilities
        .data()
        .iter()
        .zip(targets.data())
        .map(|(&probability, &target)| {
            let probability = probability.clamp(epsilon, 1.0 - epsilon);
            -(target * probability.ln() + (1.0 - target) * (1.0 - probability).ln())
        })
        .sum();

    Ok(total / probabilities.len() as f64)
}

fn threshold_probabilities(probabilities: &Tensor, threshold: f64) -> Result<Tensor> {
    validate_finite("threshold", threshold)?;
    Tensor::from_vec(
        probabilities.shape().to_vec(),
        probabilities
            .data()
            .iter()
            .map(
                |&probability| {
                    if probability >= threshold {
                        1.0
                    } else {
                        0.0
                    }
                },
            )
            .collect(),
    )
}

fn linear_mse_gradients(
    features: &Tensor,
    predictions: &Tensor,
    targets: &Tensor,
) -> Result<(Tensor, Tensor)> {
    validate_rank_two("features", features)?;
    validate_rank_two("predictions", predictions)?;
    validate_rank_two("targets", targets)?;
    ensure_same_shape("linear regression targets", predictions, targets)?;

    let rows = features.rows().expect("rank 2 tensors always have rows");
    let input_size = features.cols().expect("rank 2 tensors always have columns");
    let output_size = predictions
        .cols()
        .expect("rank 2 tensors always have columns");
    let scale = 2.0 / predictions.len() as f64;
    let mut weight_grad = vec![0.0; input_size * output_size];
    let mut bias_grad = vec![0.0; output_size];

    for row in 0..rows {
        for output_col in 0..output_size {
            let prediction = predictions.get(&[row, output_col])?;
            let target = targets.get(&[row, output_col])?;
            let error = prediction - target;
            let scaled_error = scale * error;
            bias_grad[output_col] += scaled_error;

            for input_col in 0..input_size {
                let feature = features.get(&[row, input_col])?;
                weight_grad[input_col * output_size + output_col] += feature * scaled_error;
            }
        }
    }

    Ok((
        Tensor::matrix(input_size, output_size, weight_grad)?,
        Tensor::vector(bias_grad)?,
    ))
}

fn logistic_binary_gradients(
    features: &Tensor,
    probabilities: &Tensor,
    targets: &Tensor,
) -> Result<(Tensor, Tensor)> {
    validate_rank_two("features", features)?;
    validate_rank_two("probabilities", probabilities)?;
    validate_rank_two("targets", targets)?;
    ensure_same_shape("binary classification targets", probabilities, targets)?;
    validate_binary_targets(targets)?;

    if probabilities.cols() != Some(1) {
        return Err(RustGradError::InvalidArgument {
            name: "probabilities",
            reason: format!(
                "binary classification expects one output column, got {}",
                probabilities
                    .cols()
                    .expect("rank 2 tensors always have columns")
            ),
        });
    }

    let rows = features.rows().expect("rank 2 tensors always have rows");
    let input_size = features.cols().expect("rank 2 tensors always have columns");
    let scale = 1.0 / rows as f64;
    let mut weight_grad = vec![0.0; input_size];
    let mut bias_grad = 0.0;

    for row in 0..rows {
        let error = probabilities.get(&[row, 0])? - targets.get(&[row, 0])?;
        bias_grad += error * scale;

        for (input_col, gradient) in weight_grad.iter_mut().enumerate().take(input_size) {
            *gradient += features.get(&[row, input_col])? * error * scale;
        }
    }

    Ok((
        Tensor::matrix(input_size, 1, weight_grad)?,
        Tensor::vector(vec![bias_grad])?,
    ))
}

fn validate_binary_classification_dataset(dataset: &Dataset) -> Result<()> {
    if dataset.target_size() != 1 {
        return Err(RustGradError::InvalidArgument {
            name: "targets",
            reason: format!(
                "binary classification expects one target column, got {}",
                dataset.target_size()
            ),
        });
    }

    validate_binary_targets(dataset.targets())
}

fn validate_binary_targets(targets: &Tensor) -> Result<()> {
    validate_rank_two("targets", targets)?;

    if targets
        .data()
        .iter()
        .any(|target| (*target - 0.0).abs() > f64::EPSILON && (*target - 1.0).abs() > f64::EPSILON)
    {
        return Err(RustGradError::InvalidArgument {
            name: "targets",
            reason: "binary targets must contain only 0.0 or 1.0".to_string(),
        });
    }

    Ok(())
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("dataset tensors never have zero columns")
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

fn validate_rank_two(name: &'static str, tensor: &Tensor) -> Result<()> {
    if tensor.rank() == 2 {
        Ok(())
    } else {
        Err(RustGradError::InvalidArgument {
            name,
            reason: format!("expected rank 2 tensor, got rank {}", tensor.rank()),
        })
    }
}

fn validate_positive(name: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        Err(RustGradError::InvalidArgument {
            name,
            reason: "value must be greater than zero".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_positive_f64(name: &'static str, value: f64) -> Result<()> {
    if value <= 0.0 || !value.is_finite() {
        Err(RustGradError::InvalidArgument {
            name,
            reason: "value must be finite and greater than zero".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_metric(name: &'static str, value: f64) -> Result<()> {
    if value < 0.0 || !value.is_finite() {
        Err(RustGradError::InvalidArgument {
            name,
            reason: "metric must be finite and non-negative".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_accuracy(value: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&value) || !value.is_finite() {
        Err(RustGradError::InvalidArgument {
            name: "accuracy",
            reason: "accuracy must be finite and in [0, 1]".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_finite(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(RustGradError::InvalidArgument {
            name,
            reason: "value must be finite".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        binary_accuracy, categorical_accuracy, mean_absolute_error, mean_squared_error,
        train_binary_classification, train_linear_regression, TrainingConfig, TrainingHistory,
        TrainingRecord,
    };
    use crate::data::{linear_regression, Dataset};
    use crate::tensor::Tensor;
    use crate::RustGradError;

    const EPSILON: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        assert_close_with(actual, expected, EPSILON);
    }

    fn assert_close_with(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn training_config_stores_values_and_default_logging() {
        let config = TrainingConfig::new(10, 0.05).expect("valid config");

        assert_eq!(config.epochs(), 10);
        assert_eq!(config.learning_rate(), 0.05);
        assert_eq!(config.log_every(), 1);
        assert!(config.should_log(1));
        assert!(config.should_log(10));
        assert!(config.should_log(7));
    }

    #[test]
    fn training_config_supports_custom_logging_interval() {
        let config = TrainingConfig::new(10, 0.05)
            .expect("valid config")
            .with_log_every(3)
            .expect("valid log interval");

        assert_eq!(config.log_every(), 3);
        assert!(config.should_log(1));
        assert!(config.should_log(3));
        assert!(config.should_log(6));
        assert!(config.should_log(10));
        assert!(!config.should_log(2));
        assert!(!config.should_log(4));
    }

    #[test]
    fn training_config_rejects_zero_epochs() {
        let error = TrainingConfig::new(0, 0.1).expect_err("zero epochs should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "epochs",
                reason: "value must be greater than zero".to_string(),
            }
        );
    }

    #[test]
    fn training_config_rejects_invalid_learning_rate() {
        let error =
            TrainingConfig::new(10, f64::INFINITY).expect_err("invalid learning rate should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "learning_rate",
                reason: "value must be finite and greater than zero".to_string(),
            }
        );
    }

    #[test]
    fn training_config_rejects_zero_log_interval() {
        let error = TrainingConfig::new(10, 0.1)
            .expect("valid config")
            .with_log_every(0)
            .expect_err("zero log interval should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "log_every",
                reason: "value must be greater than zero".to_string(),
            }
        );
    }

    #[test]
    fn training_record_stores_epoch_loss_and_accuracy() {
        let record = TrainingRecord::new(3, 0.25, Some(0.75)).expect("valid record");

        assert_eq!(record.epoch(), 3);
        assert_eq!(record.loss(), 0.25);
        assert_eq!(record.accuracy(), Some(0.75));
    }

    #[test]
    fn training_record_allows_missing_accuracy() {
        let record = TrainingRecord::new(1, 1.5, None).expect("valid record");

        assert_eq!(record.accuracy(), None);
    }

    #[test]
    fn training_record_rejects_invalid_epoch() {
        let error = TrainingRecord::new(0, 1.0, None).expect_err("zero epoch should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "epoch",
                reason: "value must be greater than zero".to_string(),
            }
        );
    }

    #[test]
    fn training_record_rejects_invalid_loss() {
        let error = TrainingRecord::new(1, -1.0, None).expect_err("negative loss should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "loss",
                reason: "metric must be finite and non-negative".to_string(),
            }
        );
    }

    #[test]
    fn training_record_rejects_invalid_accuracy() {
        let error =
            TrainingRecord::new(1, 1.0, Some(1.5)).expect_err("accuracy above one should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "accuracy",
                reason: "accuracy must be finite and in [0, 1]".to_string(),
            }
        );
    }

    #[test]
    fn training_history_starts_empty() {
        let history = TrainingHistory::new();

        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert!(history.last().is_none());
        assert_eq!(history.initial_loss(), None);
        assert_eq!(history.final_loss(), None);
        assert_eq!(history.best_loss(), None);
        assert_eq!(history.best_accuracy(), None);
        assert!(!history.loss_decreased());
    }

    #[test]
    fn training_history_tracks_loss_and_accuracy() {
        let mut history = TrainingHistory::new();
        history.push(TrainingRecord::new(1, 2.0, Some(0.25)).expect("valid record"));
        history.push(TrainingRecord::new(2, 1.5, Some(0.5)).expect("valid record"));
        history.push(TrainingRecord::new(3, 1.0, Some(0.75)).expect("valid record"));

        assert_eq!(history.len(), 3);
        assert_eq!(history.records()[0].epoch(), 1);
        assert_eq!(history.last().map(TrainingRecord::epoch), Some(3));
        assert_eq!(history.initial_loss(), Some(2.0));
        assert_eq!(history.final_loss(), Some(1.0));
        assert_eq!(history.best_loss(), Some(1.0));
        assert_eq!(history.best_accuracy(), Some(0.75));
        assert_eq!(history.losses(), vec![2.0, 1.5, 1.0]);
        assert!(history.loss_decreased());
    }

    #[test]
    fn training_history_can_be_created_from_records() {
        let records = vec![
            TrainingRecord::new(1, 0.8, None).expect("valid record"),
            TrainingRecord::new(2, 0.9, None).expect("valid record"),
        ];

        let history = TrainingHistory::from_records(records);

        assert_eq!(history.len(), 2);
        assert_eq!(history.best_loss(), Some(0.8));
        assert!(!history.loss_decreased());
    }

    #[test]
    fn train_linear_regression_records_one_loss_per_epoch() {
        let dataset = linear_regression(9, 2.0, 1.0).expect("valid dataset");
        let config = TrainingConfig::new(12, 0.1).expect("valid config");

        let result = train_linear_regression(&dataset, config).expect("training should succeed");

        assert_eq!(result.history().len(), 12);
        assert_eq!(result.history().records()[0].epoch(), 1);
        assert_eq!(result.history().last().map(TrainingRecord::epoch), Some(12));
        assert!(result
            .history()
            .records()
            .iter()
            .all(|record| record.accuracy().is_none()));
    }

    #[test]
    fn train_linear_regression_decreases_loss_on_simple_line() {
        let dataset = linear_regression(21, 2.0, -0.5).expect("valid dataset");
        let config = TrainingConfig::new(80, 0.15).expect("valid config");

        let result = train_linear_regression(&dataset, config).expect("training should succeed");
        let history = result.history();

        assert!(history.loss_decreased());
        assert!(
            history.final_loss().expect("final loss exists")
                < history.initial_loss().expect("initial loss exists") * 0.05,
            "expected strong loss decrease, got {:?}",
            history.losses()
        );
        assert!(history.best_loss().expect("best loss exists") <= history.final_loss().unwrap());
    }

    #[test]
    fn train_linear_regression_learns_slope_and_intercept() {
        let dataset = linear_regression(31, 1.5, 0.75).expect("valid dataset");
        let config = TrainingConfig::new(160, 0.12).expect("valid config");

        let result = train_linear_regression(&dataset, config).expect("training should succeed");

        assert_close_with(
            result
                .model()
                .weights()
                .get(&[0, 0])
                .expect("weight exists"),
            1.5,
            1e-4,
        );
        assert_close_with(
            result.model().bias().get_flat(0).expect("bias exists"),
            0.75,
            1e-4,
        );
        assert!(
            result.history().final_loss().expect("final loss exists") < 1e-6,
            "expected tiny final loss, got {:?}",
            result.history().final_loss()
        );
    }

    #[test]
    fn linear_regression_result_predicts_with_trained_model() {
        let dataset = linear_regression(25, -3.0, 2.0).expect("valid dataset");
        let config = TrainingConfig::new(180, 0.1).expect("valid config");
        let result = train_linear_regression(&dataset, config).expect("training should succeed");
        let inputs = Tensor::matrix(3, 1, vec![-1.0, 0.0, 1.0]).expect("valid input");

        let predictions = result.predict(&inputs).expect("prediction should succeed");

        assert_eq!(predictions.dims(), &[3, 1]);
        assert_close_with(
            predictions.get(&[0, 0]).expect("prediction exists"),
            5.0,
            1e-4,
        );
        assert_close_with(
            predictions.get(&[1, 0]).expect("prediction exists"),
            2.0,
            1e-4,
        );
        assert_close_with(
            predictions.get(&[2, 0]).expect("prediction exists"),
            -1.0,
            1e-4,
        );
    }

    #[test]
    fn train_linear_regression_supports_multiple_outputs() {
        let dataset = Dataset::new(
            "multi-output",
            Tensor::matrix(4, 1, vec![-1.0, 0.0, 1.0, 2.0]).expect("valid features"),
            Tensor::matrix(4, 2, vec![-1.0, 3.0, 1.0, 1.0, 3.0, -1.0, 5.0, -3.0])
                .expect("valid targets"),
        )
        .expect("valid dataset");
        let config = TrainingConfig::new(220, 0.08).expect("valid config");

        let result = train_linear_regression(&dataset, config).expect("training should succeed");

        assert_eq!(result.model().weights().dims(), &[1, 2]);
        assert_eq!(result.model().bias().dims(), &[2]);
        assert_close_with(
            result
                .model()
                .weights()
                .get(&[0, 0])
                .expect("weight exists"),
            2.0,
            1e-4,
        );
        assert_close_with(
            result
                .model()
                .weights()
                .get(&[0, 1])
                .expect("weight exists"),
            -2.0,
            1e-4,
        );
        assert_close_with(
            result.model().bias().get_flat(0).expect("bias exists"),
            1.0,
            1e-4,
        );
        assert_close_with(
            result.model().bias().get_flat(1).expect("bias exists"),
            1.0,
            1e-4,
        );
        assert!(result.history().final_loss().expect("final loss exists") < 1e-6);
    }

    #[test]
    fn train_binary_classification_records_loss_and_accuracy() {
        let dataset = Dataset::new(
            "threshold",
            Tensor::matrix(4, 1, vec![-2.0, -1.0, 1.0, 2.0]).expect("valid features"),
            Tensor::matrix(4, 1, vec![0.0, 0.0, 1.0, 1.0]).expect("valid targets"),
        )
        .expect("valid dataset");
        let config = TrainingConfig::new(20, 0.4).expect("valid config");

        let result =
            train_binary_classification(&dataset, config, 0.5).expect("training should succeed");

        assert_eq!(result.history().len(), 20);
        assert_eq!(result.history().records()[0].epoch(), 1);
        assert!(result
            .history()
            .records()
            .iter()
            .all(|record| record.accuracy().is_some()));
        assert_eq!(result.threshold(), 0.5);
    }

    #[test]
    fn train_binary_classification_decreases_loss_and_reaches_perfect_accuracy() {
        let dataset = Dataset::new(
            "threshold",
            Tensor::matrix(6, 1, vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).expect("valid features"),
            Tensor::matrix(6, 1, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("valid targets"),
        )
        .expect("valid dataset");
        let config = TrainingConfig::new(80, 0.5).expect("valid config");

        let result =
            train_binary_classification(&dataset, config, 0.5).expect("training should succeed");

        assert!(result.history().loss_decreased());
        assert!(
            result.history().final_loss().expect("final loss exists")
                < result
                    .history()
                    .initial_loss()
                    .expect("initial loss exists")
                    * 0.25,
            "expected classification loss to decrease strongly, got {:?}",
            result.history().losses()
        );
        assert_eq!(result.history().best_accuracy(), Some(1.0));
        assert!(
            result
                .model()
                .weights()
                .get(&[0, 0])
                .expect("weight exists")
                > 0.0
        );
    }

    #[test]
    fn binary_classification_result_predicts_probabilities_and_classes() {
        let dataset = Dataset::new(
            "threshold",
            Tensor::matrix(6, 1, vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).expect("valid features"),
            Tensor::matrix(6, 1, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("valid targets"),
        )
        .expect("valid dataset");
        let config = TrainingConfig::new(100, 0.5).expect("valid config");
        let result =
            train_binary_classification(&dataset, config, 0.5).expect("training should succeed");
        let inputs = Tensor::matrix(3, 1, vec![-2.0, 0.0, 2.0]).expect("valid inputs");

        let probabilities = result
            .predict_proba(&inputs)
            .expect("probabilities should compute");
        let classes = result
            .predict_classes(&inputs)
            .expect("classes should compute");

        assert_eq!(probabilities.dims(), &[3, 1]);
        assert_eq!(classes.dims(), &[3, 1]);
        assert!(probabilities.get(&[0, 0]).expect("probability exists") < 0.5);
        assert!(probabilities.get(&[2, 0]).expect("probability exists") > 0.5);
        assert_eq!(classes.data(), &[0.0, 1.0, 1.0]);
    }

    #[test]
    fn train_binary_classification_rejects_multi_column_targets() {
        let dataset = Dataset::new(
            "bad-targets",
            Tensor::matrix(2, 1, vec![-1.0, 1.0]).expect("valid features"),
            Tensor::matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("valid targets"),
        )
        .expect("valid dataset");
        let config = TrainingConfig::new(10, 0.1).expect("valid config");

        let error = train_binary_classification(&dataset, config, 0.5)
            .expect_err("multi-column targets should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "targets",
                reason: "binary classification expects one target column, got 2".to_string(),
            }
        );
    }

    #[test]
    fn train_binary_classification_rejects_non_binary_targets() {
        let dataset = Dataset::new(
            "bad-targets",
            Tensor::matrix(2, 1, vec![-1.0, 1.0]).expect("valid features"),
            Tensor::matrix(2, 1, vec![0.0, 0.5]).expect("valid targets"),
        )
        .expect("valid dataset");
        let config = TrainingConfig::new(10, 0.1).expect("valid config");

        let error = train_binary_classification(&dataset, config, 0.5)
            .expect_err("non-binary targets should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "targets",
                reason: "binary targets must contain only 0.0 or 1.0".to_string(),
            }
        );
    }

    #[test]
    fn train_binary_classification_rejects_invalid_threshold() {
        let dataset = Dataset::new(
            "threshold",
            Tensor::matrix(2, 1, vec![-1.0, 1.0]).expect("valid features"),
            Tensor::matrix(2, 1, vec![0.0, 1.0]).expect("valid targets"),
        )
        .expect("valid dataset");
        let config = TrainingConfig::new(10, 0.1).expect("valid config");

        let error = train_binary_classification(&dataset, config, f64::NAN)
            .expect_err("nan threshold should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "threshold",
                reason: "value must be finite".to_string(),
            }
        );
    }

    #[test]
    fn mean_squared_error_computes_average_squared_difference() {
        let predictions = Tensor::vector(vec![1.0, 2.0, 4.0]).expect("valid predictions");
        let targets = Tensor::vector(vec![1.0, 1.0, 1.0]).expect("valid targets");

        let value = mean_squared_error(&predictions, &targets).expect("mse should compute");

        assert_close(value, 10.0 / 3.0);
    }

    #[test]
    fn mean_absolute_error_computes_average_absolute_difference() {
        let predictions = Tensor::vector(vec![1.0, 2.0, 4.0]).expect("valid predictions");
        let targets = Tensor::vector(vec![1.0, 1.0, 1.0]).expect("valid targets");

        let value = mean_absolute_error(&predictions, &targets).expect("mae should compute");

        assert_close(value, 4.0 / 3.0);
    }

    #[test]
    fn metric_functions_reject_shape_mismatch() {
        let predictions = Tensor::vector(vec![1.0, 2.0]).expect("valid predictions");
        let targets = Tensor::matrix(1, 2, vec![1.0, 2.0]).expect("valid targets");

        assert_eq!(
            mean_squared_error(&predictions, &targets).expect_err("shape mismatch should fail"),
            RustGradError::ShapeMismatch {
                op: "mean squared error",
                left: vec![2],
                right: vec![1, 2],
            }
        );
        assert_eq!(
            mean_absolute_error(&predictions, &targets).expect_err("shape mismatch should fail"),
            RustGradError::ShapeMismatch {
                op: "mean absolute error",
                left: vec![2],
                right: vec![1, 2],
            }
        );
    }

    #[test]
    fn binary_accuracy_thresholds_predictions() {
        let predictions = Tensor::vector(vec![0.1, 0.6, 0.8, 0.2]).expect("valid predictions");
        let targets = Tensor::vector(vec![0.0, 1.0, 0.0, 0.0]).expect("valid targets");

        let value = binary_accuracy(&predictions, &targets, 0.5).expect("accuracy should compute");

        assert_close(value, 0.75);
    }

    #[test]
    fn binary_accuracy_rejects_invalid_threshold() {
        let predictions = Tensor::vector(vec![0.1]).expect("valid predictions");
        let targets = Tensor::vector(vec![0.0]).expect("valid targets");

        let error = binary_accuracy(&predictions, &targets, f64::NAN)
            .expect_err("nan threshold should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "threshold",
                reason: "value must be finite".to_string(),
            }
        );
    }

    #[test]
    fn binary_accuracy_rejects_shape_mismatch() {
        let predictions = Tensor::vector(vec![0.1, 0.6]).expect("valid predictions");
        let targets = Tensor::vector(vec![0.0]).expect("valid targets");

        let error =
            binary_accuracy(&predictions, &targets, 0.5).expect_err("shape mismatch should fail");

        assert_eq!(
            error,
            RustGradError::ShapeMismatch {
                op: "binary accuracy",
                left: vec![2],
                right: vec![1],
            }
        );
    }

    #[test]
    fn categorical_accuracy_compares_argmax_classes() {
        let predictions = Tensor::matrix(3, 3, vec![0.1, 0.8, 0.1, 0.4, 0.3, 0.3, 0.2, 0.2, 0.6])
            .expect("valid predictions");
        let targets = Tensor::matrix(3, 3, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .expect("valid targets");

        let value = categorical_accuracy(&predictions, &targets).expect("accuracy should compute");

        assert_close(value, 2.0 / 3.0);
    }

    #[test]
    fn categorical_accuracy_rejects_shape_mismatch() {
        let predictions = Tensor::matrix(1, 2, vec![0.1, 0.9]).expect("valid predictions");
        let targets = Tensor::matrix(1, 3, vec![0.0, 1.0, 0.0]).expect("valid targets");

        let error =
            categorical_accuracy(&predictions, &targets).expect_err("shape mismatch should fail");

        assert_eq!(
            error,
            RustGradError::ShapeMismatch {
                op: "categorical accuracy",
                left: vec![1, 2],
                right: vec![1, 3],
            }
        );
    }

    #[test]
    fn categorical_accuracy_rejects_rank_one_tensors() {
        let predictions = Tensor::vector(vec![0.1, 0.9]).expect("valid predictions");
        let targets = Tensor::vector(vec![0.0, 1.0]).expect("valid targets");

        let error = categorical_accuracy(&predictions, &targets).expect_err("rank one should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "predictions",
                reason: "expected rank 2 tensor, got rank 1".to_string(),
            }
        );
    }
}
