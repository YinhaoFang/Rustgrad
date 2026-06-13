//! Reusable training loops and training metrics.

use crate::autograd::{ComputationGraph, Operation};
use crate::data::{spiral, xor, Dataset};
use crate::loss::{CrossEntropyLoss, Loss};
use crate::nn::{sigmoid, softmax, Linear, Module};
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

/// A tiny two-layer sigmoid MLP specialized for XOR-style binary tasks.
#[derive(Debug, Clone, PartialEq)]
pub struct XorMlp {
    hidden: Linear,
    output: Linear,
    threshold: f64,
}

impl XorMlp {
    /// Creates a deterministic 2-2-1 sigmoid MLP.
    pub fn new(threshold: f64) -> Result<Self> {
        validate_finite("threshold", threshold)?;

        Ok(Self {
            hidden: Linear::with_parameters(
                Tensor::matrix(2, 2, vec![4.0, -4.0, 4.0, -4.0])?,
                Tensor::vector(vec![-2.0, 6.0])?,
            )?,
            output: Linear::with_parameters(
                Tensor::matrix(2, 1, vec![4.0, 4.0])?,
                Tensor::vector(vec![-6.0])?,
            )?,
            threshold,
        })
    }

    /// Creates an XOR MLP from explicit weight and bias tensors.
    pub fn from_parameters(
        hidden_weights: Tensor,
        hidden_bias: Tensor,
        output_weights: Tensor,
        output_bias: Tensor,
        threshold: f64,
    ) -> Result<Self> {
        validate_finite("threshold", threshold)?;
        Ok(Self {
            hidden: Linear::with_parameters(hidden_weights, hidden_bias)?,
            output: Linear::with_parameters(output_weights, output_bias)?,
            threshold,
        })
    }

    /// Returns the hidden linear layer.
    #[must_use]
    pub fn hidden(&self) -> &Linear {
        &self.hidden
    }

    /// Returns the output linear layer.
    #[must_use]
    pub fn output(&self) -> &Linear {
        &self.output
    }

    /// Returns the probability threshold used for class prediction.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Computes hidden activations.
    pub fn hidden_activations(&self, features: &Tensor) -> Result<Tensor> {
        sigmoid(&self.hidden.forward(features)?)
    }

    /// Predicts positive-class probabilities.
    pub fn predict_proba(&self, features: &Tensor) -> Result<Tensor> {
        let hidden = self.hidden_activations(features)?;
        sigmoid(&self.output.forward(&hidden)?)
    }

    /// Predicts binary classes as `0.0` or `1.0`.
    pub fn predict_classes(&self, features: &Tensor) -> Result<Tensor> {
        let probabilities = self.predict_proba(features)?;
        threshold_probabilities(&probabilities, self.threshold)
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut hidden_parameters = self.hidden.parameters_mut();
        let mut output_parameters = self.output.parameters_mut();
        hidden_parameters.append(&mut output_parameters);
        hidden_parameters
    }
}

/// Result produced by XOR MLP training.
#[derive(Debug, Clone, PartialEq)]
pub struct XorTrainingResult {
    model: XorMlp,
    history: TrainingHistory,
}

impl XorTrainingResult {
    /// Creates a result from a trained XOR model and its history.
    #[must_use]
    pub fn new(model: XorMlp, history: TrainingHistory) -> Self {
        Self { model, history }
    }

    /// Returns the trained XOR MLP.
    #[must_use]
    pub fn model(&self) -> &XorMlp {
        &self.model
    }

    /// Returns the training history.
    #[must_use]
    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }

    /// Predicts positive-class probabilities.
    pub fn predict_proba(&self, features: &Tensor) -> Result<Tensor> {
        self.model.predict_proba(features)
    }

    /// Predicts binary classes as `0.0` or `1.0`.
    pub fn predict_classes(&self, features: &Tensor) -> Result<Tensor> {
        self.model.predict_classes(features)
    }
}

/// Result produced by spiral multi-class classifier training.
#[derive(Debug, Clone, PartialEq)]
pub struct SpiralTrainingResult {
    model: Linear,
    history: TrainingHistory,
    classes: usize,
}

impl SpiralTrainingResult {
    /// Creates a result from a trained softmax classifier and its history.
    #[must_use]
    pub fn new(model: Linear, history: TrainingHistory, classes: usize) -> Self {
        Self {
            model,
            history,
            classes,
        }
    }

    /// Returns the trained linear classifier.
    #[must_use]
    pub fn model(&self) -> &Linear {
        &self.model
    }

    /// Returns the training history.
    #[must_use]
    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }

    /// Returns the number of output classes.
    #[must_use]
    pub fn classes(&self) -> usize {
        self.classes
    }

    /// Predicts class probabilities for raw two-dimensional spiral features.
    pub fn predict_proba(&self, features: &Tensor) -> Result<Tensor> {
        let mapped = spiral_feature_map(features)?;
        softmax(&self.model.forward(&mapped)?)
    }

    /// Predicts classes as one-hot rows.
    pub fn predict_classes(&self, features: &Tensor) -> Result<Tensor> {
        probabilities_to_one_hot(&self.predict_proba(features)?)
    }
}

/// Trains a linear layer on a supervised regression dataset using MSE and SGD.
///
/// Builds a computation graph each epoch, computes MSE loss manually, then
/// seeds the autograd engine with the MSE gradient to propagate into weights
/// and bias.
pub fn train_linear_regression(
    dataset: &Dataset,
    config: TrainingConfig,
) -> Result<LinearRegressionResult> {
    let mut model = Linear::new(dataset.input_size(), dataset.target_size())?;
    let mut optimizer = SGD::new(config.learning_rate())?;
    let mut history = TrainingHistory::new();
    let features = dataset.features();
    let targets = dataset.targets();

    for epoch in 1..=config.epochs() {
        let (weight_grad, bias_grad) = {
            let mut graph = ComputationGraph::new();
            // Constant leaves: features.
            let f_node = graph.add_leaf(features.clone(), false);
            // Trainable leaves: weights, bias.
            let trainable = model.parameters();
            let w_node = graph.add_leaf(trainable[0].clone(), true);
            let b_node = graph.add_leaf(trainable[1].clone(), true);

            // predictions = features @ weights
            let matmul_val = features.matmul(trainable[0])?;
            let matmul_node =
                graph.add_operation(Operation::MatMul, vec![f_node, w_node], matmul_val, true)?;
            // predictions = matmul + bias (row_add)
            let pred_val = graph
                .node(matmul_node)
                .expect("exists")
                .value()
                .row_add(trainable[1])?;
            let n = pred_val.len() as f64;
            let pred_shape = pred_val.shape().to_vec();
            let pred_data = pred_val.data().to_vec();
            let pred_node = graph.add_operation(
                Operation::RowAdd,
                vec![matmul_node, b_node],
                pred_val,
                true,
            )?;

            // MSE gradient: dL/dy = (2/N) * (y - target)
            let scale = 2.0 / n;
            let init_grad_data: Vec<f64> = pred_data
                .iter()
                .zip(targets.data())
                .map(|(&p, &t)| scale * (p - t))
                .collect();
            let init_grad = Tensor::from_vec(pred_shape, init_grad_data)?;
            graph.backward_with_grad(pred_node, init_grad)?;

            let grads = graph.take_gradients();
            (grads[0].clone(), grads[1].clone())
        };

        {
            let gradients = GradientSet::from_tensors(vec![weight_grad, bias_grad]);
            let mut parameters = model.parameters_mut();
            optimizer.step(&mut parameters, &gradients)?;
        }

        let predictions = model.forward(features)?;
        let loss = mean_squared_error(&predictions, targets)?;
        history.push(TrainingRecord::new(epoch, loss, None)?);
    }

    Ok(LinearRegressionResult::new(model, history))
}

/// Trains a logistic regression classifier using binary cross entropy and SGD.
///
/// Builds a computation graph each epoch, seeds the logits node with the
/// sigmoid+BCE combined gradient `(p - t) / N`, and propagates to weights.
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
    let features = dataset.features();
    let targets = dataset.targets();

    for epoch in 1..=config.epochs() {
        let (weight_grad, bias_grad) = {
            let mut graph = ComputationGraph::new();
            let f_node = graph.add_leaf(features.clone(), false);
            let trainable = model.parameters();
            let w_node = graph.add_leaf(trainable[0].clone(), true);
            let b_node = graph.add_leaf(trainable[1].clone(), true);

            let matmul_val = features.matmul(trainable[0])?;
            let matmul_node =
                graph.add_operation(Operation::MatMul, vec![f_node, w_node], matmul_val, true)?;
            let logits_val = graph
                .node(matmul_node)
                .expect("exists")
                .value()
                .row_add(trainable[1])?;
            let n = logits_val.len() as f64;
            let logits_node = graph.add_operation(
                Operation::RowAdd,
                vec![matmul_node, b_node],
                logits_val,
                true,
            )?;

            // sigmoid + BCE combined: dL/dlogits = (sigma(logits) - t) / N
            let logits_data = graph
                .node(logits_node)
                .expect("exists")
                .value()
                .data()
                .to_vec();
            let logits_shape = graph
                .node(logits_node)
                .expect("exists")
                .value()
                .shape()
                .to_vec();
            let scale = 1.0 / n;
            let init_grad_data: Vec<f64> = logits_data
                .iter()
                .zip(targets.data())
                .map(|(&l, &t)| {
                    let p = 1.0 / (1.0 + (-l).exp());
                    scale * (p - t)
                })
                .collect();
            let init_grad = Tensor::from_vec(logits_shape, init_grad_data)?;
            graph.backward_with_grad(logits_node, init_grad)?;

            let grads = graph.take_gradients();
            (grads[0].clone(), grads[1].clone())
        };

        {
            let gradients = GradientSet::from_tensors(vec![weight_grad, bias_grad]);
            let mut parameters = model.parameters_mut();
            optimizer.step(&mut parameters, &gradients)?;
        }

        let probabilities = sigmoid(&model.forward(features)?)?;
        let loss = binary_cross_entropy(&probabilities, targets)?;
        let accuracy = binary_accuracy(&probabilities, targets, threshold)?;
        history.push(TrainingRecord::new(epoch, loss, Some(accuracy))?);
    }

    Ok(BinaryClassificationResult::new(model, history, threshold))
}

/// Trains a tiny sigmoid MLP on the XOR dataset using binary cross entropy.
///
/// Builds a full computation graph with hidden sigmoid each epoch, then seeds
/// the output logits node with the sigmoid+BCE combined gradient.
pub fn train_xor_mlp(config: TrainingConfig) -> Result<XorTrainingResult> {
    let dataset = xor()?;
    let mut model = XorMlp::new(0.5)?;
    let mut optimizer = SGD::new(config.learning_rate())?;
    let mut history = TrainingHistory::new();
    let features = dataset.features();
    let targets = dataset.targets();

    for epoch in 1..=config.epochs() {
        let grads = {
            let mut graph = ComputationGraph::new();
            // Collect trainable parameters in order.
            let hidden_params = model.hidden().parameters();
            let output_params = model.output().parameters();
            let params: Vec<&Tensor> = hidden_params
                .iter()
                .chain(output_params.iter())
                .copied()
                .collect();

            let f_node = graph.add_leaf(features.clone(), false);
            let hw_node = graph.add_leaf(params[0].clone(), true); // hidden weights
            let hb_node = graph.add_leaf(params[1].clone(), true); // hidden bias
            let ow_node = graph.add_leaf(params[2].clone(), true); // output weights
            let ob_node = graph.add_leaf(params[3].clone(), true); // output bias

            // Hidden: matmul -> row_add -> sigmoid
            let h_matmul_val = features.matmul(params[0])?;
            let h_matmul = graph.add_operation(
                Operation::MatMul,
                vec![f_node, hw_node],
                h_matmul_val,
                true,
            )?;
            let h_biased_val = graph
                .node(h_matmul)
                .expect("exists")
                .value()
                .row_add(params[1])?;
            let h_biased = graph.add_operation(
                Operation::RowAdd,
                vec![h_matmul, hb_node],
                h_biased_val,
                true,
            )?;
            let hidden_val = sigmoid(graph.node(h_biased).expect("exists").value())?;
            let hidden_node =
                graph.add_operation(Operation::Sigmoid, vec![h_biased], hidden_val, true)?;

            // Output: matmul -> row_add (logits, then sigmoid manually)
            let o_matmul_val = graph
                .node(hidden_node)
                .expect("exists")
                .value()
                .matmul(params[2])?;
            let o_matmul = graph.add_operation(
                Operation::MatMul,
                vec![hidden_node, ow_node],
                o_matmul_val,
                true,
            )?;
            let logits_val = graph
                .node(o_matmul)
                .expect("exists")
                .value()
                .row_add(params[3])?;
            let n = logits_val.len() as f64;
            let logits_shape = logits_val.shape().to_vec();
            let logits_data = logits_val.data().to_vec();
            let logits_node = graph.add_operation(
                Operation::RowAdd,
                vec![o_matmul, ob_node],
                logits_val,
                true,
            )?;

            // sigmoid + BCE combined: dL/dlogits = (sigma(logits) - t) / N
            let scale = 1.0 / n;
            let init_grad_data: Vec<f64> = logits_data
                .iter()
                .zip(targets.data())
                .map(|(&l, &t)| {
                    let p = 1.0 / (1.0 + (-l).exp());
                    scale * (p - t)
                })
                .collect();
            let init_grad = Tensor::from_vec(logits_shape, init_grad_data)?;
            graph.backward_with_grad(logits_node, init_grad)?;

            graph.take_gradients()
        };

        {
            let gradients = GradientSet::from_tensors(grads);
            let mut parameters = model.parameters_mut();
            optimizer.step(&mut parameters, &gradients)?;
        }

        let probabilities = model.predict_proba(features)?;
        let loss = binary_cross_entropy(&probabilities, targets)?;
        let accuracy = binary_accuracy(&probabilities, targets, model.threshold())?;
        history.push(TrainingRecord::new(epoch, loss, Some(accuracy))?);
    }

    Ok(XorTrainingResult::new(model, history))
}

/// Trains a softmax classifier on the deterministic spiral dataset.
///
/// Builds a computation graph each epoch, puts softmax in the graph, then seeds
/// with the softmax+CE combined gradient `(softmax(logits) - targets) / N`.
pub fn train_spiral_classifier(
    samples_per_class: usize,
    classes: usize,
    config: TrainingConfig,
) -> Result<SpiralTrainingResult> {
    let dataset = spiral(samples_per_class, classes)?;
    validate_one_hot_targets(dataset.targets())?;

    let mapped_features = spiral_feature_map(dataset.features())?;
    let input_size = mapped_features
        .cols()
        .expect("rank 2 tensors always have columns");
    let mut model = Linear::new(input_size, classes)?;
    let mut optimizer = SGD::new(config.learning_rate())?;
    let loss = CrossEntropyLoss::new();
    let mut history = TrainingHistory::new();
    let features = &mapped_features;
    let targets = dataset.targets();

    for epoch in 1..=config.epochs() {
        let (weight_grad, bias_grad) = {
            let mut graph = ComputationGraph::new();
            let f_node = graph.add_leaf(features.clone(), false);
            let trainable = model.parameters();
            let w_node = graph.add_leaf(trainable[0].clone(), true);
            let b_node = graph.add_leaf(trainable[1].clone(), true);

            let matmul_val = features.matmul(trainable[0])?;
            let matmul_node =
                graph.add_operation(Operation::MatMul, vec![f_node, w_node], matmul_val, true)?;
            let logits_val = graph
                .node(matmul_node)
                .expect("exists")
                .value()
                .row_add(trainable[1])?;
            let n = logits_val.rows().expect("matrix rows") as f64;
            let logits_shape = logits_val.shape().to_vec();
            let logits_node = graph.add_operation(
                Operation::RowAdd,
                vec![matmul_node, b_node],
                logits_val,
                true,
            )?;

            // softmax + CE combined: dL/dlogits = (softmax(logits) - targets) / N
            let sm = softmax(graph.node(logits_node).expect("exists").value())?;
            let scale = 1.0 / n;
            let init_grad_data: Vec<f64> = sm
                .data()
                .iter()
                .zip(targets.data())
                .map(|(&p, &t)| scale * (p - t))
                .collect();
            let init_grad = Tensor::from_vec(logits_shape, init_grad_data)?;
            graph.backward_with_grad(logits_node, init_grad)?;

            let grads = graph.take_gradients();
            (grads[0].clone(), grads[1].clone())
        };

        {
            let gradients = GradientSet::from_tensors(vec![weight_grad, bias_grad]);
            let mut parameters = model.parameters_mut();
            optimizer.step(&mut parameters, &gradients)?;
        }

        let updated_logits = model.forward(features)?;
        let updated_probabilities = softmax(&updated_logits)?;
        let epoch_loss = loss.forward(&updated_logits, targets)?.get_flat(0)?;
        let accuracy = categorical_accuracy(&updated_probabilities, targets)?;
        history.push(TrainingRecord::new(epoch, epoch_loss, Some(accuracy))?);
    }

    Ok(SpiralTrainingResult::new(model, history, classes))
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

fn probabilities_to_one_hot(probabilities: &Tensor) -> Result<Tensor> {
    validate_rank_two("probabilities", probabilities)?;

    let rows = probabilities
        .rows()
        .expect("rank 2 tensors always have rows");
    let cols = probabilities
        .cols()
        .expect("rank 2 tensors always have columns");
    let mut data = vec![0.0; probabilities.len()];

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let class_index = argmax(&probabilities.data()[start..end]);
        data[start + class_index] = 1.0;
    }

    Tensor::matrix(rows, cols, data)
}

fn spiral_feature_map(features: &Tensor) -> Result<Tensor> {
    validate_rank_two("features", features)?;
    if features.cols() != Some(2) {
        return Err(RustGradError::InvalidArgument {
            name: "features",
            reason: format!(
                "spiral feature map expects two columns, got {}",
                features.cols().expect("rank 2 tensors always have columns")
            ),
        });
    }

    let rows = features.rows().expect("rank 2 tensors always have rows");
    let mut mapped = Vec::with_capacity(rows * 2);

    for row in 0..rows {
        let x = features.get(&[row, 0])?;
        let y = features.get(&[row, 1])?;
        let radius = (x * x + y * y).sqrt();
        let angle = x.atan2(y);
        let phase = angle - radius * std::f64::consts::TAU;
        mapped.push(phase.cos());
        mapped.push(phase.sin());
    }

    Tensor::matrix(rows, 2, mapped)
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

fn validate_one_hot_targets(targets: &Tensor) -> Result<()> {
    validate_rank_two("targets", targets)?;

    let rows = targets.rows().expect("rank 2 tensors always have rows");
    let cols = targets.cols().expect("rank 2 tensors always have columns");
    if cols < 2 {
        return Err(RustGradError::InvalidArgument {
            name: "targets",
            reason: "one-hot targets need at least two classes".to_string(),
        });
    }

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_values = &targets.data()[start..end];
        let active = row_values
            .iter()
            .filter(|&&value| (value - 1.0).abs() <= f64::EPSILON)
            .count();
        let all_binary = row_values.iter().all(|&value| {
            (value - 0.0).abs() <= f64::EPSILON || (value - 1.0).abs() <= f64::EPSILON
        });

        if active != 1 || !all_binary {
            return Err(RustGradError::InvalidArgument {
                name: "targets",
                reason: "targets must be one-hot encoded rows".to_string(),
            });
        }
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
        train_binary_classification, train_linear_regression, train_spiral_classifier,
        train_xor_mlp, TrainingConfig, TrainingHistory, TrainingRecord,
    };
    use crate::data::{linear_regression, spiral, xor, Dataset};
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
    fn train_xor_mlp_records_loss_and_accuracy() {
        let config = TrainingConfig::new(30, 0.4).expect("valid config");

        let result = train_xor_mlp(config).expect("training should succeed");

        assert_eq!(result.history().len(), 30);
        assert_eq!(result.history().records()[0].epoch(), 1);
        assert_eq!(result.history().last().map(TrainingRecord::epoch), Some(30));
        assert!(result
            .history()
            .records()
            .iter()
            .all(|record| record.accuracy().is_some()));
        assert_eq!(result.model().hidden().weights().dims(), &[2, 2]);
        assert_eq!(result.model().output().weights().dims(), &[2, 1]);
    }

    #[test]
    fn train_xor_mlp_decreases_loss_and_keeps_perfect_accuracy() {
        let config = TrainingConfig::new(120, 0.4).expect("valid config");

        let result = train_xor_mlp(config).expect("training should succeed");

        assert!(result.history().loss_decreased());
        assert!(
            result.history().final_loss().expect("final loss exists")
                < result
                    .history()
                    .initial_loss()
                    .expect("initial loss exists"),
            "expected XOR loss to decrease, got {:?}",
            result.history().losses()
        );
        assert_eq!(result.history().best_accuracy(), Some(1.0));
    }

    #[test]
    fn xor_mlp_predicts_xor_truth_table() {
        let config = TrainingConfig::new(160, 0.4).expect("valid config");
        let result = train_xor_mlp(config).expect("training should succeed");
        let dataset = xor().expect("valid xor dataset");

        let probabilities = result
            .predict_proba(dataset.features())
            .expect("probabilities should compute");
        let classes = result
            .predict_classes(dataset.features())
            .expect("classes should compute");

        assert_eq!(probabilities.dims(), &[4, 1]);
        assert_eq!(classes.data(), dataset.targets().data());
        assert!(probabilities.get(&[0, 0]).expect("probability exists") < 0.5);
        assert!(probabilities.get(&[1, 0]).expect("probability exists") > 0.5);
        assert!(probabilities.get(&[2, 0]).expect("probability exists") > 0.5);
        assert!(probabilities.get(&[3, 0]).expect("probability exists") < 0.5);
    }

    #[test]
    fn train_spiral_classifier_records_loss_and_accuracy() {
        let config = TrainingConfig::new(40, 0.5).expect("valid config");

        let result = train_spiral_classifier(6, 3, config).expect("training should succeed");

        assert_eq!(result.history().len(), 40);
        assert_eq!(result.history().records()[0].epoch(), 1);
        assert_eq!(result.history().last().map(TrainingRecord::epoch), Some(40));
        assert_eq!(result.classes(), 3);
        assert_eq!(result.model().weights().dims(), &[2, 3]);
        assert!(result
            .history()
            .records()
            .iter()
            .all(|record| record.accuracy().is_some()));
    }

    #[test]
    fn train_spiral_classifier_decreases_loss_on_mapped_features() {
        let config = TrainingConfig::new(160, 0.7).expect("valid config");

        let result = train_spiral_classifier(12, 3, config).expect("training should succeed");

        assert!(result.history().loss_decreased());
        assert!(
            result.history().final_loss().expect("final loss exists")
                < result
                    .history()
                    .initial_loss()
                    .expect("initial loss exists"),
            "expected spiral loss to decrease, got {:?}",
            result.history().losses()
        );
        assert!(
            result.history().best_accuracy().expect("accuracy exists") >= 0.8,
            "expected useful spiral accuracy, got {:?}",
            result.history().best_accuracy()
        );
    }

    #[test]
    fn spiral_classifier_predicts_probability_rows_and_one_hot_classes() {
        let dataset = spiral(8, 3).expect("valid spiral dataset");
        let config = TrainingConfig::new(120, 0.7).expect("valid config");
        let result = train_spiral_classifier(8, 3, config).expect("training should succeed");

        let probabilities = result
            .predict_proba(dataset.features())
            .expect("probabilities should compute");
        let classes = result
            .predict_classes(dataset.features())
            .expect("classes should compute");

        assert_eq!(probabilities.dims(), &[24, 3]);
        assert_eq!(classes.dims(), &[24, 3]);

        for row in 0..24 {
            let probability_sum: f64 = (0..3)
                .map(|class_index| {
                    probabilities
                        .get(&[row, class_index])
                        .expect("probability exists")
                })
                .sum();
            let class_sum: f64 = (0..3)
                .map(|class_index| classes.get(&[row, class_index]).expect("class exists"))
                .sum();

            assert_close_with(probability_sum, 1.0, 1e-10);
            assert_close(class_sum, 1.0);
        }
    }

    #[test]
    fn train_spiral_classifier_rejects_invalid_dataset_configuration() {
        let config = TrainingConfig::new(10, 0.3).expect("valid config");

        let zero_samples =
            train_spiral_classifier(0, 3, config).expect_err("zero samples should fail");
        let one_class = train_spiral_classifier(4, 1, config).expect_err("one class should fail");

        assert_eq!(
            zero_samples,
            RustGradError::InvalidArgument {
                name: "samples_per_class",
                reason: "value must be greater than zero".to_string(),
            }
        );
        assert_eq!(
            one_class,
            RustGradError::InvalidArgument {
                name: "classes",
                reason: "classes must be at least 2".to_string(),
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
