//! Optimizers for updating trainable parameters.

use crate::tensor::Tensor;
use crate::{Result, RustGradError};

/// Immutable gradients aligned with a model's trainable parameters.
///
/// RustGrad keeps gradients outside `Tensor` for the first teaching version so
/// optimizers can be demonstrated independently from the computation graph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GradientSet {
    gradients: Vec<Tensor>,
}

impl GradientSet {
    /// Creates an empty gradient collection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a gradient collection from tensors.
    #[must_use]
    pub fn from_tensors(gradients: Vec<Tensor>) -> Self {
        Self { gradients }
    }

    /// Returns the number of stored gradients.
    #[must_use]
    pub fn len(&self) -> usize {
        self.gradients.len()
    }

    /// Returns true when no gradients are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gradients.is_empty()
    }

    /// Returns a gradient by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Tensor> {
        self.gradients.get(index)
    }

    /// Iterates over gradients in parameter order.
    pub fn iter(&self) -> impl Iterator<Item = &Tensor> {
        self.gradients.iter()
    }

    /// Removes all stored gradients.
    pub fn clear(&mut self) {
        self.gradients.clear();
    }
}

impl From<Vec<Tensor>> for GradientSet {
    fn from(gradients: Vec<Tensor>) -> Self {
        Self::from_tensors(gradients)
    }
}

/// Common interface for parameter optimizers.
pub trait Optimizer {
    /// Updates parameters in place using gradients with matching order.
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &GradientSet) -> Result<()>;

    /// Returns the optimizer learning rate.
    fn learning_rate(&self) -> f64;

    /// Updates the optimizer learning rate.
    fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()>;

    /// Returns a stable optimizer name for reports and debugging.
    fn name(&self) -> &str;
}

/// Stochastic gradient descent optimizer.
///
/// The update rule is `parameter = parameter - learning_rate * gradient`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    /// Creates an SGD optimizer.
    pub fn new(learning_rate: f64) -> Result<Self> {
        validate_learning_rate(learning_rate)?;
        Ok(Self { learning_rate })
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &GradientSet) -> Result<()> {
        validate_parameter_gradient_shapes(parameters, gradients)?;

        for (parameter, gradient) in parameters.iter_mut().zip(gradients.iter()) {
            for (value, &grad) in parameter.data_mut().iter_mut().zip(gradient.data()) {
                *value -= self.learning_rate * grad;
            }
        }

        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        validate_learning_rate(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(())
    }

    fn name(&self) -> &str {
        "sgd"
    }
}

/// Momentum optimizer.
///
/// The update rule is `velocity = momentum * velocity + gradient`, followed by
/// `parameter = parameter - learning_rate * velocity`.
#[derive(Debug, Clone, PartialEq)]
pub struct Momentum {
    learning_rate: f64,
    momentum: f64,
    velocity: Vec<Tensor>,
}

impl Momentum {
    /// Creates a Momentum optimizer.
    pub fn new(learning_rate: f64, momentum: f64) -> Result<Self> {
        validate_learning_rate(learning_rate)?;
        validate_momentum(momentum)?;

        Ok(Self {
            learning_rate,
            momentum,
            velocity: Vec::new(),
        })
    }

    /// Returns the momentum coefficient.
    #[must_use]
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// Updates the momentum coefficient.
    pub fn set_momentum(&mut self, momentum: f64) -> Result<()> {
        validate_momentum(momentum)?;
        self.momentum = momentum;
        Ok(())
    }

    /// Returns the stored velocity tensors.
    #[must_use]
    pub fn velocity(&self) -> &[Tensor] {
        &self.velocity
    }

    /// Clears optimizer state.
    pub fn reset_state(&mut self) {
        self.velocity.clear();
    }

    fn ensure_velocity(&mut self, parameters: &[&mut Tensor]) -> Result<()> {
        let needs_reset = self.velocity.len() != parameters.len()
            || self
                .velocity
                .iter()
                .zip(parameters.iter())
                .any(|(velocity, parameter)| velocity.dims() != parameter.dims());

        if needs_reset {
            self.velocity = parameters
                .iter()
                .map(|parameter| Tensor::zeros(parameter.shape().to_vec()))
                .collect::<Result<Vec<_>>>()?;
        }

        Ok(())
    }
}

impl Optimizer for Momentum {
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &GradientSet) -> Result<()> {
        validate_parameter_gradient_shapes(parameters, gradients)?;
        self.ensure_velocity(parameters)?;

        for ((parameter, gradient), velocity) in parameters
            .iter_mut()
            .zip(gradients.iter())
            .zip(self.velocity.iter_mut())
        {
            for ((value, &grad), velocity_value) in parameter
                .data_mut()
                .iter_mut()
                .zip(gradient.data())
                .zip(velocity.data_mut())
            {
                *velocity_value = self.momentum * *velocity_value + grad;
                *value -= self.learning_rate * *velocity_value;
            }
        }

        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        validate_learning_rate(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(())
    }

    fn name(&self) -> &str {
        "momentum"
    }
}

/// Adam optimizer with bias-corrected first and second moment estimates.
///
/// The update rule follows the common Adam form:
/// `m = beta1 * m + (1 - beta1) * gradient`,
/// `v = beta2 * v + (1 - beta2) * gradient^2`, then parameters are updated
/// with bias-corrected `m_hat` and `v_hat`.
#[derive(Debug, Clone, PartialEq)]
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    timestep: usize,
    first_moment: Vec<Tensor>,
    second_moment: Vec<Tensor>,
}

impl Adam {
    /// Creates an Adam optimizer with common default hyperparameters.
    pub fn new(learning_rate: f64) -> Result<Self> {
        Self::with_hyperparameters(learning_rate, 0.9, 0.999, 1e-8)
    }

    /// Creates an Adam optimizer with explicit hyperparameters.
    pub fn with_hyperparameters(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Result<Self> {
        validate_learning_rate(learning_rate)?;
        validate_beta("beta1", beta1)?;
        validate_beta("beta2", beta2)?;
        validate_epsilon(epsilon)?;

        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            timestep: 0,
            first_moment: Vec::new(),
            second_moment: Vec::new(),
        })
    }

    /// Returns the first moment decay coefficient.
    #[must_use]
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Updates the first moment decay coefficient.
    pub fn set_beta1(&mut self, beta1: f64) -> Result<()> {
        validate_beta("beta1", beta1)?;
        self.beta1 = beta1;
        Ok(())
    }

    /// Returns the second moment decay coefficient.
    #[must_use]
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Updates the second moment decay coefficient.
    pub fn set_beta2(&mut self, beta2: f64) -> Result<()> {
        validate_beta("beta2", beta2)?;
        self.beta2 = beta2;
        Ok(())
    }

    /// Returns the numerical epsilon used in the update denominator.
    #[must_use]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Updates the numerical epsilon used in the update denominator.
    pub fn set_epsilon(&mut self, epsilon: f64) -> Result<()> {
        validate_epsilon(epsilon)?;
        self.epsilon = epsilon;
        Ok(())
    }

    /// Returns the number of successful Adam steps since the last state reset.
    #[must_use]
    pub fn timestep(&self) -> usize {
        self.timestep
    }

    /// Returns the stored first moment tensors.
    #[must_use]
    pub fn first_moment(&self) -> &[Tensor] {
        &self.first_moment
    }

    /// Returns the stored second moment tensors.
    #[must_use]
    pub fn second_moment(&self) -> &[Tensor] {
        &self.second_moment
    }

    /// Clears optimizer state.
    pub fn reset_state(&mut self) {
        self.timestep = 0;
        self.first_moment.clear();
        self.second_moment.clear();
    }

    fn ensure_moments(&mut self, parameters: &[&mut Tensor]) -> Result<()> {
        let needs_reset = self.first_moment.len() != parameters.len()
            || self.second_moment.len() != parameters.len()
            || self
                .first_moment
                .iter()
                .zip(parameters.iter())
                .any(|(moment, parameter)| moment.dims() != parameter.dims())
            || self
                .second_moment
                .iter()
                .zip(parameters.iter())
                .any(|(moment, parameter)| moment.dims() != parameter.dims());

        if needs_reset {
            self.timestep = 0;
            self.first_moment = parameters
                .iter()
                .map(|parameter| Tensor::zeros(parameter.shape().to_vec()))
                .collect::<Result<Vec<_>>>()?;
            self.second_moment = parameters
                .iter()
                .map(|parameter| Tensor::zeros(parameter.shape().to_vec()))
                .collect::<Result<Vec<_>>>()?;
        }

        Ok(())
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &GradientSet) -> Result<()> {
        validate_parameter_gradient_shapes(parameters, gradients)?;
        self.ensure_moments(parameters)?;
        self.timestep += 1;

        let beta1_correction = 1.0 - self.beta1.powf(self.timestep as f64);
        let beta2_correction = 1.0 - self.beta2.powf(self.timestep as f64);

        for (((parameter, gradient), first_moment), second_moment) in parameters
            .iter_mut()
            .zip(gradients.iter())
            .zip(self.first_moment.iter_mut())
            .zip(self.second_moment.iter_mut())
        {
            for (((value, &grad), first), second) in parameter
                .data_mut()
                .iter_mut()
                .zip(gradient.data())
                .zip(first_moment.data_mut())
                .zip(second_moment.data_mut())
            {
                *first = self.beta1 * *first + (1.0 - self.beta1) * grad;
                *second = self.beta2 * *second + (1.0 - self.beta2) * grad * grad;

                let first_hat = *first / beta1_correction;
                let second_hat = *second / beta2_correction;
                *value -= self.learning_rate * first_hat / (second_hat.sqrt() + self.epsilon);
            }
        }

        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        validate_learning_rate(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(())
    }

    fn name(&self) -> &str {
        "adam"
    }
}

fn validate_learning_rate(learning_rate: f64) -> Result<()> {
    if learning_rate <= 0.0 || !learning_rate.is_finite() {
        return Err(RustGradError::InvalidArgument {
            name: "learning_rate",
            reason: "learning rate must be finite and greater than zero".to_string(),
        });
    }

    Ok(())
}

fn validate_momentum(momentum: f64) -> Result<()> {
    if !(0.0..1.0).contains(&momentum) || !momentum.is_finite() {
        return Err(RustGradError::InvalidArgument {
            name: "momentum",
            reason: "momentum must be finite and in [0, 1)".to_string(),
        });
    }

    Ok(())
}

fn validate_beta(name: &'static str, beta: f64) -> Result<()> {
    if !(0.0..1.0).contains(&beta) || !beta.is_finite() {
        return Err(RustGradError::InvalidArgument {
            name,
            reason: "beta must be finite and in [0, 1)".to_string(),
        });
    }

    Ok(())
}

fn validate_epsilon(epsilon: f64) -> Result<()> {
    if epsilon <= 0.0 || !epsilon.is_finite() {
        return Err(RustGradError::InvalidArgument {
            name: "epsilon",
            reason: "epsilon must be finite and greater than zero".to_string(),
        });
    }

    Ok(())
}

fn validate_parameter_gradient_shapes(
    parameters: &[&mut Tensor],
    gradients: &GradientSet,
) -> Result<()> {
    validate_parameter_gradient_count(parameters.len(), gradients.len())?;

    for (parameter, gradient) in parameters.iter().zip(gradients.iter()) {
        validate_gradient_shape(parameter, gradient)?;
    }

    Ok(())
}

fn validate_parameter_gradient_count(parameter_count: usize, gradient_count: usize) -> Result<()> {
    if parameter_count == gradient_count {
        Ok(())
    } else {
        Err(RustGradError::InvalidArgument {
            name: "gradients",
            reason: format!("expected {parameter_count} gradients, got {gradient_count}"),
        })
    }
}

fn validate_gradient_shape(parameter: &Tensor, gradient: &Tensor) -> Result<()> {
    if parameter.dims() == gradient.dims() {
        Ok(())
    } else {
        Err(RustGradError::ShapeMismatch {
            op: "optimizer gradient",
            left: parameter.shape().to_vec(),
            right: gradient.shape().to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{GradientSet, Momentum, Optimizer, SGD};
    use crate::tensor::Tensor;
    use crate::RustGradError;

    const EPSILON: f64 = 1e-12;

    fn assert_slice_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < EPSILON,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn gradient_set_starts_empty() {
        let gradients = GradientSet::new();

        assert!(gradients.is_empty());
        assert_eq!(gradients.len(), 0);
        assert!(gradients.get(0).is_none());
        assert_eq!(gradients.iter().count(), 0);
    }

    #[test]
    fn gradient_set_keeps_gradients_in_parameter_order() {
        let first = Tensor::vector(vec![1.0, 2.0]).expect("valid first gradient");
        let second = Tensor::matrix(1, 2, vec![3.0, 4.0]).expect("valid second gradient");
        let gradients = GradientSet::from_tensors(vec![first.clone(), second.clone()]);

        assert_eq!(gradients.len(), 2);
        assert_eq!(gradients.get(0), Some(&first));
        assert_eq!(gradients.get(1), Some(&second));
        assert_eq!(
            gradients.iter().map(Tensor::dims).collect::<Vec<_>>(),
            vec![&[2][..], &[1, 2][..]]
        );
    }

    #[test]
    fn gradient_set_can_be_created_from_vec_and_cleared() {
        let mut gradients: GradientSet =
            vec![Tensor::scalar(1.0).expect("valid scalar gradient")].into();

        assert_eq!(gradients.len(), 1);

        gradients.clear();

        assert!(gradients.is_empty());
    }

    #[test]
    fn sgd_exposes_name_and_learning_rate() {
        let optimizer = SGD::new(0.05).expect("valid learning rate");

        assert_eq!(optimizer.name(), "sgd");
        assert_eq!(optimizer.learning_rate(), 0.05);
    }

    #[test]
    fn sgd_can_update_learning_rate() {
        let mut optimizer = SGD::new(0.1).expect("valid learning rate");

        optimizer
            .set_learning_rate(0.25)
            .expect("learning rate update should succeed");

        assert_eq!(optimizer.learning_rate(), 0.25);
    }

    #[test]
    fn sgd_rejects_invalid_learning_rate_on_create() {
        let error = SGD::new(0.0).expect_err("zero learning rate should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "learning_rate",
                reason: "learning rate must be finite and greater than zero".to_string(),
            }
        );
    }

    #[test]
    fn sgd_rejects_invalid_learning_rate_on_update() {
        let mut optimizer = SGD::new(0.1).expect("valid learning rate");

        let error = optimizer
            .set_learning_rate(f64::INFINITY)
            .expect_err("infinite learning rate should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "learning_rate",
                reason: "learning rate must be finite and greater than zero".to_string(),
            }
        );
        assert_eq!(optimizer.learning_rate(), 0.1);
    }

    #[test]
    fn sgd_updates_single_parameter_tensor() {
        let mut parameter = Tensor::vector(vec![1.0, -2.0, 3.0]).expect("valid parameter");
        let gradients = GradientSet::from_tensors(vec![
            Tensor::vector(vec![0.5, -1.0, 2.0]).expect("valid gradient")
        ]);
        let mut optimizer = SGD::new(0.1).expect("valid learning rate");
        let mut parameters = vec![&mut parameter];

        optimizer
            .step(&mut parameters, &gradients)
            .expect("sgd step should succeed");

        assert_slice_close(parameter.data(), &[0.95, -1.9, 2.8]);
    }

    #[test]
    fn sgd_updates_multiple_parameter_tensors() {
        let mut weights = Tensor::matrix(1, 2, vec![1.0, 2.0]).expect("valid weights");
        let mut bias = Tensor::vector(vec![0.5, -0.5]).expect("valid bias");
        let gradients = GradientSet::from_tensors(vec![
            Tensor::matrix(1, 2, vec![0.2, -0.4]).expect("valid weight gradient"),
            Tensor::vector(vec![1.0, -1.0]).expect("valid bias gradient"),
        ]);
        let mut optimizer = SGD::new(0.5).expect("valid learning rate");
        let mut parameters = vec![&mut weights, &mut bias];

        optimizer
            .step(&mut parameters, &gradients)
            .expect("sgd step should succeed");

        assert_slice_close(weights.data(), &[0.9, 2.2]);
        assert_slice_close(bias.data(), &[0.0, 0.0]);
    }

    #[test]
    fn sgd_rejects_gradient_count_mismatch() {
        let mut parameter = Tensor::scalar(1.0).expect("valid parameter");
        let gradients = GradientSet::new();
        let mut optimizer = SGD::new(0.1).expect("valid learning rate");
        let mut parameters = vec![&mut parameter];

        let error = optimizer
            .step(&mut parameters, &gradients)
            .expect_err("missing gradient should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "gradients",
                reason: "expected 1 gradients, got 0".to_string(),
            }
        );
    }

    #[test]
    fn sgd_rejects_gradient_shape_mismatch() {
        let mut parameter = Tensor::vector(vec![1.0, 2.0]).expect("valid parameter");
        let gradients = GradientSet::from_tensors(vec![
            Tensor::matrix(1, 2, vec![0.1, 0.2]).expect("valid gradient")
        ]);
        let mut optimizer = SGD::new(0.1).expect("valid learning rate");
        let mut parameters = vec![&mut parameter];

        let error = optimizer
            .step(&mut parameters, &gradients)
            .expect_err("shape mismatch should fail");

        assert_eq!(
            error,
            RustGradError::ShapeMismatch {
                op: "optimizer gradient",
                left: vec![2],
                right: vec![1, 2],
            }
        );
    }

    #[test]
    fn sgd_does_not_update_parameter_when_shape_validation_fails() {
        let mut parameter = Tensor::vector(vec![1.0, 2.0]).expect("valid parameter");
        let gradients =
            GradientSet::from_tensors(vec![Tensor::scalar(10.0).expect("invalid gradient shape")]);
        let mut optimizer = SGD::new(0.1).expect("valid learning rate");
        let mut parameters = vec![&mut parameter];

        let _ = optimizer.step(&mut parameters, &gradients);

        assert_eq!(parameter.data(), &[1.0, 2.0]);
    }

    #[test]
    fn momentum_exposes_name_learning_rate_and_momentum() {
        let optimizer = Momentum::new(0.05, 0.9).expect("valid optimizer");

        assert_eq!(optimizer.name(), "momentum");
        assert_eq!(optimizer.learning_rate(), 0.05);
        assert_eq!(optimizer.momentum(), 0.9);
        assert!(optimizer.velocity().is_empty());
    }

    #[test]
    fn momentum_can_update_hyperparameters() {
        let mut optimizer = Momentum::new(0.1, 0.5).expect("valid optimizer");

        optimizer
            .set_learning_rate(0.25)
            .expect("learning rate update should succeed");
        optimizer
            .set_momentum(0.8)
            .expect("momentum update should succeed");

        assert_eq!(optimizer.learning_rate(), 0.25);
        assert_eq!(optimizer.momentum(), 0.8);
    }

    #[test]
    fn momentum_rejects_invalid_momentum_on_create() {
        let error = Momentum::new(0.1, 1.0).expect_err("momentum must be below one");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "momentum",
                reason: "momentum must be finite and in [0, 1)".to_string(),
            }
        );
    }

    #[test]
    fn momentum_rejects_invalid_momentum_on_update() {
        let mut optimizer = Momentum::new(0.1, 0.5).expect("valid optimizer");

        let error = optimizer
            .set_momentum(f64::NAN)
            .expect_err("nan momentum should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "momentum",
                reason: "momentum must be finite and in [0, 1)".to_string(),
            }
        );
        assert_eq!(optimizer.momentum(), 0.5);
    }

    #[test]
    fn momentum_first_step_matches_sgd_and_initializes_velocity() {
        let mut parameter = Tensor::vector(vec![1.0, 2.0]).expect("valid parameter");
        let gradients =
            GradientSet::from_tensors(vec![Tensor::vector(vec![0.5, -1.0]).expect("valid grad")]);
        let mut optimizer = Momentum::new(0.1, 0.9).expect("valid optimizer");
        let mut parameters = vec![&mut parameter];

        optimizer
            .step(&mut parameters, &gradients)
            .expect("momentum step should succeed");

        assert_slice_close(parameter.data(), &[0.95, 2.1]);
        assert_eq!(optimizer.velocity().len(), 1);
        assert_slice_close(optimizer.velocity()[0].data(), &[0.5, -1.0]);
    }

    #[test]
    fn momentum_accumulates_velocity_across_steps() {
        let mut parameter = Tensor::vector(vec![1.0, 2.0]).expect("valid parameter");
        let gradients =
            GradientSet::from_tensors(vec![Tensor::vector(vec![0.5, -1.0]).expect("valid grad")]);
        let mut optimizer = Momentum::new(0.1, 0.9).expect("valid optimizer");

        {
            let mut parameters = vec![&mut parameter];
            optimizer
                .step(&mut parameters, &gradients)
                .expect("first step should succeed");
        }
        {
            let mut parameters = vec![&mut parameter];
            optimizer
                .step(&mut parameters, &gradients)
                .expect("second step should succeed");
        }

        assert_slice_close(optimizer.velocity()[0].data(), &[0.95, -1.9]);
        assert_slice_close(parameter.data(), &[0.855, 2.29]);
    }

    #[test]
    fn momentum_updates_multiple_parameter_tensors() {
        let mut weights = Tensor::matrix(1, 2, vec![1.0, 2.0]).expect("valid weights");
        let mut bias = Tensor::vector(vec![0.5, -0.5]).expect("valid bias");
        let gradients = GradientSet::from_tensors(vec![
            Tensor::matrix(1, 2, vec![0.2, -0.4]).expect("valid weight gradient"),
            Tensor::vector(vec![1.0, -1.0]).expect("valid bias gradient"),
        ]);
        let mut optimizer = Momentum::new(0.5, 0.9).expect("valid optimizer");
        let mut parameters = vec![&mut weights, &mut bias];

        optimizer
            .step(&mut parameters, &gradients)
            .expect("momentum step should succeed");

        assert_slice_close(weights.data(), &[0.9, 2.2]);
        assert_slice_close(bias.data(), &[0.0, 0.0]);
        assert_eq!(optimizer.velocity().len(), 2);
        assert_slice_close(optimizer.velocity()[0].data(), &[0.2, -0.4]);
        assert_slice_close(optimizer.velocity()[1].data(), &[1.0, -1.0]);
    }

    #[test]
    fn momentum_reset_state_clears_velocity() {
        let mut parameter = Tensor::scalar(1.0).expect("valid parameter");
        let gradients =
            GradientSet::from_tensors(vec![Tensor::scalar(0.5).expect("valid gradient")]);
        let mut optimizer = Momentum::new(0.1, 0.9).expect("valid optimizer");
        let mut parameters = vec![&mut parameter];

        optimizer
            .step(&mut parameters, &gradients)
            .expect("momentum step should succeed");
        assert_eq!(optimizer.velocity().len(), 1);

        optimizer.reset_state();

        assert!(optimizer.velocity().is_empty());
    }

    #[test]
    fn momentum_rebuilds_velocity_when_parameter_shapes_change() {
        let mut first_parameter = Tensor::vector(vec![1.0, 2.0]).expect("valid parameter");
        let first_gradients =
            GradientSet::from_tensors(vec![Tensor::vector(vec![0.5, 0.5]).expect("valid grad")]);
        let mut optimizer = Momentum::new(0.1, 0.9).expect("valid optimizer");

        {
            let mut parameters = vec![&mut first_parameter];
            optimizer
                .step(&mut parameters, &first_gradients)
                .expect("first step should succeed");
        }
        assert_eq!(optimizer.velocity()[0].dims(), &[2]);

        let mut second_parameter = Tensor::matrix(1, 1, vec![3.0]).expect("valid parameter");
        let second_gradients =
            GradientSet::from_tensors(vec![Tensor::matrix(1, 1, vec![2.0]).expect("valid grad")]);
        {
            let mut parameters = vec![&mut second_parameter];
            optimizer
                .step(&mut parameters, &second_gradients)
                .expect("second shape should rebuild velocity");
        }

        assert_eq!(optimizer.velocity().len(), 1);
        assert_eq!(optimizer.velocity()[0].dims(), &[1, 1]);
        assert_slice_close(optimizer.velocity()[0].data(), &[2.0]);
        assert_slice_close(second_parameter.data(), &[2.8]);
    }

    #[test]
    fn momentum_rejects_gradient_count_mismatch() {
        let mut parameter = Tensor::scalar(1.0).expect("valid parameter");
        let gradients = GradientSet::new();
        let mut optimizer = Momentum::new(0.1, 0.9).expect("valid optimizer");
        let mut parameters = vec![&mut parameter];

        let error = optimizer
            .step(&mut parameters, &gradients)
            .expect_err("missing gradient should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "gradients",
                reason: "expected 1 gradients, got 0".to_string(),
            }
        );
        assert!(optimizer.velocity().is_empty());
    }

    #[test]
    fn momentum_rejects_gradient_shape_mismatch_without_initializing_velocity() {
        let mut parameter = Tensor::vector(vec![1.0, 2.0]).expect("valid parameter");
        let gradients =
            GradientSet::from_tensors(vec![Tensor::scalar(1.0).expect("invalid gradient")]);
        let mut optimizer = Momentum::new(0.1, 0.9).expect("valid optimizer");
        let mut parameters = vec![&mut parameter];

        let error = optimizer
            .step(&mut parameters, &gradients)
            .expect_err("shape mismatch should fail");

        assert_eq!(
            error,
            RustGradError::ShapeMismatch {
                op: "optimizer gradient",
                left: vec![2],
                right: vec![1],
            }
        );
        assert!(optimizer.velocity().is_empty());
        assert_slice_close(parameter.data(), &[1.0, 2.0]);
    }
}
