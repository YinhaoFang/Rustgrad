//! Synthetic datasets used by examples and integration tests.

use crate::tensor::Tensor;
use crate::{Result, RustGradError};

/// In-memory supervised dataset with feature and target tensors.
///
/// Features and targets are stored as row-major matrices. Each row represents
/// one sample, which keeps batching and reporting simple for the first
/// course-project version of RustGrad.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    name: String,
    features: Tensor,
    targets: Tensor,
}

impl Dataset {
    /// Creates a supervised dataset from feature and target matrices.
    pub fn new(name: impl Into<String>, features: Tensor, targets: Tensor) -> Result<Self> {
        validate_matrix("features", &features)?;
        validate_matrix("targets", &targets)?;

        let feature_rows = features.rows().expect("features rank already validated");
        let target_rows = targets.rows().expect("targets rank already validated");
        if feature_rows != target_rows {
            return Err(RustGradError::ShapeMismatch {
                op: "dataset rows",
                left: features.shape().to_vec(),
                right: targets.shape().to_vec(),
            });
        }

        let name = name.into();
        if name.trim().is_empty() {
            return Err(RustGradError::InvalidArgument {
                name: "name",
                reason: "dataset name must not be empty".to_string(),
            });
        }

        Ok(Self {
            name,
            features,
            targets,
        })
    }

    /// Returns the dataset name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns all feature rows.
    #[must_use]
    pub fn features(&self) -> &Tensor {
        &self.features
    }

    /// Returns all target rows.
    #[must_use]
    pub fn targets(&self) -> &Tensor {
        &self.targets
    }

    /// Returns the number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.features
            .rows()
            .expect("dataset features are always rank 2")
    }

    /// Returns true when the dataset contains no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of input features per sample.
    #[must_use]
    pub fn input_size(&self) -> usize {
        self.features
            .cols()
            .expect("dataset features are always rank 2")
    }

    /// Returns the number of target values per sample.
    #[must_use]
    pub fn target_size(&self) -> usize {
        self.targets
            .cols()
            .expect("dataset targets are always rank 2")
    }

    /// Returns one sample as feature and target vectors.
    pub fn sample(&self, index: usize) -> Result<(Tensor, Tensor)> {
        if index >= self.len() {
            return Err(RustGradError::IndexOutOfBounds {
                index: vec![index],
                shape: vec![self.len()],
            });
        }

        Ok((
            row_as_vector(&self.features, index)?,
            row_as_vector(&self.targets, index)?,
        ))
    }

    /// Returns a contiguous batch as a new dataset.
    pub fn batch(&self, start: usize, batch_size: usize) -> Result<Self> {
        validate_positive("batch_size", batch_size)?;
        let end = start
            .checked_add(batch_size)
            .ok_or_else(|| RustGradError::InvalidArgument {
                name: "batch_size",
                reason: "batch range overflows usize".to_string(),
            })?;

        if start >= self.len() || end > self.len() {
            return Err(RustGradError::IndexOutOfBounds {
                index: vec![start, end],
                shape: vec![self.len()],
            });
        }

        let feature_cols = self.input_size();
        let target_cols = self.target_size();
        let feature_start = start * feature_cols;
        let feature_end = end * feature_cols;
        let target_start = start * target_cols;
        let target_end = end * target_cols;

        Self::new(
            format!("{}[{start}..{end}]", self.name),
            Tensor::matrix(
                batch_size,
                feature_cols,
                self.features.data()[feature_start..feature_end].to_vec(),
            )?,
            Tensor::matrix(
                batch_size,
                target_cols,
                self.targets.data()[target_start..target_end].to_vec(),
            )?,
        )
    }
}

/// Creates a deterministic one-dimensional linear regression dataset.
///
/// Inputs are evenly spaced in `[-1, 1]`, and targets follow
/// `y = slope * x + intercept`.
pub fn linear_regression(samples: usize, slope: f64, intercept: f64) -> Result<Dataset> {
    validate_positive("samples", samples)?;
    validate_finite("slope", slope)?;
    validate_finite("intercept", intercept)?;

    let mut features = Vec::with_capacity(samples);
    let mut targets = Vec::with_capacity(samples);
    for index in 0..samples {
        let x = evenly_spaced(index, samples, -1.0, 1.0);
        features.push(x);
        targets.push(slope * x + intercept);
    }

    Dataset::new(
        "linear-regression",
        Tensor::matrix(samples, 1, features)?,
        Tensor::matrix(samples, 1, targets)?,
    )
}

/// Creates the classic XOR classification dataset.
///
/// Targets are stored as one scalar per sample: `0.0` or `1.0`.
pub fn xor() -> Result<Dataset> {
    Dataset::new(
        "xor",
        Tensor::matrix(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?,
        Tensor::matrix(4, 1, vec![0.0, 1.0, 1.0, 0.0])?,
    )
}

/// Creates a deterministic two-dimensional spiral classification dataset.
///
/// Targets are one-hot rows with `classes` columns. The generated points are
/// arranged in class-specific arms and are deterministic by design, which makes
/// example output stable across machines.
pub fn spiral(samples_per_class: usize, classes: usize) -> Result<Dataset> {
    validate_positive("samples_per_class", samples_per_class)?;
    if classes < 2 {
        return Err(RustGradError::InvalidArgument {
            name: "classes",
            reason: "classes must be at least 2".to_string(),
        });
    }

    let total_samples = samples_per_class * classes;
    let mut features = Vec::with_capacity(total_samples * 2);
    let mut targets = Vec::with_capacity(total_samples * classes);

    for class_index in 0..classes {
        for sample_index in 0..samples_per_class {
            let ratio = unit_interval(sample_index, samples_per_class);
            let radius = ratio;
            let angle = class_index as f64 * std::f64::consts::TAU / classes as f64
                + ratio * std::f64::consts::TAU;

            features.push(radius * angle.sin());
            features.push(radius * angle.cos());
            targets.extend(one_hot(class_index, classes)?);
        }
    }

    Dataset::new(
        "spiral",
        Tensor::matrix(total_samples, 2, features)?,
        Tensor::matrix(total_samples, classes, targets)?,
    )
}

/// Creates a one-hot encoded vector.
pub fn one_hot(class_index: usize, classes: usize) -> Result<Vec<f64>> {
    if classes == 0 {
        return Err(RustGradError::InvalidArgument {
            name: "classes",
            reason: "classes must be greater than zero".to_string(),
        });
    }
    if class_index >= classes {
        return Err(RustGradError::InvalidArgument {
            name: "class_index",
            reason: format!("class index {class_index} is out of range for {classes} classes"),
        });
    }

    let mut encoded = vec![0.0; classes];
    encoded[class_index] = 1.0;
    Ok(encoded)
}

fn row_as_vector(tensor: &Tensor, row: usize) -> Result<Tensor> {
    let cols = tensor.cols().expect("dataset tensors are rank 2");
    let start = row * cols;
    let end = start + cols;
    Tensor::vector(tensor.data()[start..end].to_vec())
}

fn validate_matrix(name: &'static str, tensor: &Tensor) -> Result<()> {
    if tensor.rank() == 2 {
        Ok(())
    } else {
        Err(RustGradError::InvalidArgument {
            name,
            reason: format!("dataset {name} must be rank 2, got rank {}", tensor.rank()),
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

fn evenly_spaced(index: usize, samples: usize, start: f64, end: f64) -> f64 {
    if samples == 1 {
        start
    } else {
        start + (end - start) * index as f64 / (samples - 1) as f64
    }
}

fn unit_interval(index: usize, samples: usize) -> f64 {
    if samples == 1 {
        0.0
    } else {
        index as f64 / (samples - 1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::{linear_regression, one_hot, spiral, xor, Dataset};
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
    fn dataset_new_stores_metadata_and_shapes() {
        let dataset = Dataset::new(
            "toy",
            Tensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid features"),
            Tensor::matrix(2, 1, vec![0.0, 1.0]).expect("valid targets"),
        )
        .expect("valid dataset");

        assert_eq!(dataset.name(), "toy");
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
        assert_eq!(dataset.input_size(), 2);
        assert_eq!(dataset.target_size(), 1);
        assert_eq!(dataset.features().dims(), &[2, 2]);
        assert_eq!(dataset.targets().dims(), &[2, 1]);
    }

    #[test]
    fn dataset_new_rejects_empty_name() {
        let error = Dataset::new(
            " ",
            Tensor::matrix(1, 1, vec![1.0]).expect("valid features"),
            Tensor::matrix(1, 1, vec![1.0]).expect("valid targets"),
        )
        .expect_err("empty name should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "name",
                reason: "dataset name must not be empty".to_string(),
            }
        );
    }

    #[test]
    fn dataset_new_rejects_non_matrix_features() {
        let error = Dataset::new(
            "bad",
            Tensor::vector(vec![1.0, 2.0]).expect("invalid feature rank"),
            Tensor::matrix(2, 1, vec![0.0, 1.0]).expect("valid targets"),
        )
        .expect_err("rank one features should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "features",
                reason: "dataset features must be rank 2, got rank 1".to_string(),
            }
        );
    }

    #[test]
    fn dataset_new_rejects_non_matrix_targets() {
        let error = Dataset::new(
            "bad",
            Tensor::matrix(2, 1, vec![1.0, 2.0]).expect("valid features"),
            Tensor::vector(vec![0.0, 1.0]).expect("invalid target rank"),
        )
        .expect_err("rank one targets should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "targets",
                reason: "dataset targets must be rank 2, got rank 1".to_string(),
            }
        );
    }

    #[test]
    fn dataset_new_rejects_row_count_mismatch() {
        let error = Dataset::new(
            "bad",
            Tensor::matrix(2, 1, vec![1.0, 2.0]).expect("valid features"),
            Tensor::matrix(1, 1, vec![1.0]).expect("valid targets"),
        )
        .expect_err("row mismatch should fail");

        assert_eq!(
            error,
            RustGradError::ShapeMismatch {
                op: "dataset rows",
                left: vec![2, 1],
                right: vec![1, 1],
            }
        );
    }

    #[test]
    fn dataset_sample_returns_feature_and_target_vectors() {
        let dataset = Dataset::new(
            "toy",
            Tensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid features"),
            Tensor::matrix(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("valid targets"),
        )
        .expect("valid dataset");

        let (features, targets) = dataset.sample(1).expect("sample should exist");

        assert_eq!(features.dims(), &[2]);
        assert_eq!(targets.dims(), &[2]);
        assert_eq!(features.data(), &[3.0, 4.0]);
        assert_eq!(targets.data(), &[0.0, 1.0]);
    }

    #[test]
    fn dataset_sample_rejects_out_of_bounds_index() {
        let dataset = xor().expect("valid xor dataset");

        let error = dataset
            .sample(4)
            .expect_err("index equal to len should fail");

        assert_eq!(
            error,
            RustGradError::IndexOutOfBounds {
                index: vec![4],
                shape: vec![4],
            }
        );
    }

    #[test]
    fn dataset_batch_returns_contiguous_rows() {
        let dataset = Dataset::new(
            "toy",
            Tensor::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid features"),
            Tensor::matrix(3, 1, vec![10.0, 20.0, 30.0]).expect("valid targets"),
        )
        .expect("valid dataset");

        let batch = dataset.batch(1, 2).expect("batch should exist");

        assert_eq!(batch.name(), "toy[1..3]");
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.input_size(), 2);
        assert_eq!(batch.target_size(), 1);
        assert_eq!(batch.features().data(), &[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(batch.targets().data(), &[20.0, 30.0]);
    }

    #[test]
    fn dataset_batch_rejects_zero_batch_size() {
        let dataset = xor().expect("valid xor dataset");

        let error = dataset
            .batch(0, 0)
            .expect_err("zero batch size should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "batch_size",
                reason: "value must be greater than zero".to_string(),
            }
        );
    }

    #[test]
    fn dataset_batch_rejects_out_of_bounds_range() {
        let dataset = xor().expect("valid xor dataset");

        let error = dataset
            .batch(3, 2)
            .expect_err("range past dataset length should fail");

        assert_eq!(
            error,
            RustGradError::IndexOutOfBounds {
                index: vec![3, 5],
                shape: vec![4],
            }
        );
    }

    #[test]
    fn linear_regression_generates_evenly_spaced_samples() {
        let dataset = linear_regression(3, 2.0, 1.0).expect("valid linear dataset");

        assert_eq!(dataset.name(), "linear-regression");
        assert_eq!(dataset.features().data(), &[-1.0, 0.0, 1.0]);
        assert_eq!(dataset.targets().data(), &[-1.0, 1.0, 3.0]);
    }

    #[test]
    fn linear_regression_supports_single_sample() {
        let dataset = linear_regression(1, 3.0, -2.0).expect("valid linear dataset");

        assert_eq!(dataset.features().data(), &[-1.0]);
        assert_eq!(dataset.targets().data(), &[-5.0]);
    }

    #[test]
    fn linear_regression_rejects_zero_samples() {
        let error = linear_regression(0, 1.0, 0.0).expect_err("zero samples should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "samples",
                reason: "value must be greater than zero".to_string(),
            }
        );
    }

    #[test]
    fn linear_regression_rejects_non_finite_parameters() {
        let error =
            linear_regression(2, f64::INFINITY, 0.0).expect_err("infinite slope should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "slope",
                reason: "value must be finite".to_string(),
            }
        );
    }

    #[test]
    fn xor_dataset_has_expected_truth_table() {
        let dataset = xor().expect("valid xor dataset");

        assert_eq!(dataset.name(), "xor");
        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.input_size(), 2);
        assert_eq!(dataset.target_size(), 1);
        assert_eq!(
            dataset.features().data(),
            &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        );
        assert_eq!(dataset.targets().data(), &[0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn spiral_dataset_has_expected_shapes_and_one_hot_targets() {
        let dataset = spiral(3, 2).expect("valid spiral dataset");

        assert_eq!(dataset.name(), "spiral");
        assert_eq!(dataset.len(), 6);
        assert_eq!(dataset.input_size(), 2);
        assert_eq!(dataset.target_size(), 2);
        assert_eq!(dataset.features().dims(), &[6, 2]);
        assert_eq!(dataset.targets().dims(), &[6, 2]);
        assert_eq!(
            dataset.targets().data(),
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn spiral_dataset_is_deterministic_for_known_points() {
        let dataset = spiral(3, 2).expect("valid spiral dataset");

        assert_slice_close(&dataset.features().data()[0..2], &[0.0, 0.0]);
        assert_slice_close(&dataset.features().data()[2..4], &[0.0, -0.5]);
        assert_slice_close(&dataset.features().data()[4..6], &[0.0, 1.0]);
    }

    #[test]
    fn spiral_rejects_invalid_configuration() {
        assert_eq!(
            spiral(0, 2).expect_err("zero samples should fail"),
            RustGradError::InvalidArgument {
                name: "samples_per_class",
                reason: "value must be greater than zero".to_string(),
            }
        );
        assert_eq!(
            spiral(2, 1).expect_err("one class should fail"),
            RustGradError::InvalidArgument {
                name: "classes",
                reason: "classes must be at least 2".to_string(),
            }
        );
    }

    #[test]
    fn one_hot_encodes_class_index() {
        assert_eq!(
            one_hot(2, 4).expect("valid class"),
            vec![0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn one_hot_rejects_zero_classes() {
        let error = one_hot(0, 0).expect_err("zero classes should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "classes",
                reason: "classes must be greater than zero".to_string(),
            }
        );
    }

    #[test]
    fn one_hot_rejects_out_of_range_class() {
        let error = one_hot(3, 3).expect_err("class index should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "class_index",
                reason: "class index 3 is out of range for 3 classes".to_string(),
            }
        );
    }
}
