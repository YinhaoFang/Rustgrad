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

    /// Creates a dataset from CSV text content.
    ///
    /// The first line is a header (skipped). Each subsequent line must contain
    /// the feature columns followed by target columns, comma-separated.
    /// `num_features` determines the split: the first `num_features` columns
    /// are features, the rest are targets.
    pub fn from_csv(
        name: impl Into<String>,
        csv_text: &str,
        num_features: usize,
    ) -> Result<Self> {
        if num_features == 0 {
            return Err(RustGradError::InvalidArgument {
                name: "num_features",
                reason: "num_features must be greater than zero".to_string(),
            });
        }

        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut header_skipped = false;
        for (line_idx, line) in csv_text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Skip the first non-comment, non-empty line as header.
            if !header_skipped {
                header_skipped = true;
                continue;
            }

            let values: Vec<f64> = line
                .split(',')
                .map(|s| {
                    s.trim().parse::<f64>().map_err(|_| {
                        RustGradError::InvalidArgument {
                            name: "csv",
                            reason: format!("invalid float in line {}: '{s}'", line_idx + 1),
                        }
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            if values.is_empty() {
                continue;
            }

            let expected_cols = num_features + 1; // at least one target column
            if values.len() < expected_cols {
                return Err(RustGradError::InvalidArgument {
                    name: "csv",
                    reason: format!(
                        "line {} has {} columns, expected at least {expected_cols}",
                        line_idx + 1,
                        values.len()
                    ),
                });
            }

            rows.push(values);
        }

        if rows.is_empty() {
            return Err(RustGradError::InvalidArgument {
                name: "csv",
                reason: "no data rows found".to_string(),
            });
        }

        let num_rows = rows.len();
        let num_targets = rows[0].len() - num_features;
        let mut features = Vec::with_capacity(num_rows * num_features);
        let mut targets = Vec::with_capacity(num_rows * num_targets);

        for row in &rows {
            features.extend_from_slice(&row[..num_features]);
            targets.extend_from_slice(&row[num_features..]);
        }

        Dataset::new(
            name,
            Tensor::matrix(num_rows, num_features, features)?,
            Tensor::matrix(num_rows, num_targets, targets)?,
        )
    }

    /// Returns an iterator over (features_row, targets_row) pairs.
    pub fn iter_rows(&self) -> DatasetRowIterator<'_> {
        DatasetRowIterator {
            dataset: self,
            index: 0,
        }
    }

    /// Returns a new dataset with rows deterministically shuffled.
    ///
    /// Uses a simple linear congruential generator seeded by `seed`. Same seed
    /// always produces the same permutation.
    pub fn shuffle(&self, seed: u64) -> Result<Self> {
        let n = self.len();
        if n <= 1 {
            return Ok(self.clone());
        }

        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle with LCG.
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for i in (1..n).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let feature_cols = self.input_size();
        let target_cols = self.target_size();
        let mut features = Vec::with_capacity(n * feature_cols);
        let mut targets = Vec::with_capacity(n * target_cols);

        for &row_idx in &indices {
            let f_start = row_idx * feature_cols;
            let f_end = f_start + feature_cols;
            features.extend_from_slice(&self.features.data()[f_start..f_end]);

            let t_start = row_idx * target_cols;
            let t_end = t_start + target_cols;
            targets.extend_from_slice(&self.targets.data()[t_start..t_end]);
        }

        Dataset::new(
            format!("{}_shuffled", self.name),
            Tensor::matrix(n, feature_cols, features)?,
            Tensor::matrix(n, target_cols, targets)?,
        )
    }

    /// Splits the dataset into training and test subsets.
    ///
    /// `ratio` is the fraction allocated to training (e.g., 0.8 for 80/20 split).
    pub fn split(&self, ratio: f64) -> Result<(Dataset, Dataset)> {
        if ratio <= 0.0 || ratio >= 1.0 || !ratio.is_finite() {
            return Err(RustGradError::InvalidArgument {
                name: "ratio",
                reason: "ratio must be finite and in (0, 1)".to_string(),
            });
        }

        let n = self.len();
        let train_rows = (n as f64 * ratio).ceil() as usize;
        if train_rows == 0 || train_rows >= n {
            return Err(RustGradError::InvalidArgument {
                name: "ratio",
                reason: format!(
                    "split ratio {ratio} produces empty subset (n={n}, train={train_rows})"
                ),
            });
        }

        let train = self.batch(0, train_rows)?;
        let test = self.batch(train_rows, n - train_rows)?;

        Ok((
            Dataset::new(format!("{}_train", self.name), train.features().clone(), train.targets().clone())?,
            Dataset::new(format!("{}_test", self.name), test.features().clone(), test.targets().clone())?,
        ))
    }
}

/// Row-wise iterator over a dataset.
pub struct DatasetRowIterator<'a> {
    dataset: &'a Dataset,
    index: usize,
}

impl<'a> Iterator for DatasetRowIterator<'a> {
    type Item = crate::Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.len() {
            return None;
        }
        let result = self.dataset.sample(self.index);
        self.index += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.dataset.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for DatasetRowIterator<'a> {}

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

    // ── CSV, iterator, shuffle, split ────────────────────────────────

    #[test]
    fn from_csv_parses_header_and_data() {
        let csv = "x,y,target\n1.0,2.0,10.0\n3.0,4.0,20.0\n";
        let dataset = Dataset::from_csv("csv-test", csv, 2).expect("valid csv");

        assert_eq!(dataset.name(), "csv-test");
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.input_size(), 2);
        assert_eq!(dataset.target_size(), 1);
        assert_eq!(dataset.features().data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(dataset.targets().data(), &[10.0, 20.0]);
    }

    #[test]
    fn from_csv_skips_comment_and_empty_lines() {
        let csv = "# comment\nx,y,t\n1.0,2.0,3.0\n\n4.0,5.0,6.0\n";
        let dataset = Dataset::from_csv("csv-test", csv, 2).expect("valid csv");
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.features().data(), &[1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn from_csv_rejects_insufficient_columns() {
        let csv = "x,t\n1.0,2.0\n3.0\n";
        let error = Dataset::from_csv("bad", csv, 1).expect_err("short line");
        assert!(error.to_string().contains("has 1 columns"));
    }

    #[test]
    fn from_csv_rejects_invalid_float() {
        let csv = "x,t\nabc,1.0\n";
        let error = Dataset::from_csv("bad", csv, 1).expect_err("bad float");
        assert!(error.to_string().contains("invalid float"));
    }

    #[test]
    fn from_csv_rejects_empty_data() {
        let csv = "x,t\n";
        let error = Dataset::from_csv("empty", csv, 1).expect_err("no data");
        assert!(error.to_string().contains("no data rows found"));
    }

    #[test]
    fn iter_rows_yields_all_samples() {
        let dataset = xor().expect("valid xor");
        let mut count = 0;
        for item in dataset.iter_rows() {
            let (_features, _targets) = item.expect("valid row");
            count += 1;
        }
        assert_eq!(count, 4);
    }

    #[test]
    fn iter_rows_is_exact_size() {
        let dataset = xor().expect("valid xor");
        let iter = dataset.iter_rows();
        assert_eq!(iter.len(), 4);
    }

    #[test]
    fn shuffle_preserves_all_rows() {
        let dataset = xor().expect("valid xor");
        let shuffled = dataset.shuffle(42).expect("shuffle succeeds");

        assert_eq!(shuffled.len(), dataset.len());
        assert_eq!(shuffled.input_size(), dataset.input_size());
        assert_eq!(shuffled.target_size(), dataset.target_size());
    }

    #[test]
    fn shuffle_is_deterministic() {
        let dataset = linear_regression(5, 2.0, 1.0).expect("valid");
        let a = dataset.shuffle(123).expect("shuffle");
        let b = dataset.shuffle(123).expect("shuffle");
        assert_eq!(a.features().data(), b.features().data());
        assert_eq!(a.targets().data(), b.targets().data());
    }

    #[test]
    fn shuffle_different_seeds_differ() {
        let dataset = linear_regression(20, 2.0, 1.0).expect("valid");
        let a = dataset.shuffle(111).expect("shuffle");
        let b = dataset.shuffle(222).expect("shuffle");
        assert_ne!(a.features().data(), b.features().data());
    }

    #[test]
    fn shuffle_handles_single_row() {
        let dataset = linear_regression(1, 1.0, 0.0).expect("valid");
        let shuffled = dataset.shuffle(0).expect("shuffle");
        assert_eq!(shuffled.features().data(), dataset.features().data());
    }

    #[test]
    fn split_eighty_twenty() {
        let dataset = linear_regression(10, 2.0, 0.0).expect("valid");
        let (train, test) = dataset.split(0.8).expect("split should succeed");

        assert_eq!(train.len(), 8);
        assert_eq!(test.len(), 2);
    }

    #[test]
    fn split_handles_minority() {
        let dataset = linear_regression(5, 1.0, 0.0).expect("valid");
        let (train, test) = dataset.split(0.3).expect("split");

        assert_eq!(train.len(), 2);
        assert_eq!(test.len(), 3);
    }

    #[test]
    fn split_rejects_invalid_ratio() {
        let dataset = xor().expect("valid");
        let e0 = dataset.split(0.0).expect_err("zero ratio");
        assert!(e0.to_string().contains("ratio must be finite and in (0, 1)"));

        let e1 = dataset.split(1.0).expect_err("one ratio");
        assert!(e1.to_string().contains("ratio must be finite and in (0, 1)"));

        let enan = dataset.split(f64::NAN).expect_err("nan ratio");
        assert!(enan.to_string().contains("ratio must be finite and in (0, 1)"));
    }
}
