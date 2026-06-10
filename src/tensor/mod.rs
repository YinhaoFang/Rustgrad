//! Tensor types and tensor operations.

use crate::{Result, RustGradError};

/// Shape metadata for a tensor.
///
/// RustGrad starts with dense one-dimensional and two-dimensional tensors, but
/// the shape type accepts any non-empty list of positive dimensions so later
/// modules can add higher-rank operations without changing the public type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a shape from a list of dimensions.
    ///
    /// Every dimension must be greater than zero. Zero-sized tensors are kept
    /// out of the first version because they add many edge cases to reductions,
    /// training metrics, and error reporting.
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        if dims.is_empty() {
            return Err(RustGradError::EmptyShape);
        }

        if let Some(index) = dims.iter().position(|dim| *dim == 0) {
            return Err(RustGradError::InvalidArgument {
                name: "dims",
                reason: format!("dimension {index} must be greater than zero"),
            });
        }

        Ok(Self { dims })
    }

    /// Creates a scalar-like shape represented as one stored value.
    #[must_use]
    pub fn scalar() -> Self {
        Self { dims: vec![1] }
    }

    /// Creates a vector shape.
    pub fn vector(len: usize) -> Result<Self> {
        Self::new(vec![len])
    }

    /// Creates a matrix shape.
    pub fn matrix(rows: usize, cols: usize) -> Result<Self> {
        Self::new(vec![rows, cols])
    }

    /// Returns the dimensions as a slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns an owned copy of the dimensions.
    #[must_use]
    pub fn to_vec(&self) -> Vec<usize> {
        self.dims.clone()
    }

    /// Returns the tensor rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns the number of stored values implied by this shape.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns true when this shape is represented by one stored value.
    #[must_use]
    pub fn is_scalar_like(&self) -> bool {
        self.element_count() == 1
    }

    /// Returns true for a one-dimensional shape.
    #[must_use]
    pub fn is_vector(&self) -> bool {
        self.rank() == 1
    }

    /// Returns true for a two-dimensional shape.
    #[must_use]
    pub fn is_matrix(&self) -> bool {
        self.rank() == 2
    }

    /// Returns the matrix row count when the shape is two-dimensional.
    #[must_use]
    pub fn rows(&self) -> Option<usize> {
        self.is_matrix().then_some(self.dims[0])
    }

    /// Returns the matrix column count when the shape is two-dimensional.
    #[must_use]
    pub fn cols(&self) -> Option<usize> {
        self.is_matrix().then_some(self.dims[1])
    }

    /// Converts a multidimensional index into a row-major flat offset.
    pub fn offset(&self, index: &[usize]) -> Result<usize> {
        if index.len() != self.rank() {
            return Err(RustGradError::IndexOutOfBounds {
                index: index.to_vec(),
                shape: self.to_vec(),
            });
        }

        let mut offset = 0;
        let mut stride = 1;
        for (&idx, &dim) in index.iter().zip(self.dims.iter()).rev() {
            if idx >= dim {
                return Err(RustGradError::IndexOutOfBounds {
                    index: index.to_vec(),
                    shape: self.to_vec(),
                });
            }

            offset += idx * stride;
            stride *= dim;
        }

        Ok(offset)
    }
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.dims
    }
}

/// Dense tensor storage used by RustGrad operations.
///
/// Values are stored in row-major order. Computation graph metadata will be
/// added by the autograd module later; this type only owns numeric data and
/// shape information.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Shape,
    data: Vec<f64>,
}

impl Tensor {
    /// Creates a tensor from explicit shape metadata and dense values.
    pub fn new(shape: Shape, data: Vec<f64>) -> Result<Self> {
        if data.is_empty() {
            return Err(RustGradError::EmptyTensor);
        }

        let expected_len = shape.element_count();
        let actual_len = data.len();
        if expected_len != actual_len {
            return Err(RustGradError::ShapeDataMismatch {
                shape: shape.to_vec(),
                expected_len,
                actual_len,
            });
        }

        Ok(Self { shape, data })
    }

    /// Creates a tensor from raw dimensions and dense values.
    pub fn from_vec(dims: impl Into<Vec<usize>>, data: Vec<f64>) -> Result<Self> {
        Self::new(Shape::new(dims)?, data)
    }

    /// Creates a scalar-like tensor represented by one value.
    pub fn scalar(value: f64) -> Result<Self> {
        Self::new(Shape::scalar(), vec![value])
    }

    /// Creates a one-dimensional tensor.
    pub fn vector(values: Vec<f64>) -> Result<Self> {
        Self::new(Shape::vector(values.len())?, values)
    }

    /// Creates a two-dimensional tensor from row-major values.
    pub fn matrix(rows: usize, cols: usize, values: Vec<f64>) -> Result<Self> {
        Self::new(Shape::matrix(rows, cols)?, values)
    }

    /// Creates a tensor filled with zeros.
    pub fn zeros(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        Ok(Self {
            data: vec![0.0; shape.element_count()],
            shape,
        })
    }

    /// Creates a tensor filled with ones.
    pub fn ones(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        Ok(Self {
            data: vec![1.0; shape.element_count()],
            shape,
        })
    }

    /// Creates a tensor filled with a caller-provided value.
    pub fn full(dims: impl Into<Vec<usize>>, value: f64) -> Result<Self> {
        let shape = Shape::new(dims)?;
        Ok(Self {
            data: vec![value; shape.element_count()],
            shape,
        })
    }

    /// Returns the tensor shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the tensor dimensions.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Returns the tensor rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the number of stored values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true when the tensor has no stored values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the dense values in row-major order.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns mutable dense values in row-major order.
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Consumes the tensor and returns the underlying dense values.
    #[must_use]
    pub fn into_data(self) -> Vec<f64> {
        self.data
    }

    /// Returns the matrix row count when the tensor is two-dimensional.
    #[must_use]
    pub fn rows(&self) -> Option<usize> {
        self.shape.rows()
    }

    /// Returns the matrix column count when the tensor is two-dimensional.
    #[must_use]
    pub fn cols(&self) -> Option<usize> {
        self.shape.cols()
    }

    /// Returns a value by flat row-major index.
    pub fn get_flat(&self, index: usize) -> Result<f64> {
        self.data
            .get(index)
            .copied()
            .ok_or_else(|| RustGradError::IndexOutOfBounds {
                index: vec![index],
                shape: self.shape.to_vec(),
            })
    }

    /// Updates a value by flat row-major index.
    pub fn set_flat(&mut self, index: usize, value: f64) -> Result<()> {
        let shape = self.shape.to_vec();
        let slot = self
            .data
            .get_mut(index)
            .ok_or_else(|| RustGradError::IndexOutOfBounds {
                index: vec![index],
                shape,
            })?;

        *slot = value;
        Ok(())
    }

    /// Returns a value by multidimensional index.
    pub fn get(&self, index: &[usize]) -> Result<f64> {
        self.get_flat(self.shape.offset(index)?)
    }

    /// Updates a value by multidimensional index.
    pub fn set(&mut self, index: &[usize], value: f64) -> Result<()> {
        self.set_flat(self.shape.offset(index)?, value)
    }

    /// Returns a new tensor with the same data and a different compatible shape.
    pub fn reshape(&self, dims: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let expected_len = self.len();
        let actual_len = shape.element_count();

        if expected_len != actual_len {
            return Err(RustGradError::ShapeDataMismatch {
                shape: shape.to_vec(),
                expected_len,
                actual_len,
            });
        }

        Ok(Self {
            shape,
            data: self.data.clone(),
        })
    }

    /// Returns a one-dimensional tensor with the same data.
    pub fn flatten(&self) -> Result<Self> {
        self.reshape(vec![self.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::{Shape, Tensor};
    use crate::RustGradError;

    #[test]
    fn creates_valid_shape_metadata() {
        let shape = Shape::new(vec![2, 3, 4]).expect("shape should be valid");

        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.to_vec(), vec![2, 3, 4]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.element_count(), 24);
        assert!(!shape.is_scalar_like());
        assert!(!shape.is_vector());
        assert!(!shape.is_matrix());
        assert_eq!(shape.rows(), None);
        assert_eq!(shape.cols(), None);
    }

    #[test]
    fn creates_scalar_vector_and_matrix_shapes() {
        let scalar = Shape::scalar();
        let vector = Shape::vector(5).expect("vector shape should be valid");
        let matrix = Shape::matrix(2, 3).expect("matrix shape should be valid");

        assert_eq!(scalar.dims(), &[1]);
        assert!(scalar.is_scalar_like());
        assert_eq!(vector.dims(), &[5]);
        assert!(vector.is_vector());
        assert_eq!(matrix.dims(), &[2, 3]);
        assert!(matrix.is_matrix());
        assert_eq!(matrix.rows(), Some(2));
        assert_eq!(matrix.cols(), Some(3));
    }

    #[test]
    fn rejects_empty_shape() {
        let error = Shape::new(Vec::<usize>::new()).expect_err("empty shape should fail");

        assert_eq!(error, RustGradError::EmptyShape);
    }

    #[test]
    fn rejects_zero_dimension_shape() {
        let error = Shape::new(vec![2, 0, 4]).expect_err("zero dimension should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "dims",
                reason: "dimension 1 must be greater than zero".to_string(),
            }
        );
    }

    #[test]
    fn converts_shape_into_dimension_vector() {
        let shape = Shape::matrix(3, 2).expect("matrix shape should be valid");
        let dims: Vec<usize> = shape.into();

        assert_eq!(dims, vec![3, 2]);
    }

    #[test]
    fn creates_tensor_from_shape_and_data() {
        let shape = Shape::matrix(2, 2).expect("matrix shape should be valid");
        let tensor = Tensor::new(shape, vec![1.0, 2.0, 3.0, 4.0]).expect("tensor should be valid");

        assert_eq!(tensor.dims(), &[2, 2]);
        assert_eq!(tensor.shape().element_count(), 4);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.len(), 4);
        assert!(!tensor.is_empty());
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor.rows(), Some(2));
        assert_eq!(tensor.cols(), Some(2));
    }

    #[test]
    fn creates_tensor_from_raw_dimensions() {
        let tensor =
            Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid tensor");

        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn creates_scalar_vector_and_matrix_tensors() {
        let scalar = Tensor::scalar(3.5).expect("scalar tensor should be valid");
        let vector = Tensor::vector(vec![1.0, 2.0, 3.0]).expect("vector tensor should be valid");
        let matrix =
            Tensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix tensor should be valid");

        assert_eq!(scalar.dims(), &[1]);
        assert_eq!(scalar.data(), &[3.5]);
        assert_eq!(vector.dims(), &[3]);
        assert_eq!(vector.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(matrix.dims(), &[2, 2]);
        assert_eq!(matrix.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn creates_filled_tensors() {
        let zeros = Tensor::zeros(vec![2, 3]).expect("zeros tensor should be valid");
        let ones = Tensor::ones(vec![2, 2]).expect("ones tensor should be valid");
        let full = Tensor::full(vec![3], 7.0).expect("full tensor should be valid");

        assert_eq!(zeros.data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(ones.data(), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(full.data(), &[7.0, 7.0, 7.0]);
    }

    #[test]
    fn rejects_tensor_with_empty_data() {
        let shape = Shape::vector(3).expect("shape should be valid");
        let error = Tensor::new(shape, Vec::new()).expect_err("empty tensor should fail");

        assert_eq!(error, RustGradError::EmptyTensor);
    }

    #[test]
    fn rejects_shape_data_length_mismatch() {
        let error =
            Tensor::matrix(2, 3, vec![1.0, 2.0, 3.0]).expect_err("data length should not match");

        assert_eq!(
            error,
            RustGradError::ShapeDataMismatch {
                shape: vec![2, 3],
                expected_len: 6,
                actual_len: 3,
            }
        );
    }

    #[test]
    fn rejects_vector_constructor_with_empty_values() {
        let error = Tensor::vector(Vec::new()).expect_err("empty vector tensor should fail");

        assert_eq!(
            error,
            RustGradError::InvalidArgument {
                name: "dims",
                reason: "dimension 0 must be greater than zero".to_string(),
            }
        );
    }

    #[test]
    fn allows_mutating_dense_data() {
        let mut tensor = Tensor::ones(vec![3]).expect("tensor should be valid");

        tensor.data_mut()[1] = 5.0;

        assert_eq!(tensor.data(), &[1.0, 5.0, 1.0]);
    }

    #[test]
    fn consumes_tensor_into_data() {
        let tensor = Tensor::vector(vec![2.0, 4.0, 6.0]).expect("tensor should be valid");

        assert_eq!(tensor.into_data(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn computes_row_major_offsets() {
        let shape = Shape::new(vec![2, 3, 4]).expect("shape should be valid");

        assert_eq!(shape.offset(&[0, 0, 0]).expect("offset should exist"), 0);
        assert_eq!(shape.offset(&[0, 1, 2]).expect("offset should exist"), 6);
        assert_eq!(shape.offset(&[1, 2, 3]).expect("offset should exist"), 23);
    }

    #[test]
    fn rejects_invalid_multidimensional_index() {
        let shape = Shape::matrix(2, 3).expect("shape should be valid");

        assert_eq!(
            shape.offset(&[2, 0]).expect_err("row is out of bounds"),
            RustGradError::IndexOutOfBounds {
                index: vec![2, 0],
                shape: vec![2, 3],
            }
        );
        assert_eq!(
            shape.offset(&[1]).expect_err("rank does not match"),
            RustGradError::IndexOutOfBounds {
                index: vec![1],
                shape: vec![2, 3],
            }
        );
    }

    #[test]
    fn reads_and_writes_flat_values() {
        let mut tensor = Tensor::vector(vec![1.0, 2.0, 3.0]).expect("tensor should be valid");

        assert_eq!(tensor.get_flat(1).expect("flat index should exist"), 2.0);
        tensor
            .set_flat(1, 9.0)
            .expect("flat index should be writable");

        assert_eq!(tensor.data(), &[1.0, 9.0, 3.0]);
    }

    #[test]
    fn rejects_out_of_bounds_flat_access() {
        let mut tensor = Tensor::vector(vec![1.0, 2.0]).expect("tensor should be valid");

        assert_eq!(
            tensor.get_flat(3).expect_err("flat index is out of bounds"),
            RustGradError::IndexOutOfBounds {
                index: vec![3],
                shape: vec![2],
            }
        );
        assert_eq!(
            tensor
                .set_flat(3, 4.0)
                .expect_err("flat index is out of bounds"),
            RustGradError::IndexOutOfBounds {
                index: vec![3],
                shape: vec![2],
            }
        );
    }

    #[test]
    fn reads_and_writes_multidimensional_values() {
        let mut tensor =
            Tensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid matrix");

        assert_eq!(tensor.get(&[1, 1]).expect("index should exist"), 5.0);
        tensor.set(&[0, 2], 8.0).expect("index should be writable");

        assert_eq!(tensor.data(), &[1.0, 2.0, 8.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reshapes_tensor_when_element_count_matches() {
        let tensor = Tensor::vector(vec![1.0, 2.0, 3.0, 4.0]).expect("valid vector");
        let reshaped = tensor.reshape(vec![2, 2]).expect("reshape should match");

        assert_eq!(reshaped.dims(), &[2, 2]);
        assert_eq!(reshaped.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_reshape_with_different_element_count() {
        let tensor = Tensor::vector(vec![1.0, 2.0, 3.0, 4.0]).expect("valid vector");

        assert_eq!(
            tensor
                .reshape(vec![3, 2])
                .expect_err("reshape should not match"),
            RustGradError::ShapeDataMismatch {
                shape: vec![3, 2],
                expected_len: 4,
                actual_len: 6,
            }
        );
    }

    #[test]
    fn flattens_tensor() {
        let tensor = Tensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid matrix");
        let flattened = tensor.flatten().expect("flatten should keep all values");

        assert_eq!(flattened.dims(), &[4]);
        assert_eq!(flattened.data(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
