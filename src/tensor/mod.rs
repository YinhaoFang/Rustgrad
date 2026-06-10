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
}
