//! Shared error types used across RustGrad.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Convenient result alias for operations that can fail inside RustGrad.
pub type Result<T> = std::result::Result<T, RustGradError>;

/// Errors returned by tensor operations, autograd, training, and CLI helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RustGradError {
    /// A tensor shape contains no dimensions.
    EmptyShape,
    /// A tensor contains no values where at least one value is required.
    EmptyTensor,
    /// The provided data length does not match the requested shape.
    ShapeDataMismatch {
        /// Tensor shape as a list of dimensions.
        shape: Vec<usize>,
        /// Number of values required by the shape.
        expected_len: usize,
        /// Number of values provided by the caller.
        actual_len: usize,
    },
    /// Two shapes cannot participate in the requested operation.
    ShapeMismatch {
        /// Name of the operation that failed.
        op: &'static str,
        /// Left-hand shape.
        left: Vec<usize>,
        /// Right-hand shape.
        right: Vec<usize>,
    },
    /// Matrix multiplication dimensions are incompatible.
    MatMulShapeMismatch {
        /// Left matrix shape.
        left: Vec<usize>,
        /// Right matrix shape.
        right: Vec<usize>,
    },
    /// A requested axis is outside the tensor rank.
    InvalidAxis {
        /// Axis provided by the caller.
        axis: usize,
        /// Number of axes in the tensor.
        rank: usize,
    },
    /// A flat or multidimensional index is outside the tensor bounds.
    IndexOutOfBounds {
        /// Index provided by the caller.
        index: Vec<usize>,
        /// Tensor shape that defines the valid bounds.
        shape: Vec<usize>,
    },
    /// A user-facing argument failed validation.
    InvalidArgument {
        /// Argument name.
        name: &'static str,
        /// Human-readable validation detail.
        reason: String,
    },
    /// An operation exists in the graph but does not yet have a gradient rule.
    UnsupportedOperation {
        /// Operation name.
        op: String,
        /// Human-readable explanation.
        reason: String,
    },
}

impl Display for RustGradError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyShape => write!(f, "shape must contain at least one dimension"),
            Self::EmptyTensor => write!(f, "tensor must contain at least one value"),
            Self::ShapeDataMismatch {
                shape,
                expected_len,
                actual_len,
            } => write!(
                f,
                "shape {shape:?} requires {expected_len} values, but got {actual_len}"
            ),
            Self::ShapeMismatch { op, left, right } => {
                write!(f, "{op} cannot operate on shapes {left:?} and {right:?}")
            }
            Self::MatMulShapeMismatch { left, right } => write!(
                f,
                "matmul requires left columns to equal right rows, but got {left:?} and {right:?}"
            ),
            Self::InvalidAxis { axis, rank } => {
                write!(f, "axis {axis} is invalid for tensor rank {rank}")
            }
            Self::IndexOutOfBounds { index, shape } => {
                write!(f, "index {index:?} is out of bounds for shape {shape:?}")
            }
            Self::InvalidArgument { name, reason } => {
                write!(f, "invalid argument `{name}`: {reason}")
            }
            Self::UnsupportedOperation { op, reason } => {
                write!(f, "unsupported operation `{op}`: {reason}")
            }
        }
    }
}

impl Error for RustGradError {}

#[cfg(test)]
mod tests {
    use super::{Result, RustGradError};
    use std::error::Error;

    fn accepts_std_error(error: &dyn Error) -> String {
        error.to_string()
    }

    #[test]
    fn formats_shape_data_mismatch() {
        let error = RustGradError::ShapeDataMismatch {
            shape: vec![2, 3],
            expected_len: 6,
            actual_len: 5,
        };

        assert_eq!(
            error.to_string(),
            "shape [2, 3] requires 6 values, but got 5"
        );
    }

    #[test]
    fn implements_std_error() {
        let error = RustGradError::EmptyTensor;

        assert_eq!(
            accepts_std_error(&error),
            "tensor must contain at least one value"
        );
    }

    #[test]
    fn result_alias_uses_rustgrad_error() {
        fn always_fails() -> Result<()> {
            Err(RustGradError::InvalidAxis { axis: 2, rank: 2 })
        }

        assert_eq!(
            always_fails().expect_err("expected invalid axis"),
            RustGradError::InvalidAxis { axis: 2, rank: 2 }
        );
    }
}
