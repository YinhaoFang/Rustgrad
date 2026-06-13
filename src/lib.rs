//! RustGrad is a Rust course project that implements core deep learning pieces.
//!
//! The crate will provide tensors, automatic differentiation,
//! neural network layers, losses, optimizers, and small training examples.

pub mod autograd;
pub mod data;
pub mod error;
pub mod loss;
pub mod nn;
pub mod optim;
pub mod report;
pub mod serialize;
pub mod tensor;
pub mod train;

pub use error::{Result, RustGradError};

/// Returns the crate version from Cargo metadata.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::version;

    #[test]
    fn exposes_package_version() {
        assert_eq!(version(), "0.1.0");
    }
}
