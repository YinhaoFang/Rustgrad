//! Plain-text model checkpoint serialization.
//!
//! The format is line-oriented and human-readable:
//! ```text
//! rank,dim0,dim1,...
//! value0
//! value1
//! ...
//! ```
//!
//! Multiple tensors are concatenated. Models write one component per tensor
//! in parameter order so that `load` can reconstruct them.

use crate::nn::Linear;
use crate::tensor::{Shape, Tensor};
use crate::train::XorMlp;
use crate::{Result, RustGradError};
use std::fs;
use std::path::Path;

/// Writes a tensor to a plain-text representation.
pub fn write_tensor(tensor: &Tensor, output: &mut String) {
    output.push_str(&format!("{}", tensor.rank()));
    for &dim in tensor.dims() {
        output.push_str(&format!(",{dim}"));
    }
    output.push('\n');
    for &value in tensor.data() {
        output.push_str(&format!("{value:.17e}\n"));
    }
}

/// Reads one tensor from its text representation.
///
/// Returns the tensor and the number of bytes consumed.
pub fn read_tensor(input: &str) -> Result<(Tensor, usize)> {
    let header_end = input.find('\n').ok_or_else(|| RustGradError::InvalidArgument {
        name: "checkpoint",
        reason: "missing tensor header line".to_string(),
    })?;
    let header = &input[..header_end];
    let dims: Vec<usize> = header
        .split(',')
        .skip(1) // skip rank
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| RustGradError::InvalidArgument {
                    name: "checkpoint",
                    reason: format!("invalid dimension '{s}'"),
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let shape = Shape::new(dims)?;
    let element_count = shape.element_count();
    let mut data = Vec::with_capacity(element_count);
    let mut cursor = header_end + 1;

    for _ in 0..element_count {
        let line_end = input[cursor..]
            .find('\n')
            .ok_or_else(|| RustGradError::InvalidArgument {
                name: "checkpoint",
                reason: format!("expected {element_count} values, found {}", data.len()),
            })?;
        let line = &input[cursor..cursor + line_end];
        let value: f64 = line.trim().parse().map_err(|_| RustGradError::InvalidArgument {
            name: "checkpoint",
            reason: format!("invalid float '{line}'"),
        })?;
        data.push(value);
        cursor += line_end + 1;
    }

    Ok((Tensor::new(shape, data)?, cursor))
}

/// Saves a linear layer to a file.
pub fn save_linear(linear: &Linear, path: &Path) -> Result<()> {
    let mut content = String::new();
    write_tensor(linear.weights(), &mut content);
    write_tensor(linear.bias(), &mut content);
    fs::write(path, content).map_err(|e| RustGradError::InvalidArgument {
        name: "path",
        reason: format!("failed to write checkpoint: {e}"),
    })
}

/// Loads a linear layer from a file saved with `save_linear`.
pub fn load_linear(path: &Path) -> Result<Linear> {
    let content = fs::read_to_string(path).map_err(|e| RustGradError::InvalidArgument {
        name: "path",
        reason: format!("failed to read checkpoint: {e}"),
    })?;

    let (weights, c1) = read_tensor(&content)?;
    let (bias, _c2) = read_tensor(&content[c1..])?;

    Linear::with_parameters(weights, bias)
}

/// Saves a two-layer XOR MLP to a file.
pub fn save_xor_mlp(model: &XorMlp, path: &Path) -> Result<()> {
    let mut content = String::new();
    write_tensor(model.hidden().weights(), &mut content);
    write_tensor(model.hidden().bias(), &mut content);
    write_tensor(model.output().weights(), &mut content);
    write_tensor(model.output().bias(), &mut content);
    fs::write(path, content).map_err(|e| RustGradError::InvalidArgument {
        name: "path",
        reason: format!("failed to write checkpoint: {e}"),
    })
}

/// Loads a two-layer XOR MLP from a file saved with `save_xor_mlp`.
pub fn load_xor_mlp(path: &Path) -> Result<XorMlp> {
    let content = fs::read_to_string(path).map_err(|e| RustGradError::InvalidArgument {
        name: "path",
        reason: format!("failed to read checkpoint: {e}"),
    })?;

    let (hidden_weights, c1) = read_tensor(&content)?;
    let offset = c1;
    let (hidden_bias, c2) = read_tensor(&content[offset..])?;
    let offset = offset + c2;
    let (output_weights, c3) = read_tensor(&content[offset..])?;
    let offset = offset + c3;
    let (output_bias, _) = read_tensor(&content[offset..])?;

    Ok(XorMlp::from_parameters(
        hidden_weights,
        hidden_bias,
        output_weights,
        output_bias,
        0.5,
    )?)
}

#[cfg(test)]
mod tests {
    use super::{load_linear, load_xor_mlp, read_tensor, save_linear, save_xor_mlp, write_tensor};
    use crate::nn::{Linear, Module};
    use crate::tensor::Tensor;
    use crate::train::XorMlp;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("rustgrad-serialize-{name}-{suffix}"))
    }

    fn assert_slice_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-15, "expected {e}, got {a}");
        }
    }

    #[test]
    fn write_tensor_produces_rank_and_data_lines() {
        let tensor = Tensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid matrix");
        let mut output = String::new();
        write_tensor(&tensor, &mut output);

        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines[0], "2,2,2");
        assert!(lines.len() == 5); // header + 4 values
    }

    #[test]
    fn write_tensor_handles_vector() {
        let tensor = Tensor::vector(vec![1.5, -2.5, 0.0]).expect("valid vector");
        let mut output = String::new();
        write_tensor(&tensor, &mut output);

        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines[0], "1,3");
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn roundtrip_tensor_preserves_values() {
        let original = Tensor::matrix(3, 2, vec![1.0, 2.0, -3.5, 0.125, 1e-10, 0.0])
            .expect("valid matrix");
        let mut text = String::new();
        write_tensor(&original, &mut text);

        let (restored, _) = read_tensor(&text).expect("read should succeed");

        assert_eq!(restored.dims(), original.dims());
        assert_slice_close(restored.data(), original.data());
    }

    #[test]
    fn roundtrip_vector_tensor_preserves_values() {
        let original = Tensor::vector(vec![0.0, std::f64::consts::PI, -std::f64::consts::E]).expect("valid");
        let mut text = String::new();
        write_tensor(&original, &mut text);

        let (restored, _) = read_tensor(&text).expect("read should succeed");

        assert_eq!(restored.dims(), original.dims());
        assert_slice_close(restored.data(), original.data());
    }

    #[test]
    fn read_tensor_rejects_truncated_data() {
        let text = "1,3\n1.0\n2.0\n";
        // Only 2 values, expected 3.

        let error = read_tensor(text).expect_err("truncated should fail");
        assert!(error.to_string().contains("expected 3 values"));
    }

    #[test]
    fn read_tensor_rejects_invalid_dimension() {
        let text = "2,abc,2\n1.0\n2.0\n3.0\n4.0\n";

        let error = read_tensor(text).expect_err("invalid dim should fail");
        assert!(error.to_string().contains("invalid dimension"));
    }

    #[test]
    fn save_and_load_linear_roundtrips() {
        let dir = unique_temp_dir("linear");
        fs::create_dir_all(&dir).expect("dir created");
        let path = dir.join("linear.checkpoint");

        let original = Linear::new(3, 2).expect("valid layer");
        save_linear(&original, &path).expect("save succeeds");

        let restored = load_linear(&path).expect("load succeeds");

        assert_eq!(restored.input_size(), original.input_size());
        assert_eq!(restored.output_size(), original.output_size());
        assert_slice_close(restored.weights().data(), original.weights().data());
        assert_slice_close(restored.bias().data(), original.bias().data());

        fs::remove_dir_all(&dir).expect("cleanup succeeds");
    }

    #[test]
    fn linear_roundtrip_preserves_forward_output() {
        let dir = unique_temp_dir("linear-fwd");
        fs::create_dir_all(&dir).expect("dir created");
        let path = dir.join("linear.checkpoint");

        let original = Linear::new(2, 3).expect("valid layer");
        let input = Tensor::vector(vec![1.0, 2.0]).expect("valid input");
        let expected = original.forward(&input).expect("forward ok");

        save_linear(&original, &path).expect("save succeeds");
        let restored = load_linear(&path).expect("load succeeds");
        let actual = restored.forward(&input).expect("forward ok");

        assert_slice_close(actual.data(), expected.data());
        fs::remove_dir_all(&dir).expect("cleanup succeeds");
    }

    #[test]
    fn save_and_load_xor_mlp_roundtrips() {
        let dir = unique_temp_dir("xor");
        fs::create_dir_all(&dir).expect("dir created");
        let path = dir.join("xor.checkpoint");

        let original = XorMlp::new(0.5).expect("valid xor model");
        save_xor_mlp(&original, &path).expect("save succeeds");

        let restored = load_xor_mlp(&path).expect("load succeeds");

        assert_slice_close(
            restored.hidden().weights().data(),
            original.hidden().weights().data(),
        );
        assert_slice_close(
            restored.hidden().bias().data(),
            original.hidden().bias().data(),
        );
        assert_slice_close(
            restored.output().weights().data(),
            original.output().weights().data(),
        );
        assert_slice_close(
            restored.output().bias().data(),
            original.output().bias().data(),
        );
        assert_eq!(restored.threshold(), original.threshold());

        fs::remove_dir_all(&dir).expect("cleanup succeeds");
    }
}
