//! Automatic differentiation and computation graph utilities.

use std::collections::HashSet;

use crate::tensor::Tensor;
use crate::{Result, RustGradError};

/// Stable identifier for a node in a computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(usize);

impl NodeId {
    /// Creates a node identifier from its raw index.
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the raw graph index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

/// Operation represented by a computation graph node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    /// A leaf tensor directly provided by the user or a trainable parameter.
    Leaf,
    /// Elementwise addition.
    Add,
    /// Elementwise subtraction.
    Sub,
    /// Elementwise multiplication.
    Mul,
    /// Elementwise division.
    Div,
    /// Matrix multiplication.
    MatMul,
    /// Matrix transpose.
    Transpose,
    /// Sum reduction.
    Sum,
    /// Mean reduction.
    Mean,
    /// Rectified linear unit activation.
    Relu,
    /// Sigmoid activation.
    Sigmoid,
    /// Hyperbolic tangent activation.
    Tanh,
    /// Softmax activation.
    Softmax,
    /// Named operation used by future layers or examples.
    Custom(String),
}

impl Operation {
    /// Returns a human-readable operation name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Leaf => "leaf",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::MatMul => "matmul",
            Self::Transpose => "transpose",
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::Softmax => "softmax",
            Self::Custom(name) => name,
        }
    }
}

/// A single value-producing node in the computation graph.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphNode {
    id: NodeId,
    value: Tensor,
    parents: Vec<NodeId>,
    operation: Operation,
    requires_grad: bool,
    grad: Option<Tensor>,
}

impl GraphNode {
    /// Creates a leaf node.
    #[must_use]
    pub fn leaf(id: NodeId, value: Tensor, requires_grad: bool) -> Self {
        Self {
            id,
            value,
            parents: Vec::new(),
            operation: Operation::Leaf,
            requires_grad,
            grad: None,
        }
    }

    /// Creates a node produced by an operation.
    #[must_use]
    pub fn operation(
        id: NodeId,
        value: Tensor,
        parents: Vec<NodeId>,
        operation: Operation,
        requires_grad: bool,
    ) -> Self {
        Self {
            id,
            value,
            parents,
            operation,
            requires_grad,
            grad: None,
        }
    }

    /// Returns this node's identifier.
    #[must_use]
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Returns the tensor value produced by this node.
    #[must_use]
    pub fn value(&self) -> &Tensor {
        &self.value
    }

    /// Returns the parent node identifiers.
    #[must_use]
    pub fn parents(&self) -> &[NodeId] {
        &self.parents
    }

    /// Returns the operation that produced this node.
    #[must_use]
    pub fn operation_kind(&self) -> &Operation {
        &self.operation
    }

    /// Returns true when gradients should be tracked for this node.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Returns the currently accumulated gradient.
    #[must_use]
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    /// Replaces the currently accumulated gradient.
    pub fn set_grad(&mut self, grad: Tensor) {
        self.grad = Some(grad);
    }

    /// Adds a gradient contribution to this node.
    pub fn accumulate_grad(&mut self, grad: Tensor) -> Result<()> {
        self.grad = Some(match self.grad.take() {
            Some(existing) => existing.add(&grad)?,
            None => grad,
        });
        Ok(())
    }

    /// Removes and returns the currently accumulated gradient.
    pub fn take_grad(&mut self) -> Option<Tensor> {
        self.grad.take()
    }

    /// Clears the currently accumulated gradient.
    pub fn clear_grad(&mut self) {
        self.grad = None;
    }
}

/// Append-only computation graph used by automatic differentiation.
#[derive(Debug, Clone, Default)]
pub struct ComputationGraph {
    nodes: Vec<GraphNode>,
}

impl ComputationGraph {
    /// Creates an empty computation graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of nodes in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true when the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Adds a leaf value and returns its node identifier.
    pub fn add_leaf(&mut self, value: Tensor, requires_grad: bool) -> NodeId {
        let id = self.next_id();
        self.nodes.push(GraphNode::leaf(id, value, requires_grad));
        id
    }

    /// Adds an operation node and returns its node identifier.
    pub fn add_operation(
        &mut self,
        operation: Operation,
        parents: Vec<NodeId>,
        value: Tensor,
        requires_grad: bool,
    ) -> Result<NodeId> {
        self.validate_parents(&parents)?;

        let id = self.next_id();
        self.nodes.push(GraphNode::operation(
            id,
            value,
            parents,
            operation,
            requires_grad,
        ));
        Ok(id)
    }

    /// Returns a node by identifier.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(id.index())
    }

    /// Returns a mutable node by identifier.
    #[must_use]
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(id.index())
    }

    /// Iterates over graph nodes in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &GraphNode> {
        self.nodes.iter()
    }

    /// Clears all accumulated gradients.
    pub fn clear_gradients(&mut self) {
        for node in &mut self.nodes {
            node.clear_grad();
        }
    }

    /// Returns node identifiers in dependency-first topological order.
    pub fn topological_order(&self, output: NodeId) -> Result<Vec<NodeId>> {
        self.ensure_node_exists(output)?;

        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.visit_dependencies(output, &mut visited, &mut order)?;
        Ok(order)
    }

    /// Runs a backward pass from the output node.
    ///
    /// This method prepares the graph for automatic differentiation by clearing
    /// old gradients, seeding the output with an all-ones gradient, and walking
    /// nodes in reverse topological order. Operation-specific gradient rules are
    /// implemented incrementally; unsupported non-leaf operations return a
    /// clear error instead of silently producing incorrect gradients.
    pub fn backward(&mut self, output: NodeId) -> Result<()> {
        let order = self.topological_order(output)?;
        self.clear_gradients();
        self.seed_output_gradient(output)?;

        for node_id in order.into_iter().rev() {
            let Some(grad) = self.node(node_id).and_then(|node| node.grad()).cloned() else {
                continue;
            };
            if !self.node(node_id).is_some_and(GraphNode::requires_grad) {
                continue;
            }

            for (parent, parent_grad) in self.local_gradients(node_id, &grad)? {
                if self.node(parent).is_some_and(GraphNode::requires_grad) {
                    self.accumulate_node_grad(parent, parent_grad)?;
                }
            }
        }

        Ok(())
    }

    fn next_id(&self) -> NodeId {
        NodeId::new(self.nodes.len())
    }

    fn ensure_node_exists(&self, id: NodeId) -> Result<()> {
        if self.node(id).is_some() {
            Ok(())
        } else {
            Err(RustGradError::InvalidArgument {
                name: "node",
                reason: format!("node {} does not exist", id.index()),
            })
        }
    }

    fn validate_parents(&self, parents: &[NodeId]) -> Result<()> {
        for parent in parents {
            if self.node(*parent).is_none() {
                return Err(RustGradError::InvalidArgument {
                    name: "parents",
                    reason: format!("node {} does not exist", parent.index()),
                });
            }
        }

        Ok(())
    }

    fn visit_dependencies(
        &self,
        id: NodeId,
        visited: &mut HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) -> Result<()> {
        if !visited.insert(id) {
            return Ok(());
        }

        let node = self
            .node(id)
            .ok_or_else(|| RustGradError::InvalidArgument {
                name: "node",
                reason: format!("node {} does not exist", id.index()),
            })?;

        for parent in node.parents() {
            self.visit_dependencies(*parent, visited, order)?;
        }
        order.push(id);
        Ok(())
    }

    fn seed_output_gradient(&mut self, output: NodeId) -> Result<()> {
        let shape = self
            .node(output)
            .ok_or_else(|| RustGradError::InvalidArgument {
                name: "node",
                reason: format!("node {} does not exist", output.index()),
            })?
            .value()
            .shape()
            .to_vec();
        let grad = Tensor::ones(shape)?;
        self.accumulate_node_grad(output, grad)
    }

    fn accumulate_node_grad(&mut self, id: NodeId, grad: Tensor) -> Result<()> {
        self.node_mut(id)
            .ok_or_else(|| RustGradError::InvalidArgument {
                name: "node",
                reason: format!("node {} does not exist", id.index()),
            })?
            .accumulate_grad(grad)
    }

    fn local_gradients(&self, id: NodeId, upstream_grad: &Tensor) -> Result<Vec<(NodeId, Tensor)>> {
        let node = self
            .node(id)
            .ok_or_else(|| RustGradError::InvalidArgument {
                name: "node",
                reason: format!("node {} does not exist", id.index()),
            })?;

        match node.operation_kind() {
            Operation::Leaf => Ok(Vec::new()),
            Operation::Add => self.add_gradients(node.parents(), upstream_grad),
            Operation::Sub => self.sub_gradients(node.parents(), upstream_grad),
            Operation::Mul => self.mul_gradients(node.parents(), upstream_grad),
            Operation::Div => self.div_gradients(node.parents(), upstream_grad),
            Operation::MatMul => self.matmul_gradients(node.parents(), upstream_grad),
            Operation::Sum => self.sum_gradient(node.parents(), upstream_grad),
            Operation::Mean => self.mean_gradient(node.parents(), upstream_grad),
            Operation::Transpose => self.transpose_gradient(node.parents(), upstream_grad),
            Operation::Relu => self.relu_gradient(node.parents(), upstream_grad),
            Operation::Sigmoid => self.sigmoid_gradient(node.parents(), node.value(), upstream_grad),
            Operation::Tanh => self.tanh_gradient(node.parents(), node.value(), upstream_grad),
            Operation::Softmax => {
                self.softmax_gradient(node.parents(), node.value(), upstream_grad)
            }
            operation => Err(RustGradError::UnsupportedOperation {
                op: operation.name().to_string(),
                reason: format!(
                    "gradient rule is not implemented yet for upstream shape {:?}",
                    upstream_grad.dims()
                ),
            }),
        }
    }

    fn add_gradients(
        &self,
        parents: &[NodeId],
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (left_id, left_value, right_id, right_value) =
            self.binary_parent_values(parents, "add")?;

        Ok(vec![
            (
                left_id,
                Self::fit_gradient_to_parent(&left_value, upstream_grad.clone())?,
            ),
            (
                right_id,
                Self::fit_gradient_to_parent(&right_value, upstream_grad.clone())?,
            ),
        ])
    }

    fn sub_gradients(
        &self,
        parents: &[NodeId],
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (left_id, left_value, right_id, right_value) =
            self.binary_parent_values(parents, "sub")?;
        let negative_upstream = upstream_grad.mul(&Tensor::scalar(-1.0)?)?;

        Ok(vec![
            (
                left_id,
                Self::fit_gradient_to_parent(&left_value, upstream_grad.clone())?,
            ),
            (
                right_id,
                Self::fit_gradient_to_parent(&right_value, negative_upstream)?,
            ),
        ])
    }

    fn mul_gradients(
        &self,
        parents: &[NodeId],
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (left_id, left_value, right_id, right_value) =
            self.binary_parent_values(parents, "mul")?;
        let left_grad = upstream_grad.mul(&right_value)?;
        let right_grad = upstream_grad.mul(&left_value)?;

        Ok(vec![
            (
                left_id,
                Self::fit_gradient_to_parent(&left_value, left_grad)?,
            ),
            (
                right_id,
                Self::fit_gradient_to_parent(&right_value, right_grad)?,
            ),
        ])
    }

    fn div_gradients(
        &self,
        parents: &[NodeId],
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (left_id, left_value, right_id, right_value) =
            self.binary_parent_values(parents, "div")?;
        let left_grad = upstream_grad.div(&right_value)?;
        let right_squared = right_value.mul(&right_value)?;
        let right_grad = upstream_grad
            .mul(&left_value)?
            .div(&right_squared)?
            .mul(&Tensor::scalar(-1.0)?)?;

        Ok(vec![
            (
                left_id,
                Self::fit_gradient_to_parent(&left_value, left_grad)?,
            ),
            (
                right_id,
                Self::fit_gradient_to_parent(&right_value, right_grad)?,
            ),
        ])
    }

    fn matmul_gradients(
        &self,
        parents: &[NodeId],
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (left_id, left_value, right_id, right_value) =
            self.binary_parent_values(parents, "matmul")?;
        let left_grad = upstream_grad.matmul(&right_value.transpose()?)?;
        let right_grad = left_value.transpose()?.matmul(upstream_grad)?;

        Ok(vec![(left_id, left_grad), (right_id, right_grad)])
    }

    fn sum_gradient(
        &self,
        parents: &[NodeId],
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (parent_id, parent_value) = self.unary_parent_value(parents, "sum")?;
        let grad = Self::broadcast_reduction_gradient(&parent_value, upstream_grad, 1.0)?;

        Ok(vec![(parent_id, grad)])
    }

    fn mean_gradient(
        &self,
        parents: &[NodeId],
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (parent_id, parent_value) = self.unary_parent_value(parents, "mean")?;
        let scale = 1.0 / parent_value.len() as f64;
        let grad = Self::broadcast_reduction_gradient(&parent_value, upstream_grad, scale)?;

        Ok(vec![(parent_id, grad)])
    }

    fn transpose_gradient(
        &self,
        parents: &[NodeId],
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (parent_id, _parent_value) = self.unary_parent_value(parents, "transpose")?;
        let grad = upstream_grad.transpose()?;
        Ok(vec![(parent_id, grad)])
    }

    fn relu_gradient(
        &self,
        parents: &[NodeId],
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (parent_id, parent_value) = self.unary_parent_value(parents, "relu")?;
        let derivative_data: Vec<f64> = parent_value
            .data()
            .iter()
            .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let derivative = Tensor::from_vec(parent_value.shape().to_vec(), derivative_data)?;
        let grad = upstream_grad.mul(&derivative)?;
        Ok(vec![(parent_id, grad)])
    }

    fn sigmoid_gradient(
        &self,
        parents: &[NodeId],
        output: &Tensor,
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (parent_id, _parent_value) = self.unary_parent_value(parents, "sigmoid")?;
        let derivative_data: Vec<f64> = output
            .data()
            .iter()
            .map(|&v| v * (1.0 - v))
            .collect();
        let derivative = Tensor::from_vec(output.shape().to_vec(), derivative_data)?;
        let grad = upstream_grad.mul(&derivative)?;
        Ok(vec![(parent_id, grad)])
    }

    fn tanh_gradient(
        &self,
        parents: &[NodeId],
        output: &Tensor,
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (parent_id, _parent_value) = self.unary_parent_value(parents, "tanh")?;
        let derivative_data: Vec<f64> = output
            .data()
            .iter()
            .map(|&v| 1.0 - v * v)
            .collect();
        let derivative = Tensor::from_vec(output.shape().to_vec(), derivative_data)?;
        let grad = upstream_grad.mul(&derivative)?;
        Ok(vec![(parent_id, grad)])
    }

    fn softmax_gradient(
        &self,
        parents: &[NodeId],
        output: &Tensor,
        upstream_grad: &Tensor,
    ) -> Result<Vec<(NodeId, Tensor)>> {
        let (parent_id, _parent_value) = self.unary_parent_value(parents, "softmax")?;
        let grad_data = match output.rank() {
            1 => softmax_gradient_slice(output.data(), upstream_grad.data()),
            2 => softmax_gradient_matrix(output, upstream_grad)?,
            rank => {
                return Err(RustGradError::InvalidArgument {
                    name: "rank",
                    reason: format!("softmax gradient supports rank 1 or 2, got rank {rank}"),
                })
            }
        };
        let grad = Tensor::from_vec(output.shape().to_vec(), grad_data)?;
        Ok(vec![(parent_id, grad)])
    }

    fn binary_parent_values(
        &self,
        parents: &[NodeId],
        op: &'static str,
    ) -> Result<(NodeId, Tensor, NodeId, Tensor)> {
        let [left_id, right_id] = parents else {
            return Err(RustGradError::InvalidArgument {
                name: "parents",
                reason: format!("{op} expects exactly 2 parents, got {}", parents.len()),
            });
        };
        let left_value = self
            .node(*left_id)
            .ok_or_else(|| RustGradError::InvalidArgument {
                name: "parents",
                reason: format!("node {} does not exist", left_id.index()),
            })?
            .value()
            .clone();
        let right_value = self
            .node(*right_id)
            .ok_or_else(|| RustGradError::InvalidArgument {
                name: "parents",
                reason: format!("node {} does not exist", right_id.index()),
            })?
            .value()
            .clone();

        Ok((*left_id, left_value, *right_id, right_value))
    }

    fn unary_parent_value(&self, parents: &[NodeId], op: &'static str) -> Result<(NodeId, Tensor)> {
        let [parent_id] = parents else {
            return Err(RustGradError::InvalidArgument {
                name: "parents",
                reason: format!("{op} expects exactly 1 parent, got {}", parents.len()),
            });
        };
        let parent_value = self
            .node(*parent_id)
            .ok_or_else(|| RustGradError::InvalidArgument {
                name: "parents",
                reason: format!("node {} does not exist", parent_id.index()),
            })?
            .value()
            .clone();

        Ok((*parent_id, parent_value))
    }

    fn fit_gradient_to_parent(parent_value: &Tensor, grad: Tensor) -> Result<Tensor> {
        if parent_value.dims() == grad.dims() {
            return Ok(grad);
        }

        if parent_value.shape().is_scalar_like() {
            return grad.sum();
        }

        Err(RustGradError::ShapeMismatch {
            op: "gradient",
            left: parent_value.shape().to_vec(),
            right: grad.shape().to_vec(),
        })
    }

    fn broadcast_reduction_gradient(
        parent_value: &Tensor,
        upstream_grad: &Tensor,
        scale: f64,
    ) -> Result<Tensor> {
        if upstream_grad.shape().is_scalar_like() {
            let value = upstream_grad.get_flat(0)? * scale;
            return Tensor::full(parent_value.shape().to_vec(), value);
        }

        if upstream_grad.dims() == parent_value.dims() {
            return upstream_grad.mul(&Tensor::scalar(scale)?);
        }

        Err(RustGradError::ShapeMismatch {
            op: "reduction gradient",
            left: parent_value.shape().to_vec(),
            right: upstream_grad.shape().to_vec(),
        })
    }
}

/// Computes per-element softmax gradient for a single row.
///
/// Given softmax output `s` and upstream gradient `g`, returns
/// `grad[i] = s[i] * (g[i] - sum_j(s[j] * g[j]))`.
fn softmax_gradient_slice(softmax_output: &[f64], upstream: &[f64]) -> Vec<f64> {
    let dot: f64 = softmax_output
        .iter()
        .zip(upstream.iter())
        .map(|(&s, &g)| s * g)
        .sum();
    softmax_output
        .iter()
        .zip(upstream.iter())
        .map(|(&s, &g)| s * (g - dot))
        .collect()
}

/// Computes row-wise softmax gradient for a rank-2 tensor.
fn softmax_gradient_matrix(output: &Tensor, upstream_grad: &Tensor) -> Result<Vec<f64>> {
    let rows = output.rows().expect("rank 2 tensors always have rows");
    let cols = output.cols().expect("rank 2 tensors always have columns");
    let mut grad = Vec::with_capacity(output.len());
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        grad.extend(softmax_gradient_slice(
            &output.data()[start..end],
            &upstream_grad.data()[start..end],
        ));
    }
    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::{ComputationGraph, NodeId, Operation};
    use crate::tensor::Tensor;
    use crate::RustGradError;

    #[test]
    fn creates_sequential_node_ids() {
        let mut graph = ComputationGraph::new();
        let first = graph.add_leaf(Tensor::scalar(1.0).expect("valid scalar"), true);
        let second = graph.add_leaf(Tensor::scalar(2.0).expect("valid scalar"), false);

        assert_eq!(first.index(), 0);
        assert_eq!(second.index(), 1);
        assert_eq!(graph.len(), 2);
        assert!(!graph.is_empty());
    }

    #[test]
    fn records_operation_node_metadata() {
        let mut graph = ComputationGraph::new();
        let left = graph.add_leaf(Tensor::scalar(1.0).expect("valid scalar"), true);
        let right = graph.add_leaf(Tensor::scalar(2.0).expect("valid scalar"), true);
        let value = Tensor::scalar(3.0).expect("valid scalar");

        let output = graph
            .add_operation(Operation::Add, vec![left, right], value, true)
            .expect("parents should exist");
        let node = graph.node(output).expect("output node should exist");

        assert_eq!(node.id(), output);
        assert_eq!(node.parents(), &[left, right]);
        assert_eq!(node.operation_kind(), &Operation::Add);
        assert!(node.requires_grad());
        assert_eq!(node.value().data(), &[3.0]);
    }

    #[test]
    fn rejects_operation_with_missing_parent() {
        let mut graph = ComputationGraph::new();
        let value = Tensor::scalar(1.0).expect("valid scalar");

        assert_eq!(
            graph
                .add_operation(Operation::Mul, vec![NodeId::new(9)], value, true)
                .expect_err("parent should not exist"),
            RustGradError::InvalidArgument {
                name: "parents",
                reason: "node 9 does not exist".to_string(),
            }
        );
    }

    #[test]
    fn stores_and_clears_gradients() {
        let mut graph = ComputationGraph::new();
        let node_id = graph.add_leaf(Tensor::scalar(2.0).expect("valid scalar"), true);
        let grad = Tensor::scalar(1.0).expect("valid scalar");

        graph
            .node_mut(node_id)
            .expect("node should exist")
            .set_grad(grad);
        assert_eq!(
            graph
                .node(node_id)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[1.0][..])
        );

        graph.clear_gradients();

        assert!(graph
            .node(node_id)
            .expect("node should exist")
            .grad()
            .is_none());
    }

    #[test]
    fn operation_name_supports_builtin_and_custom_ops() {
        assert_eq!(Operation::MatMul.name(), "matmul");
        assert_eq!(Operation::Custom("dropout".to_string()).name(), "dropout");
    }

    #[test]
    fn returns_dependency_first_topological_order() {
        let mut graph = ComputationGraph::new();
        let left = graph.add_leaf(Tensor::scalar(1.0).expect("valid scalar"), true);
        let right = graph.add_leaf(Tensor::scalar(2.0).expect("valid scalar"), true);
        let hidden = graph
            .add_operation(
                Operation::Custom("identity".to_string()),
                vec![left],
                Tensor::scalar(1.0).expect("valid scalar"),
                true,
            )
            .expect("parent should exist");
        let output = graph
            .add_operation(
                Operation::Custom("join".to_string()),
                vec![hidden, right],
                Tensor::scalar(3.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");

        let order = graph
            .topological_order(output)
            .expect("topological order should exist");

        assert_eq!(order, vec![left, hidden, right, output]);
    }

    #[test]
    fn topological_order_visits_shared_parent_once() {
        let mut graph = ComputationGraph::new();
        let shared = graph.add_leaf(Tensor::scalar(2.0).expect("valid scalar"), true);
        let output = graph
            .add_operation(
                Operation::Custom("double-use".to_string()),
                vec![shared, shared],
                Tensor::scalar(4.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");

        let order = graph
            .topological_order(output)
            .expect("topological order should exist");

        assert_eq!(order, vec![shared, output]);
    }

    #[test]
    fn rejects_topological_order_for_missing_output() {
        let graph = ComputationGraph::new();

        assert_eq!(
            graph
                .topological_order(NodeId::new(1))
                .expect_err("output should be missing"),
            RustGradError::InvalidArgument {
                name: "node",
                reason: "node 1 does not exist".to_string(),
            }
        );
    }

    #[test]
    fn backward_seeds_leaf_output_gradient_with_ones() {
        let mut graph = ComputationGraph::new();
        let output = graph.add_leaf(
            Tensor::matrix(1, 3, vec![2.0, 4.0, 6.0]).expect("valid matrix"),
            true,
        );

        graph
            .backward(output)
            .expect("leaf backward should succeed");

        assert_eq!(
            graph
                .node(output)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[1.0, 1.0, 1.0][..])
        );
    }

    #[test]
    fn backward_clears_existing_gradients_before_seeding_output() {
        let mut graph = ComputationGraph::new();
        let output = graph.add_leaf(Tensor::scalar(3.0).expect("valid scalar"), true);
        graph
            .node_mut(output)
            .expect("node should exist")
            .set_grad(Tensor::scalar(99.0).expect("valid scalar"));

        graph
            .backward(output)
            .expect("leaf backward should succeed");

        assert_eq!(
            graph
                .node(output)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[1.0][..])
        );
    }

    #[test]
    fn backward_reports_unsupported_operation_for_missing_gradient_rule() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::matrix(2, 1, vec![1.0, 2.0]).expect("valid matrix"),
            true,
        );
        let output = graph
            .add_operation(
                Operation::Custom("unsupported-op".to_string()),
                vec![input],
                Tensor::matrix(2, 1, vec![1.0, 2.0]).expect("valid matrix"),
                true,
            )
            .expect("parent should exist");

        assert_eq!(
            graph
                .backward(output)
                .expect_err("custom unsupported operation should fail"),
            RustGradError::UnsupportedOperation {
                op: "unsupported-op".to_string(),
                reason: "gradient rule is not implemented yet for upstream shape [2, 1]"
                    .to_string(),
            }
        );
    }

    #[test]
    fn backward_computes_add_gradients_for_scalars() {
        let mut graph = ComputationGraph::new();
        let left = graph.add_leaf(Tensor::scalar(2.0).expect("valid scalar"), true);
        let right = graph.add_leaf(Tensor::scalar(3.0).expect("valid scalar"), true);
        let output = graph
            .add_operation(
                Operation::Add,
                vec![left, right],
                Tensor::scalar(5.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");

        graph.backward(output).expect("add backward should succeed");

        assert_eq!(
            graph
                .node(left)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[1.0][..])
        );
        assert_eq!(
            graph
                .node(right)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[1.0][..])
        );
    }

    #[test]
    fn backward_computes_sub_gradients_for_scalars() {
        let mut graph = ComputationGraph::new();
        let left = graph.add_leaf(Tensor::scalar(7.0).expect("valid scalar"), true);
        let right = graph.add_leaf(Tensor::scalar(4.0).expect("valid scalar"), true);
        let output = graph
            .add_operation(
                Operation::Sub,
                vec![left, right],
                Tensor::scalar(3.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");

        graph.backward(output).expect("sub backward should succeed");

        assert_eq!(
            graph
                .node(left)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[1.0][..])
        );
        assert_eq!(
            graph
                .node(right)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[-1.0][..])
        );
    }

    #[test]
    fn backward_computes_mul_gradients_for_scalars() {
        let mut graph = ComputationGraph::new();
        let left = graph.add_leaf(Tensor::scalar(3.0).expect("valid scalar"), true);
        let right = graph.add_leaf(Tensor::scalar(5.0).expect("valid scalar"), true);
        let output = graph
            .add_operation(
                Operation::Mul,
                vec![left, right],
                Tensor::scalar(15.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");

        graph.backward(output).expect("mul backward should succeed");

        assert_eq!(
            graph
                .node(left)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[5.0][..])
        );
        assert_eq!(
            graph
                .node(right)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[3.0][..])
        );
    }

    #[test]
    fn backward_computes_div_gradients_for_scalars() {
        let mut graph = ComputationGraph::new();
        let left = graph.add_leaf(Tensor::scalar(8.0).expect("valid scalar"), true);
        let right = graph.add_leaf(Tensor::scalar(2.0).expect("valid scalar"), true);
        let output = graph
            .add_operation(
                Operation::Div,
                vec![left, right],
                Tensor::scalar(4.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");

        graph.backward(output).expect("div backward should succeed");

        assert_eq!(
            graph
                .node(left)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[0.5][..])
        );
        assert_eq!(
            graph
                .node(right)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[-2.0][..])
        );
    }

    #[test]
    fn backward_computes_elementwise_vector_multiply_gradients() {
        let mut graph = ComputationGraph::new();
        let left = graph.add_leaf(Tensor::vector(vec![2.0, 3.0]).expect("valid vector"), true);
        let right = graph.add_leaf(Tensor::vector(vec![5.0, 7.0]).expect("valid vector"), true);
        let output = graph
            .add_operation(
                Operation::Mul,
                vec![left, right],
                Tensor::vector(vec![10.0, 21.0]).expect("valid vector"),
                true,
            )
            .expect("parents should exist");

        graph.backward(output).expect("mul backward should succeed");

        assert_eq!(
            graph
                .node(left)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[5.0, 7.0][..])
        );
        assert_eq!(
            graph
                .node(right)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[2.0, 3.0][..])
        );
    }

    #[test]
    fn backward_reduces_broadcast_gradient_back_to_scalar_parent() {
        let mut graph = ComputationGraph::new();
        let scalar = graph.add_leaf(Tensor::scalar(2.0).expect("valid scalar"), true);
        let vector = graph.add_leaf(
            Tensor::vector(vec![3.0, 4.0, 5.0]).expect("valid vector"),
            true,
        );
        let output = graph
            .add_operation(
                Operation::Mul,
                vec![scalar, vector],
                Tensor::vector(vec![6.0, 8.0, 10.0]).expect("valid vector"),
                true,
            )
            .expect("parents should exist");

        graph.backward(output).expect("mul backward should succeed");

        assert_eq!(
            graph
                .node(scalar)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[12.0][..])
        );
        assert_eq!(
            graph
                .node(vector)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[2.0, 2.0, 2.0][..])
        );
    }

    #[test]
    fn backward_accumulates_gradients_for_repeated_parent() {
        let mut graph = ComputationGraph::new();
        let value = graph.add_leaf(Tensor::scalar(2.0).expect("valid scalar"), true);
        let output = graph
            .add_operation(
                Operation::Add,
                vec![value, value],
                Tensor::scalar(4.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");

        graph.backward(output).expect("add backward should succeed");

        assert_eq!(
            graph
                .node(value)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[2.0][..])
        );
    }

    #[test]
    fn backward_matches_hand_calculated_gradient_for_z_equals_x_times_y_plus_x() {
        let mut graph = ComputationGraph::new();
        let x = graph.add_leaf(Tensor::scalar(3.0).expect("valid scalar"), true);
        let y = graph.add_leaf(Tensor::scalar(4.0).expect("valid scalar"), true);
        let product = graph
            .add_operation(
                Operation::Mul,
                vec![x, y],
                Tensor::scalar(12.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");
        let z = graph
            .add_operation(
                Operation::Add,
                vec![product, x],
                Tensor::scalar(15.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");

        graph.backward(z).expect("chained backward should succeed");

        assert_eq!(
            graph.node(x).and_then(|node| node.grad()).map(Tensor::data),
            Some(&[5.0][..])
        );
        assert_eq!(
            graph.node(y).and_then(|node| node.grad()).map(Tensor::data),
            Some(&[3.0][..])
        );
    }

    #[test]
    fn backward_skips_leaf_that_does_not_require_grad() {
        let mut graph = ComputationGraph::new();
        let trainable = graph.add_leaf(Tensor::scalar(2.0).expect("valid scalar"), true);
        let constant = graph.add_leaf(Tensor::scalar(3.0).expect("valid scalar"), false);
        let output = graph
            .add_operation(
                Operation::Mul,
                vec![trainable, constant],
                Tensor::scalar(6.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");

        graph.backward(output).expect("mul backward should succeed");

        assert_eq!(
            graph
                .node(trainable)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[3.0][..])
        );
        assert!(graph
            .node(constant)
            .expect("constant should exist")
            .grad()
            .is_none());
    }

    #[test]
    fn backward_computes_vector_add_and_sub_gradients() {
        let mut graph = ComputationGraph::new();
        let left = graph.add_leaf(Tensor::vector(vec![1.0, 2.0]).expect("valid vector"), true);
        let right = graph.add_leaf(Tensor::vector(vec![3.0, 5.0]).expect("valid vector"), true);
        let sum = graph
            .add_operation(
                Operation::Add,
                vec![left, right],
                Tensor::vector(vec![4.0, 7.0]).expect("valid vector"),
                true,
            )
            .expect("parents should exist");
        let output = graph
            .add_operation(
                Operation::Sub,
                vec![sum, right],
                Tensor::vector(vec![1.0, 2.0]).expect("valid vector"),
                true,
            )
            .expect("parents should exist");

        graph
            .backward(output)
            .expect("chained backward should succeed");

        assert_eq!(
            graph
                .node(left)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[1.0, 1.0][..])
        );
        assert_eq!(
            graph
                .node(right)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[0.0, 0.0][..])
        );
    }

    #[test]
    fn backward_computes_elementwise_vector_division_gradients() {
        let mut graph = ComputationGraph::new();
        let numerator = graph.add_leaf(Tensor::vector(vec![8.0, 9.0]).expect("valid vector"), true);
        let denominator =
            graph.add_leaf(Tensor::vector(vec![2.0, 3.0]).expect("valid vector"), true);
        let output = graph
            .add_operation(
                Operation::Div,
                vec![numerator, denominator],
                Tensor::vector(vec![4.0, 3.0]).expect("valid vector"),
                true,
            )
            .expect("parents should exist");

        graph.backward(output).expect("div backward should succeed");

        assert_eq!(
            graph
                .node(numerator)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[0.5, 1.0 / 3.0][..])
        );
        assert_eq!(
            graph
                .node(denominator)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[-2.0, -1.0][..])
        );
    }

    #[test]
    fn backward_accumulates_gradients_from_multiple_paths() {
        let mut graph = ComputationGraph::new();
        let x = graph.add_leaf(Tensor::scalar(2.0).expect("valid scalar"), true);
        let y = graph.add_leaf(Tensor::scalar(5.0).expect("valid scalar"), true);
        let product = graph
            .add_operation(
                Operation::Mul,
                vec![x, y],
                Tensor::scalar(10.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");
        let output = graph
            .add_operation(
                Operation::Add,
                vec![product, y],
                Tensor::scalar(15.0).expect("valid scalar"),
                true,
            )
            .expect("parents should exist");

        graph
            .backward(output)
            .expect("chained backward should succeed");

        assert_eq!(
            graph.node(x).and_then(|node| node.grad()).map(Tensor::data),
            Some(&[5.0][..])
        );
        assert_eq!(
            graph.node(y).and_then(|node| node.grad()).map(Tensor::data),
            Some(&[3.0][..])
        );
    }

    #[test]
    fn backward_computes_matmul_gradients() {
        let mut graph = ComputationGraph::new();
        let left = graph.add_leaf(
            Tensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid left matrix"),
            true,
        );
        let right = graph.add_leaf(
            Tensor::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0]).expect("valid right matrix"),
            true,
        );
        let output = graph
            .add_operation(
                Operation::MatMul,
                vec![left, right],
                Tensor::matrix(2, 2, vec![19.0, 22.0, 43.0, 50.0]).expect("valid output"),
                true,
            )
            .expect("parents should exist");

        graph
            .backward(output)
            .expect("matmul backward should succeed");

        assert_eq!(
            graph
                .node(left)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[11.0, 15.0, 11.0, 15.0][..])
        );
        assert_eq!(
            graph
                .node(right)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[4.0, 4.0, 6.0, 6.0][..])
        );
    }

    #[test]
    fn backward_computes_sum_gradient() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid matrix"),
            true,
        );
        let output = graph
            .add_operation(
                Operation::Sum,
                vec![input],
                Tensor::scalar(10.0).expect("valid scalar"),
                true,
            )
            .expect("parent should exist");

        graph.backward(output).expect("sum backward should succeed");

        assert_eq!(
            graph
                .node(input)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[1.0, 1.0, 1.0, 1.0][..])
        );
    }

    #[test]
    fn backward_computes_mean_gradient() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::matrix(2, 2, vec![2.0, 4.0, 6.0, 8.0]).expect("valid matrix"),
            true,
        );
        let output = graph
            .add_operation(
                Operation::Mean,
                vec![input],
                Tensor::scalar(5.0).expect("valid scalar"),
                true,
            )
            .expect("parent should exist");

        graph
            .backward(output)
            .expect("mean backward should succeed");

        assert_eq!(
            graph
                .node(input)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[0.25, 0.25, 0.25, 0.25][..])
        );
    }

    #[test]
    fn backward_chains_matmul_into_sum() {
        let mut graph = ComputationGraph::new();
        let left = graph.add_leaf(
            Tensor::matrix(1, 2, vec![2.0, 3.0]).expect("valid left matrix"),
            true,
        );
        let right = graph.add_leaf(
            Tensor::matrix(2, 1, vec![5.0, 7.0]).expect("valid right matrix"),
            true,
        );
        let product = graph
            .add_operation(
                Operation::MatMul,
                vec![left, right],
                Tensor::matrix(1, 1, vec![31.0]).expect("valid product"),
                true,
            )
            .expect("parents should exist");
        let output = graph
            .add_operation(
                Operation::Sum,
                vec![product],
                Tensor::scalar(31.0).expect("valid scalar"),
                true,
            )
            .expect("parent should exist");

        graph
            .backward(output)
            .expect("chain backward should succeed");

        assert_eq!(
            graph
                .node(left)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[5.0, 7.0][..])
        );
        assert_eq!(
            graph
                .node(right)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[2.0, 3.0][..])
        );
    }

    #[test]
    fn backward_skips_matmul_parent_that_does_not_require_grad() {
        let mut graph = ComputationGraph::new();
        let trainable = graph.add_leaf(
            Tensor::matrix(1, 2, vec![2.0, 3.0]).expect("valid matrix"),
            true,
        );
        let constant = graph.add_leaf(
            Tensor::matrix(2, 1, vec![5.0, 7.0]).expect("valid matrix"),
            false,
        );
        let output = graph
            .add_operation(
                Operation::MatMul,
                vec![trainable, constant],
                Tensor::matrix(1, 1, vec![31.0]).expect("valid output"),
                true,
            )
            .expect("parents should exist");

        graph
            .backward(output)
            .expect("matmul backward should succeed");

        assert_eq!(
            graph
                .node(trainable)
                .and_then(|node| node.grad())
                .map(Tensor::data),
            Some(&[5.0, 7.0][..])
        );
        assert!(graph
            .node(constant)
            .expect("constant should exist")
            .grad()
            .is_none());
    }

    // ── Gradient rule tests for 4.1–4.5 ──────────────────────────────

    const FD_EPSILON: f64 = 1e-6;
    const TOLERANCE: f64 = 1e-5;

    /// Finite-difference check: perturb each leaf value and compare
    /// `(f(x+h) - f(x-h)) / (2h)` to the accumulated autograd gradient.
    fn check_gradient_numeric(graph: &mut ComputationGraph, output: NodeId) {
        // Capture leaf nodes *before* backward so we know which ids to check.
        let leaf_ids: Vec<NodeId> = (0..graph.len())
            .filter(|&i| {
                graph
                    .node(NodeId::new(i))
                    .is_some_and(|n| n.operation_kind() == &Operation::Leaf && n.requires_grad())
            })
            .map(NodeId::new)
            .collect();

        assert!(!leaf_ids.is_empty(), "expected at least one leaf");

        let base_output_value: f64 = graph
            .node(output)
            .expect("output exists")
            .value()
            .sum()
            .expect("sum")
            .get_flat(0)
            .expect("scalar");

        graph.backward(output).expect("backward should succeed");

        for &leaf_id in &leaf_ids {
            let autograd_grad: Vec<f64> = graph
                .node(leaf_id)
                .and_then(|n| n.grad())
                .map(|t| t.data().to_vec())
                .expect("leaf should have gradient");

            let leaf_value: Vec<f64> = graph
                .node(leaf_id)
                .expect("leaf exists")
                .value()
                .data()
                .to_vec();

            for i in 0..leaf_value.len() {
                let orig = leaf_value[i];
                // f(x + h) – rebuild graph with perturbed value
                let hi = recompute_with_perturbation(graph, leaf_id, i, orig, FD_EPSILON, output);
                let lo = recompute_with_perturbation(graph, leaf_id, i, orig, -FD_EPSILON, output);
                let numeric_grad = (hi - lo) / (2.0 * FD_EPSILON);
                assert!(
                    (autograd_grad[i] - numeric_grad).abs() < TOLERANCE,
                    "leaf {} idx {}: autograd={}, numeric={}",
                    leaf_id.index(),
                    i,
                    autograd_grad[i],
                    numeric_grad
                );
            }
        }

        // Restore the graph to its original state for any subsequent checks.
        *graph = ComputationGraph::new();
        let _ = base_output_value; // suppress unused warning
    }

    /// Rebuild the graph from scratch with one leaf value perturbed, return scalar output.
    fn recompute_with_perturbation(
        original_graph: &ComputationGraph,
        perturb_leaf: NodeId,
        index: usize,
        orig_value: f64,
        epsilon: f64,
        output_id: NodeId,
    ) -> f64 {
        let mut new_graph = ComputationGraph::new();
        let mut id_map: Vec<Option<NodeId>> = vec![None; original_graph.len()];

        for i in 0..original_graph.len() {
            let node_id = NodeId::new(i);
            let node = original_graph.node(node_id).expect("node exists");

            match node.operation_kind() {
                Operation::Leaf => {
                    let mut values = node.value().data().to_vec();
                    if node_id == perturb_leaf {
                        values[index] = orig_value + epsilon;
                    }
                    let tensor =
                        Tensor::from_vec(node.value().shape().to_vec(), values).expect("valid tensor");
                    let new_id = new_graph.add_leaf(tensor, node.requires_grad());
                    id_map[i] = Some(new_id);
                }
                _ => {
                    let parents: Vec<NodeId> = node
                        .parents()
                        .iter()
                        .map(|p| id_map[p.index()].expect("parent should be mapped"))
                        .collect();
                    // Recompute the actual value from parents instead of copying.
                    let new_value = recompute_op(
                        &new_graph,
                        node.operation_kind(),
                        &parents,
                    )
                    .expect("recompute op");
                    let new_id = new_graph
                        .add_operation(
                            node.operation_kind().clone(),
                            parents,
                            new_value,
                            node.requires_grad(),
                        )
                        .expect("operation should build");
                    id_map[i] = Some(new_id);
                }
            }
        }

        let remapped_output = id_map[output_id.index()].expect("output should be mapped");
        new_graph
            .node(remapped_output)
            .expect("output exists")
            .value()
            .sum()
            .expect("sum")
            .get_flat(0)
            .expect("scalar")
    }

    /// Recompute the value of an operation from its parents' values in the graph.
    fn recompute_op(
        graph: &ComputationGraph,
        op: &Operation,
        parents: &[NodeId],
    ) -> crate::Result<Tensor> {
        let get_val = |i: usize| -> crate::Result<Tensor> {
            Ok(graph.node(parents[i]).expect("parent").value().clone())
        };
        match op {
            Operation::Add => get_val(0)?.add(&get_val(1)?),
            Operation::Sub => get_val(0)?.sub(&get_val(1)?),
            Operation::Mul => get_val(0)?.mul(&get_val(1)?),
            Operation::Div => get_val(0)?.div(&get_val(1)?),
            Operation::MatMul => get_val(0)?.matmul(&get_val(1)?),
            Operation::Transpose => get_val(0)?.transpose(),
            Operation::Sum => get_val(0)?.sum(),
            Operation::Mean => get_val(0)?.mean(),
            Operation::Relu => crate::nn::relu(&get_val(0)?),
            Operation::Sigmoid => crate::nn::sigmoid(&get_val(0)?),
            Operation::Tanh => crate::nn::tanh(&get_val(0)?),
            Operation::Softmax => crate::nn::softmax(&get_val(0)?),
            _ => Err(RustGradError::UnsupportedOperation {
                op: op.name().to_string(),
                reason: "finite-difference recomputation not supported".to_string(),
            }),
        }
    }

    #[test]
    fn backward_computes_transpose_gradient() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid matrix"),
            true,
        );
        let input_t = graph.node(input).expect("leaf").value().transpose().expect("transpose");
        let output = graph
            .add_operation(
                Operation::Transpose,
                vec![input],
                input_t,
                true,
            )
            .expect("parent should exist");

        // Transpose gradient: upstream (ones) transposed back.
        // For 2x3 -> 3x2 transpose, ones upstream gives all-ones grad.
        graph.backward(output).expect("transpose backward should succeed");
        assert_eq!(
            graph
                .node(input)
                .and_then(|n| n.grad())
                .map(|t| t.data()),
            Some(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0][..])
        );
    }

    #[test]
    fn backward_computes_relu_gradient_for_vector() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::vector(vec![-1.0, 0.0, 2.0, -3.0, 4.0]).expect("valid vector"),
            true,
        );
        let relu_val = crate::nn::relu(graph.node(input).expect("leaf").value()).expect("relu");
        let output = graph
            .add_operation(
                Operation::Relu,
                vec![input],
                relu_val,
                true,
            )
            .expect("parent should exist");

        graph.backward(output).expect("relu backward should succeed");
        // grad = upstream(1) * relu'(input) => 0 for <=0, 1 for >0
        assert_eq!(
            graph
                .node(input)
                .and_then(|n| n.grad())
                .map(|t| t.data()),
            Some(&[0.0, 0.0, 1.0, 0.0, 1.0][..])
        );
    }

    #[test]
    fn backward_computes_sigmoid_gradient_for_vector() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::vector(vec![0.0, 1.0, -1.0]).expect("valid vector"),
            true,
        );
        let input_val = graph.node(input).expect("leaf").value().clone();
        let output_val = crate::nn::sigmoid(&input_val).expect("sigmoid");
        let output = graph
            .add_operation(Operation::Sigmoid, vec![input], output_val, true)
            .expect("parent should exist");

        graph.backward(output).expect("sigmoid backward should succeed");

        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        let s: Vec<f64> = input_val
            .data()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        let expected: Vec<f64> = s.iter().map(|&v| v * (1.0 - v)).collect();

        let grad = graph
            .node(input)
            .and_then(|n| n.grad())
            .map(|t| t.data().to_vec())
            .expect("grad exists");
        for (g, e) in grad.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-10, "sigmoid grad: {g} vs {e}");
        }
    }

    #[test]
    fn backward_computes_tanh_gradient_for_vector() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::vector(vec![0.0, 0.5, -0.5]).expect("valid vector"),
            true,
        );
        let input_val = graph.node(input).expect("leaf").value().clone();
        let output_val = crate::nn::tanh(&input_val).expect("tanh");
        let output = graph
            .add_operation(Operation::Tanh, vec![input], output_val, true)
            .expect("parent should exist");

        graph.backward(output).expect("tanh backward should succeed");

        // tanh'(x) = 1 - tanh(x)^2
        let t: Vec<f64> = input_val
            .data()
            .iter()
            .map(|&x| x.tanh())
            .collect();
        let expected: Vec<f64> = t.iter().map(|&v| 1.0 - v * v).collect();

        let grad = graph
            .node(input)
            .and_then(|n| n.grad())
            .map(|t| t.data().to_vec())
            .expect("grad exists");
        for (g, e) in grad.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-10, "tanh grad: {g} vs {e}");
        }
    }

    #[test]
    fn backward_computes_softmax_gradient_for_vector() {
        // s = softmax([1, 2, 3]), upstream = ones
        // grad[i] = s[i] * (1 - dot) where dot = sum_j s[j]*1 = 1, so all zero.
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::vector(vec![1.0, 2.0, 3.0]).expect("valid vector"),
            true,
        );
        let input_val = graph.node(input).expect("leaf").value().clone();
        let output_val = crate::nn::softmax(&input_val).expect("softmax");
        let s: Vec<f64> = output_val.data().to_vec();
        let output = graph
            .add_operation(Operation::Softmax, vec![input], output_val, true)
            .expect("parent should exist");

        graph.backward(output).expect("softmax backward should succeed");

        let grad = graph
            .node(input)
            .and_then(|n| n.grad())
            .map(|t| t.data().to_vec())
            .expect("grad exists");

        // With upstream=ones: grad[i] = s[i] * (1 - sum(s)) = s[i] * 0 = 0
        let grad_sum: f64 = grad.iter().sum();
        assert!(
            grad_sum.abs() < 1e-10,
            "softmax gradient should sum to zero, got {grad_sum}"
        );
        for &g in &grad {
            assert!(g.abs() < 1e-10, "each softmax gradient should be ~0, got {g}");
        }

        // Also verify: dot = sum_j(s[j] * upstream[j]) = sum(s) = 1.0
        let dot: f64 = s.iter().sum();
        assert!((dot - 1.0).abs() < 1e-10);
    }

    #[test]
    fn backward_computes_softmax_gradient_for_matrix() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).expect("valid matrix"),
            true,
        );
        let input_val = graph.node(input).expect("leaf").value().clone();
        let output_val = crate::nn::softmax(&input_val).expect("softmax");
        let output = graph
            .add_operation(Operation::Softmax, vec![input], output_val, true)
            .expect("parent should exist");

        graph.backward(output).expect("softmax backward should succeed");

        let grad = graph
            .node(input)
            .and_then(|n| n.grad())
            .map(|t| t.data().to_vec())
            .expect("grad exists");

        // Each row's gradient should sum to zero.
        // Row 0: s = softmax([1,2,3]), grad sum = 0
        let row0_sum: f64 = grad[0..3].iter().sum();
        assert!(
            row0_sum.abs() < 1e-10,
            "row 0 grad sum should be 0, got {row0_sum}"
        );
        // Row 1: s = softmax([1,1,1]) = [1/3,1/3,1/3]
        // With upstream=ones: grad[i] = 1/3 * (1 - 1) = 0
        let row1_sum: f64 = grad[3..6].iter().sum();
        assert!(
            row1_sum.abs() < 1e-10,
            "row 1 grad sum should be 0, got {row1_sum}"
        );
    }

    #[test]
    fn softmax_gradient_passes_finite_difference_check() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::vector(vec![1.0, 2.0, 3.0]).expect("valid vector"),
            true,
        );
        let sm_val = crate::nn::softmax(graph.node(input).expect("leaf").value()).expect("softmax");
        let sm = graph
            .add_operation(Operation::Softmax, vec![input], sm_val, true)
            .expect("softmax ok");
        let output = graph
            .add_operation(
                Operation::Sum,
                vec![sm],
                graph.node(sm).expect("sm exists").value().clone(),
                true,
            )
            .expect("sum ok");

        check_gradient_numeric(&mut graph, output);
    }

    #[test]
    fn transpose_gradient_passes_finite_difference_check() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::matrix(2, 2, vec![2.0, 3.0, 5.0, 7.0]).expect("valid matrix"),
            true,
        );
        let input_t = graph.node(input).expect("leaf").value().transpose().expect("transpose");
        let tr = graph
            .add_operation(Operation::Transpose, vec![input], input_t, true)
            .expect("transpose ok");
        let output = graph
            .add_operation(
                Operation::Sum,
                vec![tr],
                graph.node(tr).expect("tr exists").value().clone(),
                true,
            )
            .expect("sum ok");

        check_gradient_numeric(&mut graph, output);
    }

    #[test]
    fn relu_gradient_passes_finite_difference_check() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::vector(vec![1.5, -0.5]).expect("valid vector"),
            true,
        );
        let relu_val = crate::nn::relu(graph.node(input).expect("leaf").value()).expect("relu");
        let r = graph
            .add_operation(Operation::Relu, vec![input], relu_val, true)
            .expect("relu ok");
        let output = graph
            .add_operation(
                Operation::Sum,
                vec![r],
                graph.node(r).expect("r exists").value().clone(),
                true,
            )
            .expect("sum ok");

        check_gradient_numeric(&mut graph, output);
    }

    #[test]
    fn sigmoid_gradient_passes_finite_difference_check() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::vector(vec![0.3, -0.7]).expect("valid vector"),
            true,
        );
        let sig_val = crate::nn::sigmoid(graph.node(input).expect("leaf").value()).expect("sigmoid");
        let s = graph
            .add_operation(Operation::Sigmoid, vec![input], sig_val, true)
            .expect("sigmoid ok");
        let output = graph
            .add_operation(
                Operation::Sum,
                vec![s],
                graph.node(s).expect("s exists").value().clone(),
                true,
            )
            .expect("sum ok");

        check_gradient_numeric(&mut graph, output);
    }

    #[test]
    fn tanh_gradient_passes_finite_difference_check() {
        let mut graph = ComputationGraph::new();
        let input = graph.add_leaf(
            Tensor::vector(vec![0.8, -0.3]).expect("valid vector"),
            true,
        );
        let tanh_val = crate::nn::tanh(graph.node(input).expect("leaf").value()).expect("tanh");
        let t = graph
            .add_operation(Operation::Tanh, vec![input], tanh_val, true)
            .expect("tanh ok");
        let output = graph
            .add_operation(
                Operation::Sum,
                vec![t],
                graph.node(t).expect("t exists").value().clone(),
                true,
            )
            .expect("sum ok");

        check_gradient_numeric(&mut graph, output);
    }
}
