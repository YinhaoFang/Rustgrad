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
                Operation::Transpose,
                vec![input],
                Tensor::matrix(1, 2, vec![1.0, 2.0]).expect("valid matrix"),
                true,
            )
            .expect("parent should exist");

        assert_eq!(
            graph
                .backward(output)
                .expect_err("transpose gradient rule is not implemented yet"),
            RustGradError::UnsupportedOperation {
                op: "transpose".to_string(),
                reason: "gradient rule is not implemented yet for upstream shape [1, 2]"
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
}
