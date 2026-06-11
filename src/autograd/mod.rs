//! Automatic differentiation and computation graph utilities.

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

    fn next_id(&self) -> NodeId {
        NodeId::new(self.nodes.len())
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
}
