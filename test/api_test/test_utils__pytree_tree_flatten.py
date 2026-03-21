# Test purpose: validate torch.utils._pytree.tree_flatten behavior on NPU-backed inputs,
# covering builtin container flattening, custom pytree registration, leaf overrides,
# and predictable error propagation.
# API name: torch.utils._pytree.tree_flatten
#
# Covered parameter dimensions:
#
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | tree input types | yes | tensor, list, tuple, dict, None, scalar, custom pytree node |
# | NPU tensor input | yes | tensors on npu:0 are preserved as leaves |
# | empty / boundary containers | yes | empty list, empty tuple, nested zero-arity tuple |
# | is_leaf None / non-None | yes | default traversal and explicit leaf override |
# | normal input | yes | nested builtin structures and custom registered class |
# | abnormal input | yes | `is_leaf` callback raising an exception |
# | output TreeSpec fields | yes | `type`, `context`, `num_leaves`, and `children_specs` |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | numeric value validation | tree_flatten is structural; this test intentionally does not inspect tensor contents. |
# | mutation / in-place effects | tree_flatten is read-only and does not mutate inputs. |
# | alternate device backends | this file is constrained to NPU execution and uses NPU tensors as the backend-facing input. |

import pytest
import torch
import torch_npu  # noqa: F401

from torch.utils import _pytree


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="torch_npu/NPU is not available in this environment.",
)


class Box:
    def __init__(self, x):
        self.x = x


_pytree.register_pytree_node(
    Box,
    lambda b: ((b.x,), {"kind": "box"}),
    lambda children, context: Box(children[0]),
)


def test_tree_flatten_builtin_containers_and_npu_tensor_leaves():
    tree = {
        "a": [torch.tensor([1], device="npu:0"), ()],
        "b": None,
        "c": (torch.tensor(2, device="npu:0"),),
    }

    flat, spec = _pytree.tree_flatten(tree)

    assert len(flat) == 3
    assert isinstance(flat[0], torch.Tensor)
    assert flat[0].device.type == "npu"
    assert flat[1] is None
    assert isinstance(flat[2], torch.Tensor)
    assert flat[2].device.type == "npu"
    assert spec.type is dict
    assert spec.context == ["a", "b", "c"]
    assert spec.num_leaves == 3
    assert spec.num_nodes == 7
    assert len(spec.children_specs) == 3


@pytest.mark.parametrize(
    "tree, expected_num_leaves",
    [
        ([], 0),
        ((), 0),
        ([(), []], 0),
        ([torch.tensor([], device="npu:0")], 1),
        (torch.tensor([], device="npu:0"), 1),
    ],
)
def test_tree_flatten_boundary_values(tree, expected_num_leaves):
    flat, spec = _pytree.tree_flatten(tree)

    assert len(flat) == expected_num_leaves
    assert spec.num_leaves == expected_num_leaves


def test_tree_flatten_custom_registered_node():
    tree = Box(torch.tensor([1, 2], device="npu:0"))

    flat, spec = _pytree.tree_flatten(tree)

    assert len(flat) == 1
    assert flat[0].device.type == "npu"
    assert flat[0].shape == torch.Size([2])
    assert spec.type is Box
    assert spec.context == {"kind": "box"}
    assert spec.num_leaves == 1
    assert spec.children_specs == [spec.children_specs[0]]


def test_tree_flatten_is_leaf_override_treats_nested_container_as_leaf():
    tree = [torch.tensor([1], device="npu:0"), (2, 3)]

    flat, spec = _pytree.tree_flatten(tree, is_leaf=lambda x: isinstance(x, tuple))

    assert len(flat) == 2
    assert flat[0].device.type == "npu"
    assert flat[1] == (2, 3)
    assert spec.num_leaves == 2
    assert all(child.num_leaves == 1 for child in spec.children_specs)


def test_tree_flatten_is_leaf_callback_error_propagates():
    tree = [torch.tensor([1], device="npu:0")]

    def raise_in_is_leaf(obj):
        raise RuntimeError(f"leaf check failed for {type(obj).__name__}")

    with pytest.raises(RuntimeError, match="leaf check failed"):
        _pytree.tree_flatten(tree, is_leaf=raise_in_is_leaf)
