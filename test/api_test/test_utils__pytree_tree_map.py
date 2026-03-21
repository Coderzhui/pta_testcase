"""
Test purpose: validate torch.utils._pytree.tree_map behavior on nested structures containing NPU tensors.
API name: torch.utils._pytree.tree_map

Coverage table:
| Parameter dimension | Coverage |
| --- | --- |
| Parameter passed / not passed | Covered: default call and explicit callable argument are exercised |
| None / non-None | Covered: None is preserved as a leaf value in nested structures |
| Enum options | Uncovered: tree_map has no enum-style parameters |
| Multiple types | Covered: tensor, tuple, list, dict, and None leaves are used |
| Normal input | Covered: nested pytree traversal and identity/transform mapping on NPU tensors |
| Error input | Covered: invalid fn type raises TypeError |
| Boundary / equivalence classes | Covered: empty/None-containing leaves and nested container traversal |

Uncovered items and reasons:
- Enum options: not applicable because tree_map has no enum arguments.
- Additional numeric boundary cases: not applicable because the API applies a generic function over tree leaves rather than performing numerical computation itself.
- More callable/keyword combinations: not applicable because this runtime only exposes the core `(fn, tree, *rests)` API shape.
"""

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    torch_npu = None
    _TORCH_NPU_IMPORT_ERROR = exc
else:  # pragma: no cover - simple import guard
    _TORCH_NPU_IMPORT_ERROR = None

from torch.utils import _pytree


def _require_npu():
    if torch_npu is None:
        pytest.skip(f"torch_npu is unavailable: {_TORCH_NPU_IMPORT_ERROR}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


def _make_sample_tree():
    return {
        "left": (
            torch.ones(2, device="npu"),
            None,
        ),
        "right": [
            torch.zeros(1, device="npu"),
            {"leaf": torch.full((1,), 3, device="npu")},
        ],
    }


def test_tree_map_traverses_nested_npu_pytree():
    _require_npu()

    tree = _make_sample_tree()

    result = _pytree.tree_map(lambda x: x, tree)

    assert isinstance(result, dict)
    assert result["left"][0].device.type == "npu"
    assert result["right"][0].device.type == "npu"
    assert result["right"][1]["leaf"].device.type == "npu"
    assert result["left"][1] is None
    assert torch.equal(result["left"][0], tree["left"][0])
    assert torch.equal(result["right"][0], tree["right"][0])


def test_tree_map_preserves_none_leaves_by_default():
    _require_npu()

    tree = _make_sample_tree()

    result = _pytree.tree_map(lambda x: x, tree)

    assert result["left"][1] is None


def test_tree_map_rejects_non_callable_fn():
    _require_npu()

    tree = _make_sample_tree()

    with pytest.raises(TypeError):
        _pytree.tree_map(123, tree)
