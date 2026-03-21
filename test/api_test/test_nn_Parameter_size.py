"""
Test purpose: validate torch.nn.Parameter.size behavior on NPU-backed parameters.
API name: torch.nn.Parameter.size

Coverage table:
| Parameter dimension | Coverage |
| --- | --- |
| Parameter passed / not passed | Covered: size() and size(dim) are both exercised |
| None / non-None | Covered: no-arg call and explicit integer dim are exercised |
| Enum options | Uncovered: size has no enum-style parameters |
| Multiple types | Covered: integer dim and invalid non-integer dim error path |
| Normal input | Covered: regular NPU parameters return torch.Size and int dimension sizes |
| Error input | Covered: invalid dim type raises a runtime error in this build |
| Boundary / equivalence classes | Covered: zero-sized dimensions, 1-D and 2-D shapes, and last-dimension access |

Uncovered items and reasons:
- Enum options: not applicable because size has no enum arguments.
- Additional input types beyond integers: only integer dimension selection is supported for the optional argument.
- Value-range errors beyond type checking: not included because dim values are validated by standard index semantics and the reliable cross-build behavior is the type error path.
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


def _require_npu():
    if torch_npu is None:
        pytest.skip(f"torch_npu is unavailable: {_TORCH_NPU_IMPORT_ERROR}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


def _make_parameter(shape):
    return torch.nn.Parameter(torch.arange(max(1, int(torch.tensor(shape).numel() if isinstance(shape, tuple) else 1)), device="npu").reshape(shape))


def test_parameter_size_returns_torch_size_and_dim_size_on_npu():
    _require_npu()

    parameter = torch.nn.Parameter(torch.ones((2, 3), device="npu"))

    full_size = parameter.size()
    dim0 = parameter.size(0)
    dim1 = parameter.size(1)
    last_dim = parameter.size(-1)

    assert isinstance(full_size, torch.Size)
    assert full_size == torch.Size([2, 3])
    assert dim0 == 2
    assert dim1 == 3
    assert last_dim == 3


def test_parameter_size_handles_boundary_shapes_on_npu():
    _require_npu()

    zero_dim = torch.nn.Parameter(torch.ones((0, 4), device="npu"))
    one_dim = torch.nn.Parameter(torch.ones((5,), device="npu"))

    assert zero_dim.size() == torch.Size([0, 4])
    assert zero_dim.size(0) == 0
    assert zero_dim.size(1) == 4
    assert one_dim.size() == torch.Size([5])
    assert one_dim.size(-1) == 5


def test_parameter_size_rejects_non_integer_dim():
    _require_npu()

    parameter = torch.nn.Parameter(torch.ones((2, 2), device="npu"))

    with pytest.raises((TypeError, RuntimeError)):
        parameter.size("0")
