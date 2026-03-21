"""
Test purpose: validate that torch.nn.Parameter.device.type reports the expected device string on NPU-backed parameters.
API name: torch.nn.Parameter.device.type

Coverage table:
| Parameter dimension | Coverage |
| --- | --- |
| Parameter passed / not passed | Uncovered: device.type is a property, not a callable API |
| None / non-None | Covered: valid NPU parameters are non-None; invalid construction is checked separately |
| Enum options | Uncovered: no enum-style arguments exist for this property chain |
| Multiple types | Uncovered: the property has no direct input types; coverage is on parameter shapes/devices |
| Normal input | Covered: regular and zero-sized NPU parameters report device.type == "npu" |
| Error input | Covered: invalid Parameter construction raises AttributeError |
| Boundary / equivalence classes | Covered: zero-sized vs non-zero-sized parameter tensors on NPU |

Uncovered items and reasons:
- Parameter passed / not passed: not applicable because device.type is a property, not a function.
- Enum options: not applicable because there are no enumerated parameters.
- Multiple types: not applicable because the property accepts no inputs.
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
    return torch.nn.Parameter(torch.ones(shape, device="npu"))


def test_parameter_device_type_reports_npu_for_regular_and_boundary_shapes():
    _require_npu()

    regular = _make_parameter((2, 3))
    boundary = _make_parameter((0,))

    assert isinstance(regular, torch.nn.Parameter)
    assert isinstance(boundary, torch.nn.Parameter)
    assert regular.device.type == "npu"
    assert boundary.device.type == "npu"
    assert regular.device == torch.device("npu:0")
    assert boundary.device == torch.device("npu:0")


def test_parameter_device_type_is_stable_after_tensor_operations_on_npu():
    _require_npu()

    parameter = _make_parameter((1,))
    moved = parameter.detach().clone()

    assert moved.device.type == "npu"
    assert parameter.device.type == moved.device.type


def test_parameter_construction_rejects_invalid_data_for_device_type_setup():
    _require_npu()

    with pytest.raises(AttributeError, match="detach"):
        torch.nn.Parameter("not_a_tensor")
