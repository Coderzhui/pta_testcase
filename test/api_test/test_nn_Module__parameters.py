"""
Test purpose: validate the internal torch.nn.Module._parameters registry on an NPU-backed module.
API name: torch.nn.Module._parameters

Coverage table:
| Parameter dimension | Coverage |
| --- | --- |
| Parameter passed / not passed | Covered: empty registry and populated registry states are exercised |
| None / non-None | Covered: registered None values and real Parameter values are both covered through the registry |
| Enum options | Uncovered: _parameters is a registry dict, not an enum-style API |
| Multiple types | Covered: real Parameter, None placeholder, and invalid registration input are exercised |
| Normal input | Covered: registering a valid NPU Parameter updates the registry and preserves NPU device type |
| Error input | Covered: invalid parameter registration raises TypeError / KeyError where appropriate |
| Boundary / equivalence classes | Covered: empty registry vs single-entry registry and None placeholder registration |

Uncovered items and reasons:
- Enum options: not applicable because _parameters is a storage registry rather than an enum-driven API.
- Additional numeric boundary cases: not applicable because this registry stores parameters instead of computing with them.
- Deep internal invariants beyond registration semantics: omitted to avoid relying on private implementation details that may differ across builds.
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


def test_module_parameters_registry_is_empty_then_populated_on_npu():
    _require_npu()

    module = torch.nn.Module()
    assert isinstance(module._parameters, dict)
    assert module._parameters == {}

    param = torch.nn.Parameter(torch.ones(2, device="npu"))
    module.register_parameter("weight", param)

    assert "weight" in module._parameters
    assert module._parameters["weight"] is param
    assert module._parameters["weight"].device.type == "npu"


def test_module_parameters_registry_supports_none_placeholder():
    _require_npu()

    module = torch.nn.Module()
    module.register_parameter("bias", None)

    assert "bias" in module._parameters
    assert module._parameters["bias"] is None


def test_module_parameters_registry_rejects_invalid_registration():
    _require_npu()

    module = torch.nn.Module()

    with pytest.raises((TypeError, KeyError)):
        module.register_parameter("", torch.nn.Parameter(torch.ones(1, device="npu")))

    with pytest.raises(TypeError):
        module.register_parameter("not_a_param", torch.ones(1, device="npu"))
