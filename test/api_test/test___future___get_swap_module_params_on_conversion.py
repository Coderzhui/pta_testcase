"""
Test purpose: validate the behavior of torch.__future__.get_swap_module_params_on_conversion on an NPU-enabled runtime.
API name: torch.__future__.get_swap_module_params_on_conversion

Coverage table:
| Parameter dimension | Coverage |
| --- | --- |
| Parameter passed / not passed | Covered: getter called with no parameters; error case checks unexpected parameters |
| None / non-None | Uncovered: the API accepts no arguments |
| Enum options | Uncovered: the API accepts no enum-like arguments |
| Multiple types | Uncovered: the API accepts no typed inputs |
| Normal input | Covered: getter returns a bool; paired setter is exercised when available |
| Error input | Covered: wrong-arity call raises TypeError |
| Boundary / equivalence classes | Covered: repeated calls are stable; state round-trip is checked when setter exists |

Uncovered items and reasons:
- None / non-None, enum options, multiple types: not applicable because this getter takes no arguments.
- Direct value boundary checks on input arguments: not applicable for the same reason.
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


def test_get_swap_module_params_on_conversion_returns_bool_and_is_stable():
    _require_npu()

    npu_tensor = torch.tensor([1], device="npu")
    assert npu_tensor.device.type == "npu"

    getter = torch.__future__.get_swap_module_params_on_conversion
    first = getter()
    second = getter()

    assert isinstance(first, bool)
    assert isinstance(second, bool)
    assert first == second


def test_get_swap_module_params_on_conversion_rejects_unexpected_arguments():
    _require_npu()

    with pytest.raises(TypeError):
        torch.__future__.get_swap_module_params_on_conversion(True)


def test_get_swap_module_params_on_conversion_round_trip_with_setter_if_available():
    _require_npu()

    setter = getattr(torch.__future__, "set_swap_module_params_on_conversion", None)
    if setter is None:
        pytest.skip("torch.__future__.set_swap_module_params_on_conversion is not available in this build.")

    getter = torch.__future__.get_swap_module_params_on_conversion
    original = getter()

    try:
        setter(not original)
        assert getter() is (not original)

        setter(original)
        assert getter() is original
    finally:
        setter(original)
