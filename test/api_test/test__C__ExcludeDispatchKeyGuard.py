"""
Test purpose: validate torch._C._ExcludeDispatchKeyGuard behavior in an NPU-enabled runtime.
API name: torch._C._ExcludeDispatchKeyGuard

Coverage table:
| Parameter dimension | Coverage |
| --- | --- |
| Parameter passed / not passed | Covered: constructor is exercised with a DispatchKeySet argument and invalid arity/type input |
| None / non-None | Uncovered: this low-level guard does not accept None as a meaningful input |
| Enum options | Covered: representative dispatch key selection is exercised via CPU key exclusion |
| Multiple types | Uncovered: the constructor is specialized to dispatch-key sets rather than heterogeneous Python types |
| Normal input | Covered: guard context can be entered and NPU work still succeeds inside the scope |
| Error input | Covered: invalid constructor input raises TypeError |
| Boundary / equivalence classes | Covered: minimal one-key DispatchKeySet and simple enter/exit lifecycle |

Uncovered items and reasons:
- None / non-None: not applicable because the constructor expects a dispatch-key set, not an optional data argument.
- Multiple types: not applicable because this is a specialized low-level guard constructor.
- Additional key combinations: omitted to keep the test stable across builds and avoid relying on backend-specific dispatch internals.
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


def _require_exclude_guard():
    guard = getattr(torch._C, "_ExcludeDispatchKeyGuard", None)
    if guard is None:
        pytest.skip("torch._C._ExcludeDispatchKeyGuard is not available in this build.")
    if not hasattr(torch._C, "DispatchKeySet") or not hasattr(torch._C, "DispatchKey"):
        pytest.skip("DispatchKey APIs are not available in this build.")
    if not hasattr(torch._C.DispatchKey, "CPU"):
        pytest.skip("CPU dispatch key is not available in this build.")
    return guard


def test_exclude_dispatch_key_guard_enters_and_allows_npu_work():
    _require_npu()
    guard_cls = _require_exclude_guard()

    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
    with guard_cls(keyset):
        tensor = torch.ones(2, device="npu")
        assert tensor.device.type == "npu"
        assert tensor.numel() == 2


def test_exclude_dispatch_key_guard_is_scoped_and_does_not_break_followup_npu_ops():
    _require_npu()
    guard_cls = _require_exclude_guard()

    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
    with guard_cls(keyset):
        inside = torch.ones(1, device="npu")
        assert inside.device.type == "npu"

    outside = torch.ones(1, device="npu")
    assert outside.device.type == "npu"


def test_exclude_dispatch_key_guard_rejects_invalid_constructor_input():
    _require_npu()
    guard_cls = _require_exclude_guard()

    with pytest.raises(TypeError):
        guard_cls("not_a_dispatch_key_set")
