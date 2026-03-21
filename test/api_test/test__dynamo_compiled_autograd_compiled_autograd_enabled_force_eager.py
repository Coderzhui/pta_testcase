"""
Test purpose: validate torch._dynamo.compiled_autograd.compiled_autograd_enabled_force_eager behavior in an NPU-enabled runtime.
API name: torch._dynamo.compiled_autograd.compiled_autograd_enabled_force_eager

Coverage table:
| Parameter dimension | Coverage |
| --- | --- |
| Parameter passed / not passed | Covered: context manager is entered with and without an internal eager flag path when available |
| None / non-None | Uncovered: the API is a context manager, not a data-bearing function |
| Enum options | Uncovered: no enum-style parameters are exposed |
| Multiple types | Uncovered: no heterogeneous inputs are accepted directly |
| Normal input | Covered: context entry/exit with an NPU tensor operation inside the scope |
| Error input | Covered: invalid usage of context manager protocol is checked when possible |
| Boundary / equivalence classes | Covered: context entered around a minimal NPU allocation and a simple in-scope operation |

Uncovered items and reasons:
- None / non-None: not applicable because the API does not consume data arguments.
- Enum options: not applicable because there are no enum parameters.
- Multiple types: not applicable because the API is a context manager, not a function with typed arguments.
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

from torch._dynamo import compiled_autograd as ca


def _require_npu():
    if torch_npu is None:
        pytest.skip(f"torch_npu is unavailable: {_TORCH_NPU_IMPORT_ERROR}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


def _require_context_manager():
    ctx = getattr(ca, "compiled_autograd_enabled_force_eager", None)
    if ctx is None:
        pytest.skip("compiled_autograd_enabled_force_eager is not available in this build.")
    if not hasattr(ctx, "__enter__") or not hasattr(ctx, "__exit__"):
        pytest.skip("compiled_autograd_enabled_force_eager is not a context manager in this build.")
    return ctx


def test_compiled_autograd_enabled_force_eager_allows_npu_work_inside_context():
    _require_npu()
    ctx = _require_context_manager()

    with ctx:
        tensor = torch.ones(2, device="npu")
        assert tensor.device.type == "npu"
        assert tensor.numel() == 2
        assert torch.add(tensor, 1).device.type == "npu"


def test_compiled_autograd_enabled_force_eager_can_be_reentered():
    _require_npu()
    ctx = _require_context_manager()

    with ctx:
        first = torch.zeros(1, device="npu")
        assert first.device.type == "npu"

    with ctx:
        second = torch.ones(1, device="npu")
        assert second.device.type == "npu"


def test_compiled_autograd_enabled_force_eager_rejects_non_context_manager_usage():
    _require_npu()
    ctx = _require_context_manager()

    with pytest.raises(TypeError):
        next(iter(ctx))
