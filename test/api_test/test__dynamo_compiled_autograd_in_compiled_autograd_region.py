# Test purpose: validate torch._dynamo.compiled_autograd.in_compiled_autograd_region state
# transitions around compiled autograd execution on NPU-backed tensors, plus basic misuse
# and error-recovery behavior.
# API name: torch._dynamo.compiled_autograd.in_compiled_autograd_region
#
# Covered parameter dimensions:
#
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | region state before / during / after | yes | default False, True inside compiled autograd runtime, restored after exit |
# | tensor shape boundary values | yes | scalar, zero-sized, and small dense NPU tensors |
# | normal input | yes | backward on NPU tensors under `_enable` |
# | abnormal input | yes | calling the boolean flag as a function raises `TypeError` |
# | runtime error recovery | yes | compiled function raising `RuntimeError` still restores the flag |
# | callable compiler_fn / non-callable compiler_fn | partial | callable compiler_fn exercised; non-callable is not a meaningful public API contract here |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | direct mutation of the flag | the API is a read-only module-level state variable; mutation is internal to compiled autograd. |
# | nested compiled-autograd reentry semantics | not needed for basic state coverage and not reliable in this minimal NPU-focused test. |
# | full Dynamo/Inductor compilation behavior | this file targets the region flag only and avoids testing unrelated compiler backends. |

import pytest
import torch
import torch_npu  # noqa: F401

from torch._dynamo import compiled_autograd as ca


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="torch_npu/NPU is not available in this environment.",
)


@pytest.mark.parametrize("shape", [(), (0,), (2,)])
def test_in_compiled_autograd_region_flips_true_inside_runtime(shape):
    seen = []

    def compiler_fn(gm):
        def compiled_fn(*args, **kwargs):
            seen.append(ca.in_compiled_autograd_region)
            assert ca.in_compiled_autograd_region is True
            return gm(*args, **kwargs)

        return compiled_fn

    x = torch.ones(shape, device="npu:0", requires_grad=True)

    assert ca.in_compiled_autograd_region is False
    with ca._enable(compiler_fn):
        (x * 2).sum().backward()
    assert seen == [True]
    assert ca.in_compiled_autograd_region is False
    assert x.grad is not None
    assert tuple(x.grad.shape) == tuple(shape)
    assert x.grad.device.type == "npu"


def test_in_compiled_autograd_region_resets_after_runtime_error():
    calls = []

    def compiler_fn(gm):
        def compiled_fn(*args, **kwargs):
            calls.append(ca.in_compiled_autograd_region)
            raise RuntimeError("compiled autograd failure")

        return compiled_fn

    x = torch.ones(1, device="npu:0", requires_grad=True)

    with pytest.raises(RuntimeError, match="compiled autograd failure"):
        with ca._enable(compiler_fn):
            (x.sin() * 3).sum().backward()

    assert calls == [True]
    assert ca.in_compiled_autograd_region is False


def test_in_compiled_autograd_region_is_not_callable():
    with pytest.raises(TypeError):
        ca.in_compiled_autograd_region()
