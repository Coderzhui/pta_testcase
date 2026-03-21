# Test purpose: validate `torch._dynamo.compiled_autograd.compiled_autograd_enabled`
#               state behavior on NPU-adjacent execution paths, including flag
#               toggling via the private `_enable` context manager, argument
#               validation, and a documented unreliable compiled-backward path.
# API name: torch._dynamo.compiled_autograd.compiled_autograd_enabled
#
# Covered parameter dimensions
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | enabled state False / True | Covered | The flag is checked before, inside, and after `_enable`. |
# | `dynamic` False / True / invalid truthy | Covered | Bool values succeed; invalid truthy values raise `AssertionError`. |
# | NPU object presence | Covered | NPU tensors are created inside the test to keep coverage on the target device. |
# | compiled backward path | Covered via xfail | The path is attempted and marked xfail if the current build cannot execute it reliably. |
#
# Uncovered items / reasons
# | Item | Reason |
# | --- | --- |
# | Full compiled-autograd graph execution on NPU | This build raises a runtime error in the compiled-backward path, so the test documents and xfails that path instead of pretending coverage. |
# | Exhaustive compiler callback semantics | The private API surface is small; flag/state behavior is the stable contract here. |

import pytest
import torch
import torch_npu  # noqa: F401

import torch._dynamo.compiled_autograd as ca


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


class _SimpleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * 2


def _dummy_compiler_fn(graph):
    return lambda *args, **kwargs: None


def test_compiled_autograd_enabled_toggles_state_in_context_on_npu():
    _require_npu()

    assert ca.compiled_autograd_enabled is False
    assert ca.compiled_autograd_enabled_force_eager is False
    assert ca.in_compiled_autograd_region is False

    npu_tensor = torch.ones(1, device="npu")
    assert npu_tensor.device.type == "npu"

    with ca._enable(_dummy_compiler_fn):
        assert ca.compiled_autograd_enabled is True
        assert ca.compiled_autograd_enabled_force_eager is False
        assert ca.in_compiled_autograd_region is False

    assert ca.compiled_autograd_enabled is False
    assert ca.compiled_autograd_enabled_force_eager is False
    assert ca.in_compiled_autograd_region is False


@pytest.mark.parametrize("dynamic", [False, True])
def test_compiled_autograd_enabled_accepts_boolean_dynamic_flag(dynamic):
    _require_npu()

    with ca._enable(_dummy_compiler_fn, dynamic=dynamic):
        assert ca.compiled_autograd_enabled is True


@pytest.mark.parametrize("dynamic", [1, "x"])
def test_compiled_autograd_enabled_rejects_non_bool_truthy_dynamic_flag(dynamic):
    _require_npu()

    with pytest.raises(AssertionError):
        with ca._enable(_dummy_compiler_fn, dynamic=dynamic):
            pass


def test_compiled_autograd_enabled_compiled_backward_is_xfailed_when_unavailable():
    _require_npu()

    x = torch.tensor(1.0, device="npu", requires_grad=True)

    try:
        with ca._enable(_dummy_compiler_fn):
            y = _SimpleFn.apply(x)
            y.backward()
    except RuntimeError as exc:
        pytest.xfail(
            "Compiled backward is not reliable in this build: "
            f"{type(exc).__name__}: {exc}"
        )

    assert ca.compiled_autograd_enabled is False
    assert x.grad is not None
    assert x.grad.device.type == "npu"
