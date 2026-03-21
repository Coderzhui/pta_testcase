# Test purpose: verify Tensor.register_hook behavior on NPU tensors, including
# hook registration, hook removal, and error handling for invalid inputs.
# API name: torch.Tensor.register_hook
#
# Covered parameter dimensions:
# | Dimension | Covered cases |
# | --- | --- |
# | hook parameter presence | provided explicitly in all tests |
# | hook parameter type | callable function, callable object, non-callable object |
# | tensor requires_grad | True, False |
# | runtime behavior | hook invoked, hook removed before backward, backward error path |
# | output device behavior | NPU tensor/grad/device observed on successful runs |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | missing hook argument | not applicable; the API requires one positional hook argument |
# | numeric accuracy validation | intentionally omitted; the test targets interface and behavior only |

import pytest
import torch
import torch_npu  # noqa: F401 - imported to ensure NPU runtime is initialized.


def _require_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU device is not available in this environment.")
    torch_npu.npu.set_device(0)


def test_register_hook_runs_on_npu_and_populates_grad_device():
    _require_npu()

    x = torch.tensor([1.0], device="npu:0", requires_grad=True)
    seen = {"calls": 0, "device": None}

    def hook(grad):
        seen["calls"] += 1
        seen["device"] = grad.device.type
        return grad

    handle = x.register_hook(hook)
    y = (x * 2).sum()
    y.backward()

    assert isinstance(handle, torch.utils.hooks.RemovableHandle)
    assert seen["calls"] == 1
    assert seen["device"] == "npu"
    assert x.grad is not None
    assert x.grad.device.type == "npu"


def test_register_hook_remove_prevents_invocation():
    _require_npu()

    x = torch.tensor([1.0], device="npu:0", requires_grad=True)
    called = []

    class CallableHook:
        def __call__(self, grad):
            called.append(grad.device.type)
            return grad

    handle = x.register_hook(CallableHook())
    handle.remove()
    (x * 2).sum().backward()

    assert called == []
    assert x.grad is not None
    assert x.grad.device.type == "npu"


def test_register_hook_rejects_tensor_without_grad():
    _require_npu()

    x = torch.tensor([1.0], device="npu:0", requires_grad=False)

    with pytest.raises(RuntimeError, match="doesn't require gradient"):
        x.register_hook(lambda grad: grad)


def test_register_hook_non_callable_fails_on_backward():
    _require_npu()

    x = torch.tensor([1.0], device="npu:0", requires_grad=True)
    handle = x.register_hook(123)

    with pytest.raises(TypeError, match="not callable"):
        (x * 2).sum().backward()

    handle.remove()
