# Test purpose: validate `torch.autograd.Variable._execution_engine.queue_callback`
#               behavior on NPU during autograd backward execution, including
#               accepted callback registration, callback execution order, and
#               reliable failure outside backward.
# API name: torch.autograd.Variable._execution_engine.queue_callback
#
# Covered parameter dimensions
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | call during backward / outside backward | Covered | Positive path queues from inside backward; error path calls outside backward. |
# | callback callable / behavior | Covered | Callback is a Python callable and is executed after being queued. |
# | single vs multiple callbacks | Covered | Both single and multi-callback registration are exercised. |
# | return value | Covered | `None` return is asserted. |
# | output device behavior | Covered | Backward runs on NPU and produces NPU gradients. |
#
# Uncovered items / reasons
# | Item | Reason |
# | --- | --- |
# | Exhaustive invalid callback type matrix inside backward | Outside-backward failure is the reliable documented error path in this environment. |
# | Cross-device backward variants | This API is internal to autograd engine behavior and the NPU path is the required target. |

import pytest
import torch
import torch_npu  # noqa: F401


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


class _QueueCallbackFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, engine, callbacks):
        ctx.engine = engine
        ctx.callbacks = callbacks
        return x * 2

    @staticmethod
    def backward(ctx, grad_out):
        for cb in ctx.callbacks:
            ret = ctx.engine.queue_callback(cb)
            assert ret is None
        return grad_out * 2, None, None


def test_queue_callback_runs_during_backward_on_npu():
    _require_npu()
    engine = torch.autograd.Variable._execution_engine
    seen = []

    def cb():
        seen.append("called")

    x = torch.tensor(1.0, device="npu", requires_grad=True)
    y = _QueueCallbackFn.apply(x, engine, [cb])

    y.backward()

    assert y.device.type == "npu"
    assert x.grad is not None
    assert x.grad.device.type == "npu"
    assert tuple(x.grad.shape) == ()
    assert seen == ["called"]


def test_queue_callback_multiple_callbacks_preserve_order():
    _require_npu()
    engine = torch.autograd.Variable._execution_engine
    seen = []

    def first():
        seen.append("first")

    def second():
        seen.append("second")

    x = torch.tensor(1.0, device="npu", requires_grad=True)
    y = _QueueCallbackFn.apply(x, engine, [first, second])

    y.backward()

    assert seen == ["first", "second"]
    assert x.grad is not None
    assert x.grad.device.type == "npu"


def test_queue_callback_outside_backward_raises():
    _require_npu()
    engine = torch.autograd.Variable._execution_engine

    with pytest.raises(RuntimeError, match="Final callbacks can only be installed during backward pass"):
        engine.queue_callback(lambda: None)
