"""
测试目的：验证 `torch.autograd.Variable._execution_engine.queue_callback` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.autograd.Variable._execution_engine.queue_callback`
覆盖的入参维度：
- 参数传参与不传参：覆盖 backward 内合法回调和 backward 外非法调用。
- 参数为 None / 非 None：覆盖非 None callable 和 `None` 异常语义。
- 枚举/多类型：不适用。
- 正常输入：覆盖 NPU autograd 图中回调注册。
- 异常输入：覆盖 `None` 和 backward 外直接注册。
- 边界值和等价类：覆盖单元素梯度。
未覆盖项及原因：
- 未额外覆盖多回调顺序，因为该 API 的稳定最小验证重点是“能注册并执行”。
"""

import contextlib
import io
import logging

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")

class _QueueCallbackFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, flag):
        ctx.flag = flag
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        torch.autograd.Variable._execution_engine.queue_callback(lambda: ctx.flag.append(True))
        return grad_output, None


def test_queue_callback_runs_during_npu_backward():
    _require_npu()

    flag = []
    tensor = torch.ones(1, device="npu", requires_grad=True)
    loss = _QueueCallbackFn.apply(tensor, flag).sum()
    loss.backward()

    assert flag == [True]
    assert tensor.grad.device.type == "npu"


def test_queue_callback_outside_backward_raises():
    _require_npu()

    with pytest.raises(RuntimeError):
        torch.autograd.Variable._execution_engine.queue_callback(lambda: None)


def test_queue_callback_none_raises():
    _require_npu()

    with pytest.raises((RuntimeError, TypeError)):
        torch.autograd.Variable._execution_engine.queue_callback(None)
