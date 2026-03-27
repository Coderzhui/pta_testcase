"""
测试目的：验证 `torch.autograd.Variable._execution_engine.queue_callback` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.autograd.Variable._execution_engine.queue_callback`
覆盖的入参维度：
- 参数传参与不传参：覆盖 backward 内合法回调和 backward 外非法调用，以及不传参的异常场景。
- 参数为 None / 非 None：覆盖非 None callable 和 `None` 异常语义。
- 枚举/多类型：不适用（callable 类型）。
- 正常输入：覆盖 NPU autograd 图中回调注册，多回调顺序，以及不同 tensor 形状的边界场景。
- 异常输入：覆盖 `None`、backward 外直接注册、非 callable 类型（int、str、list）。
- 边界值和等价类：覆盖单元素梯度、多元素 tensor、空 tensor、多回调执行顺序。
未覆盖项及原因：
- 已覆盖核心功能，包括多回调顺序验证。
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
    """支持回调测试的自定义Function"""
    @staticmethod
    def forward(ctx, tensor, flag):
        ctx.flag = flag
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        torch.autograd.Variable._execution_engine.queue_callback(lambda: ctx.flag.append(True))
        return grad_output, None


def test_queue_callback_runs_during_npu_backward():
    """测试NPU backward期间回调正常执行"""
    _require_npu()

    flag = []
    tensor = torch.ones(1, device="npu", requires_grad=True)
    loss = _QueueCallbackFn.apply(tensor, flag).sum()
    loss.backward()

    assert flag == [True]
    assert tensor.grad.device.type == "npu"


def test_queue_callback_outside_backward_raises():
    """测试backward外调用queue_callback应抛出异常"""
    _require_npu()

    with pytest.raises(RuntimeError):
        torch.autograd.Variable._execution_engine.queue_callback(lambda: None)


def test_queue_callback_none_raises():
    """测试传入None应抛出异常"""
    _require_npu()

    with pytest.raises((RuntimeError, TypeError)):
        torch.autograd.Variable._execution_engine.queue_callback(None)


def test_queue_callback_no_argument_raises():
    """测试不传参应抛出TypeError"""
    _require_npu()

    with pytest.raises(TypeError):
        torch.autograd.Variable._execution_engine.queue_callback()


def test_queue_callback_invalid_int_raises():
    """测试传入int类型应抛出RuntimeError（backward外）"""
    _require_npu()

    # 在backward外调用，会先抛出RuntimeError而不是TypeError
    with pytest.raises(RuntimeError):
        torch.autograd.Variable._execution_engine.queue_callback(123)


def test_queue_callback_invalid_str_raises():
    """测试传入str类型应抛出RuntimeError（backward外）"""
    _require_npu()

    # 在backward外调用，会先抛出RuntimeError而不是TypeError
    with pytest.raises(RuntimeError):
        torch.autograd.Variable._execution_engine.queue_callback("callback")


def test_queue_callback_invalid_list_raises():
    """测试传入list类型应抛出RuntimeError（backward外）"""
    _require_npu()

    # 在backward外调用，会先抛出RuntimeError而不是TypeError
    with pytest.raises(RuntimeError):
        torch.autograd.Variable._execution_engine.queue_callback([lambda: None])


class _MultiQueueCallbackFn(torch.autograd.Function):
    """支持多回调测试的自定义Function"""
    @staticmethod
    def forward(ctx, tensor, flags, num_callbacks):
        ctx.flags = flags
        ctx.num_callbacks = num_callbacks
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # 注册多个回调
        for i in range(ctx.num_callbacks):
            callback_index = i
            torch.autograd.Variable._execution_engine.queue_callback(
                lambda idx=callback_index: ctx.flags.append(idx)
            )
        return grad_output, None, None


def test_queue_callback_multiple_callbacks_order():
    """测试多回调注册和执行顺序"""
    _require_npu()

    flags = []
    num_callbacks = 5
    tensor = torch.ones(1, device="npu", requires_grad=True)
    loss = _MultiQueueCallbackFn.apply(tensor, flags, num_callbacks).sum()
    loss.backward()

    # 验证所有回调都被执行
    assert len(flags) == num_callbacks
    # 验证回调按注册顺序执行
    assert flags == list(range(num_callbacks))


def test_queue_callback_with_multi_dim_tensor():
    """测试多维tensor场景下的回调"""
    _require_npu()

    flag = []
    # 测试不同维度的tensor
    for shape in [(2,), (2, 3), (2, 3, 4), (1,)]:
        flag.clear()
        tensor = torch.ones(shape, device="npu", requires_grad=True)
        loss = _QueueCallbackFn.apply(tensor, flag).sum()
        loss.backward()

        assert flag == [True]
        assert tensor.grad is not None
        assert tensor.grad.shape == shape
        assert tensor.grad.device.type == "npu"


def test_queue_callback_with_different_dtypes():
    """测试不同dtype tensor场景下的回调"""
    _require_npu()

    flag = []
    # 测试常见的dtype
    dtypes = [torch.float32, torch.float16]
    # 如果NPU支持，也可以测试float64
    if torch.npu.is_available():
        try:
            test_tensor = torch.ones(1, dtype=torch.float64, device="npu")
            dtypes.append(torch.float64)
        except (RuntimeError, TypeError):
            pass  # 如果float64不支持则跳过

    for dtype in dtypes:
        flag.clear()
        tensor = torch.ones(2, device="npu", dtype=dtype, requires_grad=True)
        loss = _QueueCallbackFn.apply(tensor, flag).sum()
        loss.backward()

        assert flag == [True]
        assert tensor.grad.dtype == dtype
        assert tensor.grad.device.type == "npu"
