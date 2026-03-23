"""
测试目的：验证 `Tensor.register_hook` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`Tensor.register_hook`
覆盖的入参维度：
- 参数传参与不传参：覆盖合法 hook 和非法非 callable hook。
- 参数为 None / 非 None：覆盖非 None 的 lambda hook，以及 `None` 异常分支。
- 枚举/多类型：不适用，该 API 仅接收单个 hook 参数。
- 正常输入：覆盖 NPU Tensor 的反向传播回调。
- 异常输入：覆盖非 callable 入参。
- 边界值和等价类：覆盖单元素梯度。
未覆盖项及原因：
- 更复杂的 hook 链顺序不属于该 API 的最小稳定用例。
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

def test_tensor_register_hook_runs_on_npu_backward():
    _require_npu()

    tensor = torch.ones(3, device="npu", requires_grad=True)
    seen = []
    handle = tensor.register_hook(lambda grad: seen.append(grad.clone()) or grad)
    (tensor * 2).sum().backward()

    assert type(handle).__name__ == "RemovableHandle"
    assert len(seen) == 1
    assert seen[0].device.type == "npu"
    assert tensor.grad.device.type == "npu"
    handle.remove()


def test_tensor_register_hook_invalid_hook_raises_on_backward():
    _require_npu()

    tensor = torch.ones(1, device="npu", requires_grad=True)
    tensor.register_hook(None)

    with pytest.raises(TypeError):
        tensor.sum().backward()
