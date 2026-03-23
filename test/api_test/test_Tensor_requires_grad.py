"""
测试目的：验证 `Tensor.requires_grad` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`Tensor.requires_grad`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为只读/可赋值属性访问。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖 `False` / `True` 两种布尔状态。
- 正常输入：覆盖 NPU Tensor 的 eager 属性读取和 `requires_grad_` 切换。
- 异常输入：覆盖非法非布尔赋值。
- 边界值和等价类：覆盖 `detach()` 后的 `False` 状态。
未覆盖项及原因：
- 该属性没有独立的位置参数签名，因此未构造“缺参”用例。
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

def test_tensor_requires_grad_property_on_npu():
    _require_npu()

    plain = torch.ones(2, device="npu")
    trainable = torch.ones(2, device="npu", requires_grad=True)

    assert plain.requires_grad is False
    assert trainable.requires_grad is True
    assert trainable.detach().requires_grad is False

    plain.requires_grad_(True)
    assert plain.requires_grad is True


def test_tensor_requires_grad_invalid_assignment_raises():
    _require_npu()

    tensor = torch.ones(1, device="npu")
    with pytest.raises(RuntimeError):
        tensor.requires_grad = "bad"
