"""
测试目的：验证 `torch.nn.Parameter.grad` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Parameter.grad`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为属性访问。
- 参数为 None / 非 None：覆盖 backward 前 `None` 和 backward 后 Tensor。
- 枚举/多类型：覆盖属性空值与 Tensor 值。
- 正常输入：覆盖 NPU Parameter 梯度生成。
- 异常输入：覆盖非法非 Tensor 赋值。
- 边界值和等价类：覆盖单参数标量 loss。
未覆盖项及原因：
- 未覆盖梯度累积多步行为。
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

def test_parameter_grad_none_then_tensor_on_npu():
    _require_npu()

    parameter = torch.nn.Parameter(torch.ones(2, device="npu"))
    assert parameter.grad is None

    (parameter * 2).sum().backward()
    assert isinstance(parameter.grad, torch.Tensor)
    assert parameter.grad.device.type == "npu"


def test_parameter_grad_invalid_assignment_raises():
    _require_npu()

    parameter = torch.nn.Parameter(torch.ones(1, device="npu"))
    with pytest.raises(TypeError):
        parameter.grad = "bad"
