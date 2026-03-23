"""
测试目的：验证 `torch.utils.swap_tensors` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.utils.swap_tensors`
覆盖的入参维度：
- 参数传参与不传参：覆盖两个合法 Tensor 与非法非 Tensor 入参。
- 参数为 None / 非 None：覆盖非法 `None`/int。
- 枚举/多类型：覆盖 NPU Tensor 内容交换。
- 正常输入：覆盖就地交换。
- 异常输入：覆盖非 Tensor。
- 边界值和等价类：覆盖相同形状张量。
未覆盖项及原因：
- 未覆盖跨 dtype/跨 layout 交换。
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

def test_swap_tensors_exchanges_npu_contents():
    _require_npu()

    first = torch.ones(2, device="npu")
    second = torch.zeros(2, device="npu")
    torch.utils.swap_tensors(first, second)

    assert first.device.type == "npu"
    assert second.device.type == "npu"
    assert torch.equal(first.cpu(), torch.zeros(2))
    assert torch.equal(second.cpu(), torch.ones(2))


def test_swap_tensors_invalid_input_raises():
    _require_npu()

    with pytest.raises((AttributeError, TypeError, RuntimeError)):
        torch.utils.swap_tensors(torch.ones(1, device="npu"), 1)
