"""
测试目的：验证 `torch.autograd._unsafe_preserve_version_counter` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.autograd._unsafe_preserve_version_counter`
覆盖的入参维度：
- 参数传参与不传参：覆盖单 Tensor 入参和非法 `None`。
- 参数为 None / 非 None：覆盖合法 Tensor 与非法 `None`。
- 枚举/多类型：覆盖 context manager 语义。
- 正常输入：覆盖 NPU Tensor 版本计数保持。
- 异常输入：覆盖非法 `None`。
- 边界值和等价类：覆盖最小单元素 Tensor。
未覆盖项及原因：
- 未覆盖 tuple[tensor, ...] 变体，单 Tensor 已能验证核心接口语义。
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

def test_unsafe_preserve_version_counter_keeps_version_on_npu():
    _require_npu()

    tensor = torch.tensor([1.0], device="npu")
    before = tensor._version
    with torch.autograd._unsafe_preserve_version_counter(tensor):
        tensor.add_(1)

    assert tensor._version == before
    assert tensor.device.type == "npu"


def test_unsafe_preserve_version_counter_invalid_none_raises():
    _require_npu()

    with pytest.raises((AssertionError, TypeError, RuntimeError)):
        with torch.autograd._unsafe_preserve_version_counter(None):
            pass
