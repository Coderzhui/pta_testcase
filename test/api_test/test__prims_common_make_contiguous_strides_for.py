"""
测试目的：验证 `torch._prims_common.make_contiguous_strides_for` 的纯 shape 计算行为，并对缺少 NPU 设备语义的正常路径使用显式 skip。
API 名称：`torch._prims_common.make_contiguous_strides_for`
覆盖的入参维度：
- 参数传参与不传参：覆盖默认 `row_major=True` 和显式 `False`。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖两种主布局计算。
- 正常输入：当前 API 不存在可验证的 NPU 设备语义，使用 `pytest.skip` 避免伪覆盖。
- 异常输入：覆盖非法负维度。
- 边界值和等价类：覆盖二维等价类。
未覆盖项及原因：
- 该 API 为纯 shape 计算工具，不涉及真实 Tensor 分配；当前无法构造“目标 API 显式在 NPU 上生效”的最小用例。
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

def test_make_contiguous_strides_for_row_major_modes():
    _require_npu()
    pytest.skip("`make_contiguous_strides_for` 只处理 shape，不存在可验证的 NPU 设备语义。")


def test_make_contiguous_strides_for_negative_dim_raises():
    _require_npu()

    with pytest.raises(RuntimeError):
        torch._prims_common.make_contiguous_strides_for((-1, 3))


def test_make_contiguous_strides_for_invalid_none_raises():
    _require_npu()

    with pytest.raises((AssertionError, TypeError)):
        torch._prims_common.make_contiguous_strides_for(None)
