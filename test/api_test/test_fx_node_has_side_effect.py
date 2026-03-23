"""
测试目的：说明 `torch.fx.node.has_side_effect` 在当前环境中缺少可验证的 NPU 设备语义，避免伪覆盖。
API 名称：`torch.fx.node.has_side_effect`
覆盖的入参维度：
- 参数传参与不传参：当前环境不存在可验证的 NPU 设备语义，正常路径不做伪覆盖。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：不适用。
- 正常输入：使用 `pytest.skip` 说明阻塞原因。
- 异常输入：当前版本缺少稳定异常签名。
- 边界值和等价类：不适用。
未覆盖项及原因：
- 该 API 只标记 Python callable，当前无法构造“目标 API 显式在 NPU 上生效”的稳定最小用例，因此不伪造覆盖。
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

def test_has_side_effect_marks_function():
    _require_npu()
    pytest.skip("`torch.fx.node.has_side_effect` 缺少可验证的 NPU 设备语义，避免通过无关 NPU Tensor 伪造覆盖。")
