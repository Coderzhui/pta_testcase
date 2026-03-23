"""
测试目的：验证 `torch.Event` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.Event`
覆盖的入参维度：
- 参数传参与不传参：覆盖位置参数和关键字参数构造。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖 `enable_timing=True/False`。
- 正常输入：覆盖 NPU Event 构造与方法存在性。
- 异常输入：覆盖非法 device 字符串。
- 边界值和等价类：覆盖最小构造路径。
未覆盖项及原因：
- 未覆盖跨 stream 的时序语义，因为该阶段不运行长链路时序验证。
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

def test_event_constructs_on_npu():
    _require_npu()

    event_a = torch.Event("npu", enable_timing=False)
    event_b = torch.Event(device="npu", enable_timing=True)

    for event in (event_a, event_b):
        assert isinstance(event, torch.Event)
        assert str(event.device).startswith("npu")
        assert hasattr(event, "record")
        assert hasattr(event, "wait")


def test_event_invalid_device_raises():
    _require_npu()

    with pytest.raises(RuntimeError):
        torch.Event(device="bad", enable_timing=False)
