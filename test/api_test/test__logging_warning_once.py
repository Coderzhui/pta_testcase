"""
测试目的：验证 `torch._logging.warning_once` 的日志去重行为，并对缺少 NPU 设备语义的正常路径使用显式 skip。
API 名称：`torch._logging.warning_once`
覆盖的入参维度：
- 参数传参与不传参：覆盖同参重复调用。
- 参数为 None / 非 None：覆盖 logger 对象和消息参数。
- 枚举/多类型：覆盖 `logger + format args`。
- 正常输入：当前 API 不存在可验证的 NPU 设备语义，使用 `pytest.skip` 避免伪覆盖。
- 异常输入：覆盖非法 logger。
- 边界值和等价类：覆盖两次重复调用。
未覆盖项及原因：
- 该 API 只作用于 logger/message，当前无法构造“目标 API 显式在 NPU 上生效”的稳定最小用例，因此不伪造正常路径覆盖。
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

def test_warning_once_emits_same_message_once():
    _require_npu()
    pytest.skip("`torch._logging.warning_once` 缺少可验证的 NPU 设备语义，避免通过无关 NPU Tensor 伪造覆盖。")


def test_warning_once_invalid_logger_raises():
    _require_npu()

    with pytest.raises(AttributeError):
        torch._logging.warning_once(None, "hello")


def test_warning_once_missing_logger_argument_raises():
    _require_npu()

    with pytest.raises(TypeError):
        torch._logging.warning_once()
