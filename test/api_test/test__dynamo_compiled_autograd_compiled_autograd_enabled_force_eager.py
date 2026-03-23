"""
测试目的：验证 `torch._dynamo.compiled_autograd.compiled_autograd_enabled_force_eager` 在 NPU eager 场景下可读取并返回稳定的布尔结果。
API 名称：`torch._dynamo.compiled_autograd.compiled_autograd_enabled_force_eager`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为布尔配置/状态读取。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖布尔返回值。
- 正常输入：覆盖 NPU eager 场景。
- 异常输入：无独立异常签名。
- 边界值和等价类：覆盖当前默认值。
未覆盖项及原因：
- 该类 API 为状态读取，不覆盖全局配置变更流程。
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

def test_flag_like_api_returns_bool_in_npu_eager_mode():
    _require_npu()

    tensor = torch.ones(1, device="npu")
    value = torch._dynamo.compiled_autograd.compiled_autograd_enabled_force_eager

    assert tensor.device.type == "npu"
    assert isinstance(value, bool)
