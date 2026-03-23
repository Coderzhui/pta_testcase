"""
测试目的：验证 `torch._dynamo.config.skip_fsdp_hooks` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch._dynamo.config.skip_fsdp_hooks`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为布尔配置读取。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖布尔返回值。
- 正常输入：覆盖 NPU 上下文中的属性读取。
- 异常输入：无独立异常签名。
- 边界值和等价类：覆盖当前默认值。
未覆盖项及原因：
- 该阶段不修改全局配置以避免污染外层 pipeline。
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

def test_skip_fsdp_hooks_is_bool_in_npu_context():
    _require_npu()

    tensor = torch.ones(1, device="npu")
    assert tensor.device.type == "npu"
    assert isinstance(torch._dynamo.config.skip_fsdp_hooks, bool)
