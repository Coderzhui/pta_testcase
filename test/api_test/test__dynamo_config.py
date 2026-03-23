"""
测试目的：验证 `torch._dynamo.config` 模块对象在 NPU 上下文中可稳定访问关键配置属性。
API 名称：`torch._dynamo.config`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为模块对象访问。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖布尔配置项访问。
- 正常输入：覆盖模块属性读取。
- 异常输入：覆盖缺失属性访问。
- 边界值和等价类：覆盖已有配置项。
未覆盖项及原因：
- 未覆盖全局配置修改回滚。
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

def test_dynamo_config_module_is_accessible_from_npu_context():
    _require_npu()

    tensor = torch.ones(1, device="npu")
    assert tensor.device.type == "npu"
    assert isinstance(torch._dynamo.config.skip_fsdp_hooks, bool)
    assert hasattr(torch._dynamo.config, "verbose")


def test_dynamo_config_missing_attr_raises():
    _require_npu()

    with pytest.raises(AttributeError):
        getattr(torch._dynamo.config, "not_a_real_config")
