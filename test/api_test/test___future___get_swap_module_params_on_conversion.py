"""
测试目的：验证 `torch.__future__.get_swap_module_params_on_conversion` 可在 NPU 场景下稳定读取并与 setter 形成正确的配置 round-trip。
API 名称：`torch.__future__.get_swap_module_params_on_conversion`
覆盖的入参维度：
- 参数传参与不传参：覆盖 getter 无参调用和 setter 切换后的 getter 调用。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖 `True` / `False` 两种布尔配置。
- 正常输入：覆盖设置项读取和 NPU Module 转换。
- 异常输入：无直接异常签名，使用 setter 恢复现场。
- 边界值和等价类：覆盖配置 round-trip。
未覆盖项及原因：
- 该 API 为全局配置 getter，本文件不验证更大范围的模块转换副作用。
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

def test_get_swap_module_params_on_conversion_round_trip_with_npu_module():
    _require_npu()

    original = torch.__future__.get_swap_module_params_on_conversion()
    module = torch.nn.Linear(2, 2).to("npu")
    assert next(module.parameters()).device.type == "npu"
    assert isinstance(original, bool)

    try:
        torch.__future__.set_swap_module_params_on_conversion(not original)
        assert torch.__future__.get_swap_module_params_on_conversion() is (not original)
        torch.__future__.set_swap_module_params_on_conversion(original)
        assert torch.__future__.get_swap_module_params_on_conversion() is original
    finally:
        torch.__future__.set_swap_module_params_on_conversion(original)
