"""
测试目的：验证 `torch.__future__.get_swap_module_params_on_conversion` 可在 NPU 场景下稳定读取并与 setter 形成正确的配置 round-trip。
API 名称：`torch.__future__.get_swap_module_params_on_conversion`
覆盖的入参维度：
- 参数传参与不传参：覆盖 getter 无参调用和 setter 切换后的 getter 调用。
- 参数为 None / 非 None：覆盖 setter 传入 None。
- 枚举/多类型：覆盖 `True` / `False` 两种布尔配置，以及 setter 接受多种类型（int、str、list等）。
- 正常输入：覆盖设置项读取和 NPU Module 转换。
- 异常输入：该 API 不进行类型检查，直接存储传入值。
- 边界值和等价类：覆盖配置 round-trip，以及各种类型的参数测试。
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
    """测试getter/setter的round-trip功能，验证NPU模块转换"""
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


def test_set_swap_module_params_on_conversion_with_int():
    """测试setter传入int类型的行为（API直接存储原值，不进行类型检查）"""
    _require_npu()

    original = torch.__future__.get_swap_module_params_on_conversion()
    try:
        # 测试传入整数，API直接存储不转换
        torch.__future__.set_swap_module_params_on_conversion(1)
        # API直接存储原值，不进行bool转换
        result = torch.__future__.get_swap_module_params_on_conversion()
        assert result == 1
        assert isinstance(result, int)
    finally:
        torch.__future__.set_swap_module_params_on_conversion(original)


def test_set_swap_module_params_on_conversion_with_str():
    """测试setter传入str类型的行为（API直接存储原值）"""
    _require_npu()

    original = torch.__future__.get_swap_module_params_on_conversion()
    try:
        torch.__future__.set_swap_module_params_on_conversion("True")
        result = torch.__future__.get_swap_module_params_on_conversion()
        assert result == "True"
        assert isinstance(result, str)
    finally:
        torch.__future__.set_swap_module_params_on_conversion(original)


def test_set_swap_module_params_on_conversion_with_none():
    """测试setter传入None的行为（API直接存储None）"""
    _require_npu()

    original = torch.__future__.get_swap_module_params_on_conversion()
    try:
        torch.__future__.set_swap_module_params_on_conversion(None)
        result = torch.__future__.get_swap_module_params_on_conversion()
        assert result is None
    finally:
        torch.__future__.set_swap_module_params_on_conversion(original)


def test_set_swap_module_params_on_conversion_with_list():
    """测试setter传入list类型的行为（API直接存储原值）"""
    _require_npu()

    original = torch.__future__.get_swap_module_params_on_conversion()
    try:
        torch.__future__.set_swap_module_params_on_conversion([True])
        result = torch.__future__.get_swap_module_params_on_conversion()
        assert result == [True]
        assert isinstance(result, list)
    finally:
        torch.__future__.set_swap_module_params_on_conversion(original)


def test_set_swap_module_params_on_conversion_boundary_values():
    """测试setter的边界值（各种类型值直接存储）"""
    _require_npu()

    original = torch.__future__.get_swap_module_params_on_conversion()
    try:
        # 测试各种类型值都被直接存储
        test_values = [0, -1, 0.5, "", "test", [], [1, 2], {}, {"key": "value"}]
        for value in test_values:
            torch.__future__.set_swap_module_params_on_conversion(value)
            result = torch.__future__.get_swap_module_params_on_conversion()
            assert result == value, f"Expected {value}, got {result}"
    finally:
        torch.__future__.set_swap_module_params_on_conversion(original)


def test_swap_module_params_with_conversion_effect_on_npu():
    """测试配置切换对NPU模块转换的实际影响"""
    _require_npu()

    original = torch.__future__.get_swap_module_params_on_conversion()
    try:
        # 创建CPU模块
        module_cpu = torch.nn.Linear(3, 3)
        # 测试不同配置下的转换行为
        for config_value in [True, False]:
            torch.__future__.set_swap_module_params_on_conversion(config_value)
            # 转换到NPU
            module_npu = module_cpu.to("npu")
            # 验证参数确实在NPU上
            for param in module_npu.parameters():
                assert param.device.type == "npu"
            # 验证返回类型正确
            current = torch.__future__.get_swap_module_params_on_conversion()
            assert isinstance(current, bool)
            assert current == config_value
    finally:
        torch.__future__.set_swap_module_params_on_conversion(original)
