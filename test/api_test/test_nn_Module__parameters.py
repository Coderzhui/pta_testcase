"""
测试目的：验证 `torch.nn.Module._parameters` 在 NPU Parameter 注册后会反映内部参数字典状态。
API 名称：`torch.nn.Module._parameters`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为内部属性访问。
- 参数为 None / 非 None：覆盖空字典和含 Parameter 的字典。
- 枚举/多类型：覆盖顶层 Parameter 存取。
- 正常输入：覆盖 NPU Parameter 注册后内部字典。
- 异常输入：覆盖缺失键访问。
- 边界值和等价类：覆盖空模块和单参数模块。
未覆盖项及原因：
- 该属性属于内部结构，不覆盖手动篡改字典后的未定义行为。
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

def test_module_parameters_internal_dict_tracks_npu_parameter():
    _require_npu()

    module = torch.nn.Module()
    assert module._parameters == {}

    module.weight = torch.nn.Parameter(torch.ones(1, device="npu"))
    assert "weight" in module._parameters
    assert module._parameters["weight"].device.type == "npu"


def test_module_parameters_missing_key_raises():
    _require_npu()

    with pytest.raises(KeyError):
        _ = torch.nn.Module()._parameters["missing"]
