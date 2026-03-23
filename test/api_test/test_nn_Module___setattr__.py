"""
测试目的：验证 `torch.nn.Module.__setattr__` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Module.__setattr__`
覆盖的入参维度：
- 参数传参与不传参：覆盖 Parameter/Module 赋值和非法替换。
- 参数为 None / 非 None：覆盖合法对象与非法字符串。
- 枚举/多类型：覆盖 `Parameter` 和 `Module`。
- 正常输入：覆盖自动注册 `_parameters` / `_modules`。
- 异常输入：覆盖参数槽位非法赋值。
- 边界值和等价类：覆盖最小自定义模块。
未覆盖项及原因：
- 未覆盖 buffer 自动注册，因为该路径推荐使用 `register_buffer`。
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

def test_module_setattr_registers_parameter_and_child_module():
    _require_npu()

    module = torch.nn.Module()
    module.param = torch.nn.Parameter(torch.ones(1, device="npu"))
    module.child = torch.nn.Linear(1, 1).to("npu")

    assert "param" in module._parameters
    assert module._parameters["param"].device.type == "npu"
    assert "child" in module._modules


def test_module_setattr_invalid_parameter_replacement_raises():
    _require_npu()

    module = torch.nn.Module()
    module.param = torch.nn.Parameter(torch.ones(1, device="npu"))
    with pytest.raises(TypeError):
        module.param = "bad"
