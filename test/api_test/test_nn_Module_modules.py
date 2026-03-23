"""
测试目的：验证 `torch.nn.Module.modules` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Module.modules`
覆盖的入参维度：
- 参数传参与不传参：覆盖实例调用和错误 self 调用。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖父模块与子模块类型。
- 正常输入：覆盖嵌套模块迭代。
- 异常输入：覆盖错误 self 类型。
- 边界值和等价类：覆盖空前缀根模块。
未覆盖项及原因：
- 未覆盖去重策略差异，因为 `modules()` 默认会去重。
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

class _ModulesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Linear(2, 2).to("npu")
        self.seq = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(2, 2)).to("npu")


def test_module_modules_iterates_nested_modules():
    _require_npu()

    module = _ModulesModule()
    names = [type(item).__name__ for item in module.modules()]

    assert names[0] == "_ModulesModule"
    assert "Linear" in names
    assert "Sequential" in names


def test_module_modules_invalid_self_raises():
    _require_npu()

    with pytest.raises(AttributeError):
        list(torch.nn.Module.modules("bad"))
