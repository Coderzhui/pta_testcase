"""
测试目的：验证 `torch.nn.Module.named_modules` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Module.named_modules`
覆盖的入参维度：
- 参数传参与不传参：覆盖默认参数、显式 `prefix` 和 `remove_duplicate=True/False`。
- 参数为 None / 非 None：覆盖 `memo=None` 默认路径。
- 枚举/多类型：覆盖根模块与子模块。
- 正常输入：覆盖嵌套模块名称枚举。
- 异常输入：覆盖错误 self 类型。
- 边界值和等价类：覆盖根名称空字符串。
未覆盖项及原因：
- 未覆盖共享模块的 `remove_duplicate=False` 场景。
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

class _NamedModulesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU()).to("npu")


def test_module_named_modules_with_prefix():
    _require_npu()

    module = _NamedModulesModule()
    pairs = list(module.named_modules(prefix="root"))
    names = [name for name, _ in pairs]

    assert "root" in names
    assert "root.block" in names
    assert any(name.endswith("0") for name in names)


def test_module_named_modules_invalid_self_raises():
    _require_npu()

    with pytest.raises(AttributeError):
        list(torch.nn.Module.named_modules("bad"))


class _SharedNamedModulesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shared = torch.nn.Linear(1, 1).to("npu")
        self.first = shared
        self.second = shared


def test_module_named_modules_remove_duplicate_modes():
    _require_npu()

    module = _SharedNamedModulesModule()
    deduped = [name for name, _ in module.named_modules(remove_duplicate=True)]
    repeated = [name for name, _ in module.named_modules(remove_duplicate=False)]

    assert deduped == ["", "first"]
    assert repeated == ["", "first", "second"]
