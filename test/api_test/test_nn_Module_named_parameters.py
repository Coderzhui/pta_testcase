"""
测试目的：验证 `torch.nn.Module.named_parameters` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Module.named_parameters`
覆盖的入参维度：
- 参数传参与不传参：覆盖 `prefix`、`recurse=True/False` 和 `remove_duplicate=True/False`。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖权重、偏置和自定义 Parameter。
- 正常输入：覆盖 NPU Parameter 名称枚举。
- 异常输入：覆盖错误 self 类型。
- 边界值和等价类：覆盖递归关闭仅返回顶层参数。
未覆盖项及原因：
- 未覆盖共享 Parameter 的去重策略。
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

class _NamedParametersModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.top = torch.nn.Parameter(torch.ones(1, device="npu"))
        self.child = torch.nn.Linear(2, 2).to("npu")


def test_module_named_parameters_recurse_and_prefix():
    _require_npu()

    module = _NamedParametersModule()
    shallow = dict(module.named_parameters(recurse=False))
    deep = dict(module.named_parameters(prefix="root"))

    assert list(shallow) == ["top"]
    assert "root.top" in deep
    assert "root.child.weight" in deep
    assert all(param.device.type == "npu" for param in deep.values())


def test_module_named_parameters_invalid_self_raises():
    _require_npu()

    with pytest.raises(AttributeError, match="_named_members"):
        list(torch.nn.Module.named_parameters("bad"))


class _SharedNamedParametersModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shared = torch.nn.Parameter(torch.ones(1, device="npu"))
        self.top = shared
        self.alias = shared


def test_module_named_parameters_remove_duplicate_modes():
    _require_npu()

    module = _SharedNamedParametersModule()
    deduped = [name for name, _ in module.named_parameters(remove_duplicate=True)]
    repeated = [name for name, _ in module.named_parameters(remove_duplicate=False)]

    assert deduped == ["top"]
    assert repeated == ["top", "alias"]
