"""
测试目的：验证 `torch.autograd.graph._MultiHandle` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.autograd.graph._MultiHandle`
覆盖的入参维度：
- 参数传参与不传参：覆盖合法 handles 元组和非法 `None`。
- 参数为 None / 非 None：覆盖有效 handle tuple 与 `None`。
- 枚举/多类型：覆盖多个 RemovableHandle。
- 正常输入：覆盖 remove 能力。
- 异常输入：覆盖非法入参。
- 边界值和等价类：覆盖 2 个 handle 的最小组合。
未覆盖项及原因：
- 未覆盖更长 handle 列表和重复 remove 的幂等性。
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

def test_multihandle_wraps_and_removes_handles():
    _require_npu()

    module_a = torch.nn.Linear(1, 1).to("npu")
    module_b = torch.nn.Linear(1, 1).to("npu")
    handle_a = module_a.register_forward_hook(lambda *args: None)
    handle_b = module_b.register_forward_hook(lambda *args: None)
    multi = torch.autograd.graph._MultiHandle((handle_a, handle_b))

    assert type(multi).__name__ == "_MultiHandle"
    assert hasattr(multi, "remove")
    multi.remove()


def test_multihandle_none_handles_raise_on_remove():
    _require_npu()

    multi = torch.autograd.graph._MultiHandle(None)

    with pytest.raises(TypeError):
        multi.remove()
