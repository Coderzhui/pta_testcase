"""
测试目的：验证 `torch.nn.Module.buffers` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Module.buffers`
覆盖的入参维度：
- 参数传参与不传参：覆盖 `recurse=True/False`。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖父模块 buffer 与子模块 buffer。
- 正常输入：覆盖 NPU buffer 枚举。
- 异常输入：覆盖错误 self 类型。
- 边界值和等价类：覆盖仅父层与递归层级。
未覆盖项及原因：
- 未覆盖持久化 buffer 标志位差异，当前接口测试重点是迭代行为。
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

class _BuffersModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("root_buf", torch.ones(1, device="npu"))
        self.child = torch.nn.Module()
        self.child.register_buffer("child_buf", torch.zeros(2, device="npu"))


def test_module_buffers_recurse_modes():
    _require_npu()

    module = _BuffersModule()
    shallow = list(module.buffers(recurse=False))
    deep = list(module.buffers(recurse=True))

    assert len(shallow) == 1
    assert len(deep) == 2
    assert all(buf.device.type == "npu" for buf in deep)


def test_module_buffers_invalid_self_raises():
    _require_npu()

    with pytest.raises(AttributeError):
        list(torch.nn.Module.buffers("bad"))
