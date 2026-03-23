"""
测试目的：验证 `torch.compiler.is_compiling` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.compiler.is_compiling`
覆盖的入参维度：
- 参数传参与不传参：覆盖无参正常调用和多余参数异常调用。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖布尔返回值。
- 正常输入：覆盖 eager NPU 场景。
- 异常输入：覆盖多余参数。
- 边界值和等价类：覆盖 eager 模式的 `False`。
未覆盖项及原因：
- 未覆盖 `torch.compile` 真实编译态；该阶段只生成静态最小接口测试。
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

def test_compiler_is_compiling_returns_bool_in_eager_mode():
    _require_npu()

    tensor = torch.ones(1, device="npu")
    assert tensor.device.type == "npu"
    assert torch.compiler.is_compiling() is False


def test_compiler_is_compiling_rejects_extra_args():
    _require_npu()

    with pytest.raises(TypeError):
        torch.compiler.is_compiling("bad")
