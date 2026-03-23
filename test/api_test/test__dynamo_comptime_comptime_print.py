"""
测试目的：验证 `torch._dynamo.comptime.comptime.print` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch._dynamo.comptime.comptime.print`
覆盖的入参维度：
- 参数传参与不传参：覆盖单参数正常调用和缺参异常。
- 参数为 None / 非 None：覆盖字符串与 NPU Tensor。
- 枚举/多类型：覆盖字符串/Tensor 两类主要输入。
- 正常输入：覆盖 eager 模式 stdout 输出。
- 异常输入：覆盖缺少必填参数的 `TypeError`。
- 边界值和等价类：覆盖最小字符串。
未覆盖项及原因：
- 未覆盖 compile/export 上下文的更深层 comptime 语义。
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

def test_comptime_print_writes_eager_output():
    _require_npu()

    tensor = torch.ones(1, device="npu")
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = torch._dynamo.comptime.comptime.print(tensor)
        torch._dynamo.comptime.comptime.print("npu-eager")

    output = stream.getvalue()
    assert result is None
    assert "npu-eager" in output
    assert "device='npu:0'" in output or 'device="npu:0"' in output


def test_comptime_print_missing_argument_raises():
    _require_npu()

    with pytest.raises(TypeError):
        torch._dynamo.comptime.comptime.print()
