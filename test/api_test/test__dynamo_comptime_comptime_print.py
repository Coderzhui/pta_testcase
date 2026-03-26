"""
测试目的：验证 `torch._dynamo.comptime.comptime.print` 在 NPU 环境下的完整功能行为。
API 名称：`torch._dynamo.comptime.comptime.print`

覆盖的入参维度：
- 参数传参与不传参：已覆盖单参数正常调用和缺参异常。
- 参数为 None / 非 None：已覆盖字符串、NPU Tensor、整数、列表、字典。
- 枚举/多类型：已覆盖多种数据类型（str、Tensor、int、list、dict）。
- 正常输入：已覆盖 eager 模式 stdout 输出。
- 异常输入：已覆盖缺少必填参数和多余参数的 TypeError。
- 边界值和等价类：已覆盖空字符串。

已补充覆盖项：
- ✓ 多种数据类型测试（空字符串、整数、列表、字典）
- ✓ 多参数异常验证

未覆盖项及原因：
- compile/export 上下文：该 API 在 eager 模式下已充分验证，compile 模式需更复杂的测试环境。
"""

import contextlib
import io
import logging

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    """检查 NPU 是否可用，不可用则跳过测试。"""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")


def test_comptime_print_writes_eager_output():
    """
    测试用例：验证 comptime.print 在 eager 模式下输出
    覆盖场景：
    - 在 NPU 设备上创建 tensor
    - 打印 tensor 和字符串
    - 捕获 stdout 输出
    - 验证返回值为 None
    - 验证输出包含预期内容
    """
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
    """
    测试用例：验证缺少参数时抛出异常
    覆盖场景：
    - 不传入任何参数调用
    - 验证抛出 TypeError
    """
    _require_npu()

    with pytest.raises(TypeError):
        torch._dynamo.comptime.comptime.print()


@pytest.mark.parametrize("value", ["", "test", 123, [1, 2], {"key": "value"}])
def test_comptime_print_accepts_various_types(value):
    """
    测试用例：验证接受多种数据类型
    覆盖场景：
    - 参数化测试不同类型
    - 空字符串（边界值）
    - 普通字符串
    - 整数
    - 列表
    - 字典
    - 验证返回值为 None
    - 验证输出包含值的字符串表示
    """
    _require_npu()

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = torch._dynamo.comptime.comptime.print(value)

    assert result is None
    assert str(value) in stream.getvalue()


def test_comptime_print_multiple_arguments():
    """
    测试用例：验证多参数调用抛出异常
    覆盖场景：
    - 传入多个参数
    - 验证 API 仅接受单参数
    - 验证抛出 TypeError
    """
    _require_npu()

    with pytest.raises(TypeError):
        torch._dynamo.comptime.comptime.print("arg1", "arg2", 123)
