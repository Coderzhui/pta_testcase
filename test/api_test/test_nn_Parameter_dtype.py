"""
测试目的：验证 `torch.nn.Parameter.dtype` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Parameter.dtype`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为属性访问。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖 `float16` 与 `float32`。
- 正常输入：覆盖 NPU Parameter dtype 读取。
- 异常输入：覆盖只读属性非法赋值。
- 边界值和等价类：覆盖主流浮点类型。
未覆盖项及原因：
- 未覆盖整型 Parameter，因为当前用例已覆盖 dtype 属性语义。
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

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_parameter_dtype_on_npu(dtype):
    _require_npu()

    parameter = torch.nn.Parameter(torch.ones(2, device="npu", dtype=dtype))
    assert parameter.dtype is dtype


def test_parameter_dtype_read_only_assignment_raises():
    _require_npu()

    parameter = torch.nn.Parameter(torch.ones(1, device="npu"))
    with pytest.raises(AttributeError):
        parameter.dtype = torch.float16
