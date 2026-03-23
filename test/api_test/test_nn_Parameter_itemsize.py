"""
测试目的：验证 `torch.nn.Parameter.itemsize` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Parameter.itemsize`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为属性访问。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖 `float16` 与 `float32` itemsize。
- 正常输入：覆盖 NPU Parameter itemsize 读取。
- 异常输入：覆盖只读属性非法赋值。
- 边界值和等价类：覆盖两种主流字节宽度。
未覆盖项及原因：
- 未覆盖 bfloat16/complex 类型。
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

@pytest.mark.parametrize("dtype, expected", [(torch.float16, 2), (torch.float32, 4)])
def test_parameter_itemsize_on_npu(dtype, expected):
    _require_npu()

    parameter = torch.nn.Parameter(torch.ones(2, device="npu", dtype=dtype))
    assert parameter.itemsize == expected


def test_parameter_itemsize_read_only_assignment_raises():
    _require_npu()

    parameter = torch.nn.Parameter(torch.ones(1, device="npu"))
    with pytest.raises(AttributeError):
        parameter.itemsize = 8
