"""
测试目的：验证 `torch.dtype` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.dtype`
覆盖的入参维度：
- 参数传参与不传参：覆盖 dtype 属性读取和空构造异常。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖 `torch.float16` 与 `torch.int32`。
- 正常输入：覆盖 NPU Tensor 上的 dtype 对象。
- 异常输入：覆盖直接构造 `torch.dtype()`。
- 边界值和等价类：覆盖不同主类型。
未覆盖项及原因：
- 该 API 是类型对象而非普通函数，未覆盖更深层序列化语义。
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

@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
def test_dtype_objects_from_npu_tensors(dtype):
    _require_npu()

    tensor = torch.ones(2, device="npu", dtype=dtype)
    assert isinstance(tensor.dtype, torch.dtype)
    assert tensor.device.type == "npu"
    assert tensor.dtype is dtype


def test_dtype_constructor_raises():
    _require_npu()

    with pytest.raises(TypeError):
        torch.dtype()
