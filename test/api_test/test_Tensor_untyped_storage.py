"""
测试目的：验证 `Tensor.untyped_storage` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`Tensor.untyped_storage`
覆盖的入参维度：
- 参数传参与不传参：覆盖实例调用和 unbound 错误调用。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖 float32 与 int32 Tensor。
- 正常输入：覆盖 NPU Tensor 的 storage 访问。
- 异常输入：覆盖错误 self 类型。
- 边界值和等价类：覆盖空张量和普通张量。
未覆盖项及原因：
- 未覆盖更底层 storage 复用语义，因为该场景超出接口级功能验证范围。
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

@pytest.mark.parametrize("shape, dtype", [((2,), torch.float32), ((0,), torch.int32)])
def test_tensor_untyped_storage_returns_storage_on_npu(shape, dtype):
    _require_npu()

    tensor = torch.empty(shape, device="npu", dtype=dtype)
    if tensor.numel() > 0:
        tensor.fill_(1)
    storage = tensor.untyped_storage()

    assert isinstance(storage, torch.UntypedStorage)
    assert str(storage.device).startswith("npu")
    assert len(storage) >= 0


def test_tensor_untyped_storage_invalid_self_raises():
    _require_npu()

    with pytest.raises(TypeError):
        torch.Tensor.untyped_storage("bad")
