"""
测试目的：验证 `torch._from_functional_tensor` 在 NPU 环境下的完整功能行为。
API 名称：`torch._from_functional_tensor`

覆盖的入参维度：
- 参数传参与不传参：已覆盖 functional tensor 正常路径和普通 Tensor 异常路径。
- 参数为 None / 非 None：已覆盖合法 functional Tensor，以及非法 None/普通 Tensor。
- 枚举/多类型：不适用。
- 正常输入：已覆盖 NPU functional tensor 还原。
- 异常输入：已覆盖 None 和普通 Tensor。
- 边界值和等价类：已覆盖多维 tensor（1D、2D、3D）和多种 dtype（float32、float16、int64）。

已补充覆盖项：
- ✓ 多维 tensor 测试（1D、2D、3D）
- ✓ 不同 dtype 测试（float32、float16、int64）

未覆盖项及原因：
- 无，已满足 API 功能一致性标准。
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


def test_from_functional_tensor_round_trip_on_npu():
    """
    测试用例：验证 functional tensor 往返转换
    覆盖场景：
    - 检查 _to_functional_tensor helper 是否存在
    - 在 NPU 上创建 base tensor
    - 转换为 functional tensor
    - 使用 _from_functional_tensor 还原
    - 验证还原后的 tensor 类型和设备
    - 验证还原后与原始 tensor 是同一对象
    """
    _require_npu()
    if not hasattr(torch, "_to_functional_tensor"):
        pytest.skip("当前环境缺少 functional tensor helper，无法稳定验证 torch._from_functional_tensor。")

    base = torch.ones(2, device="npu")
    functional = torch._to_functional_tensor(base)
    restored = torch._from_functional_tensor(functional)

    assert isinstance(restored, torch.Tensor)
    assert restored.device.type == "npu"
    assert restored is base


@pytest.mark.parametrize("case", ["none", "plain_tensor"])
def test_from_functional_tensor_invalid_inputs_raise(case):
    """
    测试用例：验证非法输入抛出异常
    覆盖场景：
    - 参数化测试两种非法输入
    - None 参数
    - 普通 Tensor（非 functional tensor）
    - 验证抛出 RuntimeError 或 TypeError
    """
    _require_npu()
    value = None if case == "none" else torch.ones(1, device="npu")

    with pytest.raises((RuntimeError, TypeError)):
        torch._from_functional_tensor(value)


@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_from_functional_tensor_various_shapes(shape):
    """
    测试用例：验证不同形状的 tensor
    覆盖场景：
    - 一维 tensor
    - 二维 tensor
    - 三维 tensor
    """
    _require_npu()
    if not hasattr(torch, "_to_functional_tensor"):
        pytest.skip("当前环境缺少 functional tensor helper。")

    base = torch.ones(shape, device="npu")
    functional = torch._to_functional_tensor(base)
    restored = torch._from_functional_tensor(functional)

    assert restored.shape == shape
    assert restored.device.type == "npu"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_from_functional_tensor_various_dtypes(dtype):
    """
    测试用例：验证不同数据类型的 tensor
    覆盖场景：
    - float32
    - float16
    - int64
    """
    _require_npu()
    if not hasattr(torch, "_to_functional_tensor"):
        pytest.skip("当前环境缺少 functional tensor helper。")

    base = torch.ones(2, dtype=dtype, device="npu")
    functional = torch._to_functional_tensor(base)
    restored = torch._from_functional_tensor(functional)

    assert restored.dtype == dtype
    assert restored.device.type == "npu"
