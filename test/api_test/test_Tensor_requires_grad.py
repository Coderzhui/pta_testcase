# 测试目的: 验证 Tensor.requires_grad 在 NPU Tensor 上的读取、赋值和异常行为。
# API 名称: torch.Tensor.requires_grad
# 覆盖维度表:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 不适用 | requires_grad 是属性，不接收函数参数 |
# | None / 非 None | 不适用 | 属性赋值只接受布尔值，不存在 None 入参 |
# | 枚举选项 | 覆盖 | 通过 False -> True -> False 覆盖布尔两态 |
# | 多类型 | 覆盖 | 覆盖 float Tensor 正常场景与 int Tensor 异常场景 |
# | 正常输入 | 覆盖 | leaf NPU Tensor、空 Tensor |
# | 异常输入 | 覆盖 | non-leaf Tensor、整型 Tensor 赋值 True |
# | 边界值和等价类 | 覆盖 | 空 Tensor、重复赋值、leaf/non-leaf 等价类 |
# 未覆盖项及原因:
# - 参数传参与不传参: 不适用，Tensor.requires_grad 是属性，不是可调用 API。
# - None / 非 None: 不适用，属性赋值不接受 None。

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - import-time environment guard
    torch_npu = None  # type: ignore[assignment]
    _TORCH_NPU_IMPORT_ERROR = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None


def _require_npu() -> None:
    if _TORCH_NPU_IMPORT_ERROR is not None:
        pytest.skip(f"torch_npu import failed: {_TORCH_NPU_IMPORT_ERROR}")
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu backend is unavailable in this environment")
    if not torch.npu.is_available():
        pytest.skip("NPU device is unavailable in this environment")


def _npu_device() -> torch.device:
    _require_npu()
    return torch.device("npu:0")


def test_requires_grad_leaf_roundtrip_on_npu() -> None:
    _require_npu()
    tensor = torch.ones(4, device=_npu_device(), dtype=torch.float32)

    assert tensor.device.type == "npu"
    assert tensor.requires_grad is False
    assert tensor.is_leaf is True

    tensor.requires_grad = True
    assert tensor.requires_grad is True
    assert tensor.device.type == "npu"
    assert tensor.is_leaf is True

    tensor.requires_grad = False
    assert tensor.requires_grad is False
    assert tensor.device.type == "npu"


def test_requires_grad_empty_tensor_boundary_on_npu() -> None:
    _require_npu()
    tensor = torch.empty(0, device=_npu_device(), dtype=torch.float32)

    assert tensor.numel() == 0
    assert tensor.requires_grad is False

    tensor.requires_grad = True
    assert tensor.requires_grad is True
    assert tensor.device.type == "npu"
    assert tensor.numel() == 0


def test_requires_grad_assignment_errors_on_npu() -> None:
    _require_npu()

    base = torch.arange(4, device=_npu_device(), dtype=torch.float32)
    base.requires_grad = True
    non_leaf = base + 1
    assert non_leaf.is_leaf is False

    with pytest.raises(RuntimeError):
        non_leaf.requires_grad = False

    int_tensor = torch.ones(2, device=_npu_device(), dtype=torch.int32)
    assert int_tensor.requires_grad is False

    with pytest.raises(RuntimeError):
        int_tensor.requires_grad = True
