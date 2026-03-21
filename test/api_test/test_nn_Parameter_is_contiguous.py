# 测试目的: 验证 torch.nn.Parameter.is_contiguous 在 NPU 环境下的默认连续性、非连续性、memory_format 分支和异常行为。
# API 名称: torch.nn.Parameter.is_contiguous
# 覆盖维度表:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖不传参数与传入 memory_format |
# | None / 非 None | 不适用 | 该 API 不以 None 作为自然分支输入 |
# | 枚举选项 | 覆盖 | 覆盖默认 contiguous_format 与 channels_last |
# | 多类型 | 覆盖 | 覆盖普通 Parameter、非连续 Parameter、空 Parameter |
# | 正常输入 | 覆盖 | NPU 上的连续、非连续、channels_last 参数 |
# | 异常输入 | 覆盖 | 非法 memory_format 入参 |
# | 边界值和等价类 | 覆盖 | 空 Parameter、1D/2D/4D 典型等价类 |
# 未覆盖项及原因:
# - None / 非 None: 该 API 没有自然的 None 参数分支，默认态已由不传参覆盖。
# - channels_last_3d: 为保持用例稳定与最小化，优先覆盖更常见的 channels_last 维度。

import pytest
import torch
import torch.nn as nn

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


def test_parameter_is_contiguous_default_and_boundary_on_npu() -> None:
    _require_npu()

    contiguous = nn.Parameter(torch.ones((2, 3), device=_npu_device(), dtype=torch.float32))
    assert contiguous.is_contiguous() is True
    assert contiguous.device.type == "npu"

    empty = nn.Parameter(torch.empty(0, device=_npu_device(), dtype=torch.float32))
    assert empty.is_contiguous() is True
    assert empty.numel() == 0
    assert empty.device.type == "npu"

    non_contiguous = nn.Parameter(
        torch.ones((2, 3), device=_npu_device(), dtype=torch.float32).transpose(0, 1)
    )
    assert non_contiguous.is_contiguous() is False
    assert non_contiguous.shape == torch.Size([3, 2])
    assert non_contiguous.device.type == "npu"


def test_parameter_is_contiguous_memory_format_on_npu() -> None:
    _require_npu()

    base = torch.empty((1, 2, 3, 4), device=_npu_device(), dtype=torch.float32)
    try:
        channels_last_tensor = base.contiguous(memory_format=torch.channels_last)
    except Exception as exc:
        pytest.skip(f"channels_last layout is not available/reliable on this NPU build: {exc}")

    param = nn.Parameter(channels_last_tensor)

    assert param.device.type == "npu"
    assert param.is_contiguous() is False
    assert param.is_contiguous(memory_format=torch.channels_last) is True


def test_parameter_is_contiguous_rejects_invalid_memory_format_on_npu() -> None:
    _require_npu()

    param = nn.Parameter(torch.ones((2, 3), device=_npu_device(), dtype=torch.float32))

    with pytest.raises(TypeError):
        param.is_contiguous(memory_format="invalid")  # type: ignore[arg-type]
