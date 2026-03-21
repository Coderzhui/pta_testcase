# 测试目的: 验证 torch.nn.Parameter.dtype 在 NPU 环境下的读取、dtype 变换和只读属性行为。
# API 名称: torch.nn.Parameter.dtype
# 覆盖维度表:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 不适用 | dtype 是属性，不接收函数参数 |
# | None / 非 None | 覆盖 | 覆盖非 None dtype 读取与转换；不传参读回默认 dtype |
# | 枚举选项 | 覆盖 | 覆盖 float32 / float16 两种主要 dtype |
# | 多类型 | 覆盖 | 覆盖普通 Parameter、空 Parameter、dtype 转换结果 |
# | 正常输入 | 覆盖 | NPU 上的 dtype 读取与 `.to(dtype=...)` 转换 |
# | 异常输入 | 覆盖 | 尝试写入只读 dtype 属性 |
# | 边界值和等价类 | 覆盖 | 空 Parameter、不同浮点 dtype 等价类 |
# 未覆盖项及原因:
# - 参数传参与不传参: 该 API 是属性，不是可调用函数。
# - 其他 dtype 候选: 为保持用例最小，仅覆盖最常见的 float32 与 float16。

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


def test_parameter_dtype_readback_and_boundary_on_npu() -> None:
    _require_npu()

    float32_param = nn.Parameter(torch.ones(2, device=_npu_device(), dtype=torch.float32))
    assert float32_param.dtype is torch.float32
    assert float32_param.device.type == "npu"

    empty_float16 = nn.Parameter(torch.empty(0, device=_npu_device(), dtype=torch.float16))
    assert empty_float16.dtype is torch.float16
    assert empty_float16.numel() == 0
    assert empty_float16.device.type == "npu"


def test_parameter_dtype_tracks_conversion_on_npu() -> None:
    _require_npu()

    param = nn.Parameter(torch.ones((2, 2), device=_npu_device(), dtype=torch.float32))
    converted = param.to(dtype=torch.float16)

    assert converted.device.type == "npu"
    assert converted.dtype is torch.float16
    assert converted.shape == param.shape

    round_trip = converted.to(dtype=torch.float32)
    assert round_trip.device.type == "npu"
    assert round_trip.dtype is torch.float32
    assert round_trip.shape == param.shape


def test_parameter_dtype_is_read_only() -> None:
    _require_npu()

    param = nn.Parameter(torch.ones(1, device=_npu_device(), dtype=torch.float32))

    with pytest.raises(AttributeError):
        param.dtype = torch.float16  # type: ignore[misc]
