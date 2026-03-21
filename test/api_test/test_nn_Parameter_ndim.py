# Test purpose: verify torch.nn.Parameter.ndim behavior on NPU-backed
# parameters, including normal access, boundary shapes, and read-only failure
# behavior.
# API name: torch.nn.Parameter.ndim
#
# Covered parameter dimensions:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 不适用 | ndim 是属性，不接收函数参数 |
# | None / 非 None | 不适用 | 属性访问不接收 None 入参 |
# | 枚举选项 | 不适用 | ndim 属性无枚举参数 |
# | 多类型 | 覆盖 | 覆盖 scalar、empty、1D、2D NPU Parameter 的维度读取 |
# | 正常输入 | 覆盖 | 正常 NPU Parameter 读取 ndim，且与 dim() 一致 |
# | 异常输入 | 覆盖 | ndim 属性赋值触发 AttributeError |
# | 边界值和等价类 | 覆盖 | scalar/empty/1D/2D 参数形状边界与只读边界 |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | 函数参数的 None/非 None / 枚举分支 | not applicable because ndim is a read-only property |
# | 数值精度校验 | intentionally omitted; this API exposes shape metadata only |

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
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


def test_parameter_ndim_reports_expected_rank_on_npu() -> None:
    _require_npu()

    scalar = torch.nn.Parameter(torch.tensor(3.0, device=_npu_device()))
    empty = torch.nn.Parameter(torch.empty(0, device=_npu_device()))
    vector = torch.nn.Parameter(torch.ones(2, device=_npu_device()))
    matrix = torch.nn.Parameter(torch.ones(2, 3, device=_npu_device()))

    assert scalar.ndim == 0
    assert scalar.ndim == scalar.dim()
    assert empty.ndim == 1
    assert empty.ndim == empty.dim()
    assert vector.ndim == 1
    assert vector.ndim == vector.dim()
    assert matrix.ndim == 2
    assert matrix.ndim == matrix.dim()

    assert scalar.device.type == "npu"
    assert empty.device.type == "npu"
    assert vector.device.type == "npu"
    assert matrix.device.type == "npu"

    class_attr = torch.nn.Parameter.ndim
    assert "ndim" in repr(class_attr)


def test_parameter_ndim_preserves_boundary_shapes_and_dtype_variants_on_npu() -> None:
    _require_npu()

    cases = [
        (torch.float32, torch.ones(1, device=_npu_device(), dtype=torch.float32), 1),
        (torch.float16, torch.ones(4, device=_npu_device(), dtype=torch.float16), 1),
        (torch.bfloat16, torch.ones(2, 2, device=_npu_device(), dtype=torch.bfloat16), 2),
    ]

    for dtype, tensor, expected_ndim in cases:
        param = torch.nn.Parameter(tensor)
        assert param.device.type == "npu"
        assert param.dtype is dtype
        assert param.ndim == expected_ndim
        assert param.ndim == param.dim()


def test_parameter_ndim_is_read_only() -> None:
    _require_npu()

    param = torch.nn.Parameter(torch.zeros(1, device=_npu_device()))

    with pytest.raises(AttributeError, match="not writable"):
        param.ndim = 3
