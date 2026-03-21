# Test purpose: verify torch.nn.Parameter.device behavior on NPU-backed
# parameters, including normal access, boundary cases, and read-only failure
# behavior.
# API name: torch.nn.Parameter.device
#
# Covered parameter dimensions:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 不适用 | device 是属性，不接收函数参数 |
# | None / 非 None | 不适用 | 属性访问不接收 None 入参 |
# | 枚举选项 | 不适用 | device 属性无枚举参数 |
# | 多类型 | 覆盖 | 覆盖 float / float16 / empty NPU Parameter 的 dtype 与设备读取 |
# | 正常输入 | 覆盖 | NPU Parameter 正常读取 device、is_leaf、requires_grad |
# | 异常输入 | 覆盖 | device 属性赋值触发 AttributeError |
# | 边界值和等价类 | 覆盖 | 空 Parameter、不同 dtype Parameter、属性只读边界 |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | 函数参数的 None/非 None / 枚举分支 | not applicable because device is a read-only property |
# | 数值精度校验 | intentionally omitted; this API exposes device metadata only |

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


def test_parameter_device_reports_npu_device_on_normal_and_boundary_values() -> None:
    _require_npu()

    normal = torch.nn.Parameter(torch.ones(2, device=_npu_device(), dtype=torch.float32))
    empty = torch.nn.Parameter(torch.empty(0, device=_npu_device(), dtype=torch.float16))

    assert isinstance(normal.device, torch.device)
    assert normal.device.type == "npu"
    assert normal.device == torch.device("npu:0")
    assert normal.requires_grad is True
    assert normal.is_leaf is True

    assert isinstance(empty.device, torch.device)
    assert empty.device.type == "npu"
    assert empty.device == torch.device("npu:0")
    assert empty.numel() == 0
    assert empty.shape == torch.Size([0])

    class_attr = torch.nn.Parameter.device
    assert "device" in repr(class_attr)


def test_parameter_device_preserves_dtype_and_device_for_multiple_types() -> None:
    _require_npu()

    cases = [
        (torch.float32, 1),
        (torch.float16, 2),
        (torch.bfloat16, 3),
    ]
    for dtype, size in cases:
        param = torch.nn.Parameter(torch.arange(size, device=_npu_device(), dtype=dtype))
        assert param.device.type == "npu"
        assert param.dtype is dtype
        assert param.numel() == size


def test_parameter_device_is_read_only() -> None:
    _require_npu()

    param = torch.nn.Parameter(torch.zeros(1, device=_npu_device()))

    with pytest.raises(AttributeError, match="not writable"):
        param.device = torch.device("cpu")
