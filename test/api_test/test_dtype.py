# Test purpose: verify torch.dtype object semantics and NPU tensor integration,
# including dtype identity, aliasing, hashability, serialization, and failure
# cases for invalid dtype construction/call patterns.
# API name: torch.dtype
#
# Covered parameter dimensions:
# | Dimension | Coverage | Notes |
# | --- | --- | --- |
# | 参数传参与不传参 | 不适用 | torch.dtype 是类型对象，不接收函数参数 |
# | None / 非 None | 不适用 | 该 API 本身不接收 None 入参 |
# | 枚举选项 | 覆盖 | 覆盖 float32/float16/bfloat16/int64/bool 及别名 |
# | 多类型 | 覆盖 | 覆盖浮点、整数、布尔 dtype 与 dtype 对象本身 |
# | 正常输入 | 覆盖 | NPU Tensor 按不同 dtype 构造并读取 dtype |
# | 异常输入 | 覆盖 | torch.dtype() 与 dtype 对象调用均触发 TypeError |
# | 边界值和等价类 | 覆盖 | bool、bfloat16、空/小张量、dtype 别名等价类 |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | 数值精度校验 | intentionally omitted; this file focuses on dtype identity and interface behavior |
# | None 相关分支 | not applicable; torch.dtype is not a function and does not accept parameters |

import pickle

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


def test_dtype_identity_aliases_and_npu_tensor_construction() -> None:
    _require_npu()

    alias_pairs = [
        (torch.float, torch.float32),
        (torch.half, torch.float16),
        (torch.long, torch.int64),
        (torch.double, torch.float64),
    ]
    for alias, canonical in alias_pairs:
        assert isinstance(alias, torch.dtype)
        assert alias is canonical
        assert alias == canonical

    samples = [
        (torch.float32, (2,)),
        (torch.float16, (0,)),
        (torch.bfloat16, (1,)),
        (torch.int64, (3,)),
        (torch.bool, (4,)),
    ]
    for dtype, shape in samples:
        tensor = torch.ones(shape, device=_npu_device(), dtype=dtype)
        assert isinstance(dtype, torch.dtype)
        assert tensor.device.type == "npu"
        assert tensor.dtype is dtype
        assert tensor.dtype == dtype
        assert str(dtype) == repr(dtype)


def test_dtype_hash_and_pickle_roundtrip() -> None:
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool]

    dtype_set = set(dtypes)
    assert len(dtype_set) == len(dtypes)

    for dtype in dtypes:
        restored = pickle.loads(pickle.dumps(dtype))
        assert restored is dtype or restored == dtype
        assert hash(restored) == hash(dtype)


def test_dtype_invalid_construction_and_call_patterns() -> None:
    with pytest.raises(TypeError, match="cannot create 'torch.dtype' instances"):
        torch.dtype()

    with pytest.raises(TypeError, match="not callable"):
        torch.float32()

    with pytest.raises(TypeError, match="not callable"):
        torch.int64()
