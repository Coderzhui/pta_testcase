# 测试目的: 验证 torch.utils._python_dispatch.is_traceable_wrapper_subclass 在 NPU 环境下对可追踪 wrapper subclass 的识别、边界与负例行为。
# API 名称: torch.utils._python_dispatch.is_traceable_wrapper_subclass
# 覆盖维度表:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 不适用 | 该 API 是单参数 predicate |
# | None / 非 None | 覆盖 | 覆盖普通 Tensor/对象与 traceable wrapper subclass |
# | 枚举选项 | 不适用 | 该 API 无枚举参数 |
# | 多类型 | 覆盖 | 覆盖 plain Tensor、Parameter、traceable wrapper subclass、non-traceable wrapper subclass |
# | 正常输入 | 覆盖 | NPU-backed traceable wrapper subclass 识别为 True |
# | 异常输入 | 覆盖 | companion type helper 的非法类型输入触发 TypeError |
# | 边界值和等价类 | 覆盖 | 空 Tensor wrapper、单元素 Tensor wrapper、普通 Tensor 等价类 |
# 未覆盖项及原因:
# - 参数传参与不传参: 该 API 只接收一个对象参数，不存在可选参数组合。
# - 枚举选项: 该 API 无枚举参数。
# - 目标函数的异常分支: is_traceable_wrapper_subclass 本身是纯布尔谓词，没有可靠的直接异常路径；异常覆盖通过相关的 type helper 完成。

import pytest
import torch
from torch.utils import _python_dispatch as python_dispatch

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


class TraceableWrapper(torch.Tensor):
    def __new__(cls, inner: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            inner.size(),
            strides=inner.stride(),
            storage_offset=inner.storage_offset(),
            dtype=inner.dtype,
            layout=inner.layout,
            requires_grad=inner.requires_grad,
            device=inner.device,
        )

    def __init__(self, inner: torch.Tensor):
        self._inner = inner

    def __tensor_flatten__(self):
        return ["_inner"], {"device": str(self._inner.device)}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        inner = inner_tensors["_inner"]
        return TraceableWrapper(inner)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise AssertionError("TraceableWrapper.__torch_dispatch__ is not exercised in this test")


class NonTraceableWrapper(torch.Tensor):
    def __new__(cls, inner: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            inner.size(),
            strides=inner.stride(),
            storage_offset=inner.storage_offset(),
            dtype=inner.dtype,
            layout=inner.layout,
            requires_grad=inner.requires_grad,
            device=inner.device,
        )

    def __init__(self, inner: torch.Tensor):
        self._inner = inner

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise AssertionError("NonTraceableWrapper.__torch_dispatch__ is not exercised in this test")


def test_is_traceable_wrapper_subclass_positive_and_negative_on_npu() -> None:
    _require_npu()

    traceable = TraceableWrapper(torch.ones(1, device=_npu_device(), dtype=torch.float32))
    empty_traceable = TraceableWrapper(torch.empty(0, device=_npu_device(), dtype=torch.float32))
    non_traceable = NonTraceableWrapper(torch.ones(2, device=_npu_device(), dtype=torch.float32))

    assert traceable.device.type == "npu"
    assert empty_traceable.device.type == "npu"
    assert non_traceable.device.type == "npu"

    assert python_dispatch.is_traceable_wrapper_subclass(traceable) is True
    assert python_dispatch.is_traceable_wrapper_subclass(empty_traceable) is True
    assert python_dispatch.is_traceable_wrapper_subclass(non_traceable) is False
    assert python_dispatch.is_traceable_wrapper_subclass(torch.ones(1, device=_npu_device())) is False
    assert python_dispatch.is_traceable_wrapper_subclass(torch.nn.Parameter(torch.ones(1, device=_npu_device()))) is False


def test_is_traceable_wrapper_subclass_type_and_invalid_input_on_npu() -> None:
    _require_npu()

    traceable_type = type(TraceableWrapper(torch.ones(1, device=_npu_device(), dtype=torch.float32)))
    non_traceable_type = type(NonTraceableWrapper(torch.ones(1, device=_npu_device(), dtype=torch.float32)))

    assert python_dispatch.is_traceable_wrapper_subclass_type(traceable_type) is True
    assert python_dispatch.is_traceable_wrapper_subclass_type(non_traceable_type) is False
    assert python_dispatch.is_traceable_wrapper_subclass_type(torch.Tensor) is False

    with pytest.raises(TypeError):
        python_dispatch.is_traceable_wrapper_subclass_type(123)  # type: ignore[arg-type]
