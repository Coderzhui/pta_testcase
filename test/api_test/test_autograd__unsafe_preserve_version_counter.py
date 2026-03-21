# Test purpose: verify torch.autograd._unsafe_preserve_version_counter behavior
# on NPU tensors, including version-counter preservation across in-place ops,
# tuple and empty-tuple boundaries, and reliable error handling for invalid
# inputs.
# API name: torch.autograd._unsafe_preserve_version_counter
#
# Covered parameter dimensions:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖缺参 TypeError 与显式传参 |
# | None / 非 None | 覆盖 | None 触发 AssertionError，非 None 张量/元组正常 |
# | 枚举选项 | 不适用 | 该 API 没有枚举参数 |
# | 多类型 | 覆盖 | 覆盖单 Tensor、Tensor 元组、空元组、列表、标量 |
# | 正常输入 | 覆盖 | NPU Tensor 版本号在上下文内被保存并在退出后恢复 |
# | 异常输入 | 覆盖 | 缺参、None、列表、非 Tensor 元组元素触发异常 |
# | 边界值和等价类 | 覆盖 | 单 Tensor、多个 Tensor、空元组、in-place 边界 |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | 枚举分支 | not applicable because the API exposes no enum-style parameter |
# | 数值精度校验 | intentionally omitted; this API only preserves version counters, not numeric results |
# | More exotic container types | not covered because representative tensor / tuple / list cases already exercise the relevant validation paths |

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


def test_preserve_version_counter_single_tensor_restores_version_on_npu() -> None:
    _require_npu()

    tensor = torch.zeros(1, device=_npu_device())
    assert tensor._version == 0

    with torch.autograd._unsafe_preserve_version_counter(tensor):
        tensor.add_(1)
        assert tensor._version == 1

    assert tensor._version == 0
    assert tensor.device.type == "npu"


def test_preserve_version_counter_tuple_and_empty_tuple_boundaries_on_npu() -> None:
    _require_npu()

    left = torch.zeros(1, device=_npu_device())
    right = torch.zeros(2, device=_npu_device())

    with torch.autograd._unsafe_preserve_version_counter((left, right)):
        left.add_(1)
        right.add_(1)
        assert left._version == 1
        assert right._version == 1

    assert left._version == 0
    assert right._version == 0

    empty = torch.zeros(1, device=_npu_device())
    with torch.autograd._unsafe_preserve_version_counter(()):
        empty.add_(1)
        assert empty._version == 1
    assert empty._version == 1


def test_preserve_version_counter_invalid_inputs_raise_on_npu() -> None:
    _require_npu()

    tensor = torch.zeros(1, device=_npu_device())

    with pytest.raises(TypeError):
        torch.autograd._unsafe_preserve_version_counter()

    with pytest.raises(AssertionError):
        with torch.autograd._unsafe_preserve_version_counter(None):
            pass

    with pytest.raises(AssertionError):
        with torch.autograd._unsafe_preserve_version_counter([tensor]):
            pass

    with pytest.raises(AttributeError, match="_version"):
        with torch.autograd._unsafe_preserve_version_counter((tensor, 1)):
            pass
