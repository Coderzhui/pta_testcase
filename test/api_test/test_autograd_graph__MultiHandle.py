# Test purpose: verify torch.autograd.graph._MultiHandle behavior on NPU hook
# handles, including aggregated removal, empty-handle boundaries, idempotent
# removal, and deferred error handling for malformed handle containers.
# API name: torch.autograd.graph._MultiHandle
#
# Covered parameter dimensions:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖显式传入 handles tuple，以及空 tuple 边界 |
# | None / 非 None | 覆盖 | None 传入后在 remove() 阶段触发异常，非 None 正常 |
# | 枚举选项 | 不适用 | 该 API 无枚举参数 |
# | 多类型 | 覆盖 | 覆盖 tuple[RemovableHandle], empty tuple, None, int, mixed tuple |
# | 正常输入 | 覆盖 | NPU Tensor 上注册的多个 hook 被统一移除 |
# | 异常输入 | 覆盖 | None / int / mixed tuple 在 remove() 时触发异常 |
# | 边界值和等价类 | 覆盖 | 空 handles、双 hook handles、重复 remove 的幂等边界 |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | 枚举分支 | not applicable because the API exposes no enum-style parameter |
# | 数值精度校验 | intentionally omitted; this API manages hook handles, not tensor values |

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


def test_multihandle_removes_multiple_npu_hooks_and_preserves_empty_boundary() -> None:
    _require_npu()

    tensor = torch.tensor([1.0], device=_npu_device(), requires_grad=True)
    calls = []

    def hook_one(grad):
        calls.append("one")
        return grad

    def hook_two(grad):
        calls.append("two")
        return grad

    handle_one = tensor.register_hook(hook_one)
    handle_two = tensor.register_hook(hook_two)

    multi_handle = torch.autograd.graph._MultiHandle((handle_one, handle_two))
    assert isinstance(multi_handle.next_id, int)
    assert multi_handle.next_id >= 0

    multi_handle.remove()
    (tensor * 2).sum().backward()

    assert calls == []
    assert tensor.grad is not None
    assert tensor.grad.device.type == "npu"

    empty_multi_handle = torch.autograd.graph._MultiHandle(())
    assert isinstance(empty_multi_handle.next_id, int)
    empty_multi_handle.remove()
    empty_multi_handle.remove()


def test_multihandle_deferred_error_paths_on_npu() -> None:
    _require_npu()

    invalid_cases = [
        None,
        123,
        (123,),
        (torch.tensor(1),),
    ]

    for handles in invalid_cases:
        multi_handle = torch.autograd.graph._MultiHandle(handles)
        if handles is None:
            with pytest.raises(TypeError, match="not iterable"):
                multi_handle.remove()
        elif handles == 123:
            with pytest.raises(TypeError, match="not iterable"):
                multi_handle.remove()
        elif handles == (123,):
            with pytest.raises(AttributeError, match="remove"):
                multi_handle.remove()
        else:
            with pytest.raises(AttributeError, match="remove"):
                multi_handle.remove()
