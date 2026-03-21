# Test purpose: verify torch.nn.Module.buffers behavior on NPU modules,
# including generator return semantics, recursion over nested buffers, handling
# of None buffers, and error behavior for invalid self objects.
# API name: torch.nn.Module.buffers
#
# Covered parameter dimensions:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖默认调用与显式 recurse 参数 |
# | None / 非 None | 覆盖 | recurse=None 与非 None 值均覆盖 |
# | 枚举选项 | 覆盖 | recurse=True / False 两态覆盖 |
# | 多类型 | 覆盖 | recurse 使用 bool、None、int、str、list、object |
# | 正常输入 | 覆盖 | 含 NPU buffer、嵌套子模块 buffer、空模块、None buffer |
# | 异常输入 | 覆盖 | 非 Module self 调用触发 AttributeError |
# | 边界值和等价类 | 覆盖 | 空模块、一次性 generator、None buffer、嵌套模块边界 |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | 数值精度校验 | intentionally omitted; buffers() only exposes registered tensors and their traversal |
# | recurse 参数严格类型错误 | not covered because current runtime coerces non-bool values instead of rejecting them |

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


def test_buffers_default_and_recursive_iteration_on_npu() -> None:
    _require_npu()

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("top", torch.ones(2, device=_npu_device()))
            self.child = torch.nn.Module()
            self.child.register_buffer("leaf", torch.zeros(3, device=_npu_device()))

    net = Net()

    default_buffers = net.buffers()
    assert isinstance(default_buffers, type((x for x in [])))
    assert [tuple(t.shape) for t in default_buffers] == [(2,), (3,)]

    shallow_buffers = list(net.buffers(recurse=False))
    assert [tuple(t.shape) for t in shallow_buffers] == [(2,)]

    named_recursive = list(net.named_buffers())
    assert [name for name, _ in named_recursive] == ["top", "child.leaf"]
    assert [tensor.device.type for _, tensor in named_recursive] == ["npu", "npu"]

    named_shallow = list(net.named_buffers(recurse=False))
    assert [name for name, _ in named_shallow] == ["top"]
    assert named_shallow[0][1].device.type == "npu"


def test_buffers_none_buffer_and_generator_boundary_on_npu() -> None:
    _require_npu()

    module = torch.nn.Module()
    module.register_buffer("missing", None)
    module.register_buffer("present", torch.ones(1, device=_npu_device()))

    first_pass = list(module.buffers())
    second_pass = list(module.buffers())

    assert len(first_pass) == 1
    assert len(second_pass) == 1
    assert first_pass[0].device.type == "npu"
    assert first_pass[0].shape == torch.Size([1])
    assert module.missing is None
    assert list(module.named_buffers()) == [("present", first_pass[0])]


def test_buffers_recurse_coercion_and_invalid_self_error_on_npu() -> None:
    _require_npu()

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("top", torch.ones(1, device=_npu_device()))
            self.child = torch.nn.Module()
            self.child.register_buffer("leaf", torch.zeros(1, device=_npu_device()))

    net = Net()

    assert [tuple(t.shape) for t in net.buffers(recurse=None)] == [(1,)]
    assert [tuple(t.shape) for t in net.buffers(recurse=0)] == [(1,)]
    assert [tuple(t.shape) for t in net.buffers(recurse=1)] == [(1,), (1,)]
    assert [tuple(t.shape) for t in net.buffers(recurse="yes")] == [(1,), (1,)]
    assert [tuple(t.shape) for t in net.buffers(recurse=[])] == [(1,)]
    assert [tuple(t.shape) for t in net.buffers(recurse=object())] == [(1,), (1,)]

    with pytest.raises(AttributeError, match="named_buffers"):
        list(torch.nn.Module.buffers(None))

    with pytest.raises(AttributeError, match="named_buffers"):
        list(torch.nn.Module.buffers(object()))
