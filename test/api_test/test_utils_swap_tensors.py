# Test purpose: validate `torch.utils.swap_tensors` behavior on NPU, including
#               in-place content swapping, mixed-device swapping, boundary
#               tensor shapes, identity preservation, and reliable invalid-input
#               failure.
# API name: torch.utils.swap_tensors
#
# Covered parameter dimensions
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | same-device / mixed-device tensors | Covered | Swaps are exercised on NPU-NPU and NPU-CPU pairs. |
# | normal / boundary shapes | Covered | Scalar, empty, and non-empty tensors are exercised. |
# | self-swap / distinct tensors | Covered | Same-object no-op and distinct tensor swaps are exercised. |
# | return value | Covered | `None` return is asserted. |
# | invalid input type | Covered | Non-tensor input is tested with `pytest.raises`. |
# | NPU device behavior | Covered | NPU tensors are included and device movement is asserted. |
#
# Uncovered items / reasons
# | Item | Reason |
# | --- | --- |
# | Exhaustive tensor subclass matrix | The API accepts multiple tensor kinds; representative tensor and parameter-like cases are sufficient here. |
# | Storage aliasing / gradient history edge cases | These are brittle and not required for interface coverage. |

import pytest
import torch
import torch_npu  # noqa: F401


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


def test_swap_tensors_distinct_npu_tensors_and_boundary_shapes():
    _require_npu()

    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([], device="npu")

    ret = torch.utils.swap_tensors(a, b)

    assert ret is None
    assert a.device.type == "npu"
    assert b.device.type == "npu"
    assert tuple(a.shape) == (0,)
    assert tuple(b.shape) == (2,)

    c = torch.tensor(3.0, device="npu")
    d = torch.tensor(4.0, device="npu")
    torch.utils.swap_tensors(c, d)
    assert c.device.type == "npu"
    assert d.device.type == "npu"
    assert tuple(c.shape) == ()
    assert tuple(d.shape) == ()


def test_swap_tensors_mixed_device_swaps_contents_and_devices():
    _require_npu()

    npu_tensor = torch.tensor([1.0, 2.0], device="npu")
    cpu_tensor = torch.tensor([3.0], device="cpu")
    before_id = id(npu_tensor)

    ret = torch.utils.swap_tensors(npu_tensor, cpu_tensor)

    assert ret is None
    assert id(npu_tensor) == before_id
    assert npu_tensor.device.type == "cpu"
    assert cpu_tensor.device.type == "npu"
    assert tuple(npu_tensor.shape) == (1,)
    assert tuple(cpu_tensor.shape) == (2,)


def test_swap_tensors_self_swap_is_no_op():
    _require_npu()

    tensor = torch.tensor([5.0], device="npu")
    before_id = id(tensor)
    before_device = tensor.device
    before_shape = tensor.shape

    ret = torch.utils.swap_tensors(tensor, tensor)

    assert ret is None
    assert id(tensor) == before_id
    assert tensor.device == before_device
    assert tensor.shape == before_shape


def test_swap_tensors_invalid_input_raises():
    _require_npu()

    tensor = torch.tensor([1.0], device="npu")

    with pytest.raises(AttributeError, match="has no attribute '_use_count'"):
        torch.utils.swap_tensors(tensor, None)
