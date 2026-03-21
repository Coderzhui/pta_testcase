# Test purpose: validate `Tensor.new_zeros` on NPU for basic callable behavior,
#               keyword-argument handling, boundary shapes, and reliable error paths.
# API name: torch.Tensor.new_zeros
#
# Covered parameter dimensions
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | size passed / not passed | Covered | Explicit size tuple and torch.Size are exercised; missing size is not applicable to this API. |
# | size boundary values | Covered | Zero-sized and empty-dimension shapes are exercised. |
# | size normal / invalid | Covered | Normal shapes and negative / malformed size inputs are tested. |
# | dtype omitted / explicit override | Covered | Default dtype and explicit dtype override are exercised. |
# | device omitted / explicit override | Covered | Default NPU device and explicit NPU device override are exercised. |
# | requires_grad False / True | Covered | Both values are exercised. |
# | layout main candidates | Covered | `torch.strided` and `torch.sparse_coo` are exercised. |
# | pin_memory | Covered | Reliable failure path is exercised on NPU. |
#
# Uncovered items / reasons
# | Item | Reason |
# | --- | --- |
# | CPU output path via `device='cpu'` | This test is intentionally NPU-focused and must run on NPU. |
# | Exhaustive dtype matrix | API behavior is consistent across many dtypes; a representative override is sufficient here. |
# | Exhaustive invalid type matrix | A few representative malformed inputs cover the error branch without duplicating cases. |

import pytest
import torch
import torch_npu  # noqa: F401


def _npu_base_tensor():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")
    return torch.empty((2, 3), device="npu", dtype=torch.float16)


def test_tensor_new_zeros_defaults_and_boundary_shapes():
    base = _npu_base_tensor()

    y0 = base.new_zeros((0,))
    y1 = base.new_zeros(torch.Size([2, 1]))
    y2 = base.new_zeros((2, 0), requires_grad=True)

    assert y0.device.type == "npu"
    assert y1.device.type == "npu"
    assert y2.device.type == "npu"
    assert tuple(y0.shape) == (0,)
    assert tuple(y1.shape) == (2, 1)
    assert tuple(y2.shape) == (2, 0)
    assert y0.dtype == base.dtype
    assert y1.dtype == base.dtype
    assert y2.dtype == base.dtype
    assert y0.layout == torch.strided
    assert y1.layout == torch.strided
    assert y2.layout == torch.strided
    assert y0.requires_grad is False
    assert y2.requires_grad is True


def test_tensor_new_zeros_keyword_overrides():
    base = _npu_base_tensor()

    y_dtype = base.new_zeros((4,), dtype=torch.int64)
    y_device = base.new_zeros((1,), device=torch.device("npu:0"))
    y_layout = base.new_zeros((1,), layout=torch.sparse_coo)
    y_requires_grad = base.new_zeros((3,), requires_grad=True)

    assert y_dtype.device.type == "npu"
    assert y_device.device.type == "npu"
    assert y_layout.device.type == "npu"
    assert y_requires_grad.device.type == "npu"
    assert tuple(y_dtype.shape) == (4,)
    assert tuple(y_device.shape) == (1,)
    assert tuple(y_layout.shape) == (1,)
    assert tuple(y_requires_grad.shape) == (3,)
    assert y_dtype.dtype == torch.int64
    assert y_device.dtype == base.dtype
    assert y_layout.layout == torch.sparse_coo
    assert y_requires_grad.requires_grad is True


@pytest.mark.parametrize(
    "bad_size, expected_exc",
    [
        ((-1,), RuntimeError),
        ((2, -1), RuntimeError),
        ("abc", TypeError),
        (3.14, TypeError),
        (None, TypeError),
    ],
)
def test_tensor_new_zeros_invalid_size_raises(bad_size, expected_exc):
    base = _npu_base_tensor()

    with pytest.raises(expected_exc):
        base.new_zeros(bad_size)


def test_tensor_new_zeros_pin_memory_rejected_on_npu():
    base = _npu_base_tensor()

    with pytest.raises(RuntimeError, match="Only dense CPU tensors can be pinned"):
        base.new_zeros((2,), pin_memory=True)
