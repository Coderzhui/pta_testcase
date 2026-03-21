# Test purpose: validate Tensor.new_empty API behavior on NPU, focusing on callability,
# device/dtype inheritance and override behavior, boundary shapes, and reliable error handling.
# API name: Tensor.new_empty
#
# Covered parameter dimensions:
#
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | size passed / not passed | yes | normal call and missing-size error |
# | size boundary values | yes | scalar shape, zero-sized dims, mixed dims |
# | dtype omitted / explicit override | yes | inherited dtype and explicit override |
# | device omitted / explicit override | yes | inherited NPU device and explicit NPU device |
# | requires_grad omitted / explicit override | partial | explicit True covered; default False observed |
# | layout default / explicit override | partial | default strided covered; non-strided not reliable on NPU |
# | normal input | yes | valid tensor inputs on NPU |
# | abnormal input | yes | missing size and negative dimension |
# | pin_memory | no | not reliable for this NPU-focused test |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | non-strided layout values | Dense NPU tensors reliably support strided layout; other layouts are backend-dependent and not needed for the functional contract here. |
# | pin_memory=True | Not a reliable NPU execution path for this API in this environment. |
# | CPU output device override | This file is constrained to run on NPU and should not shift the primary execution path off-device. |

import pytest
import torch
import torch_npu  # noqa: F401


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="torch_npu/NPU is not available in this environment.",
)


def _make_npu_base_tensor() -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0], device="npu:0")


def test_tensor_new_empty_inherits_npu_defaults_and_accepts_overrides():
    base = _make_npu_base_tensor()

    inherited = base.new_empty((2, 0))
    assert inherited.device.type == "npu"
    assert inherited.shape == (2, 0)
    assert inherited.dtype == base.dtype
    assert inherited.layout == torch.strided

    overridden = base.new_empty(torch.Size([1, 2]), dtype=torch.float16, device=torch.device("npu:0"), requires_grad=True)
    assert overridden.device.type == "npu"
    assert overridden.shape == (1, 2)
    assert overridden.dtype == torch.float16
    assert overridden.requires_grad is True
    assert overridden.layout == torch.strided


@pytest.mark.parametrize(
    "size",
    [
        (),
        (0,),
        (2, 0, 3),
        torch.Size([1, 1]),
    ],
)
def test_tensor_new_empty_boundary_shapes(size):
    base = _make_npu_base_tensor()

    out = base.new_empty(size)
    assert out.device.type == "npu"
    assert tuple(out.shape) == tuple(size)
    assert out.layout == torch.strided
    assert out.numel() == 0 or out.numel() == 1


def test_tensor_new_empty_missing_size_raises_type_error():
    base = _make_npu_base_tensor()

    with pytest.raises(TypeError):
        base.new_empty()


@pytest.mark.parametrize("bad_size", [(2, -1), torch.Size([1, -2])])
def test_tensor_new_empty_negative_dimension_raises(bad_size):
    base = _make_npu_base_tensor()

    with pytest.raises(RuntimeError):
        base.new_empty(bad_size)
