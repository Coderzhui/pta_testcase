# Test purpose: validate torch.nn.Parameter.itemsize behavior on NPU-backed Parameters,
# covering dtype-dependent element size, shape boundaries, and basic attribute access.
# API name: torch.nn.Parameter.itemsize
#
# Covered parameter dimensions:
#
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | parameter constructed / not constructed | yes | valid Parameter creation and one constructor error path |
# | dtype None / non-None | partial | explicit dtypes covered; default dtype is not required for itemsize behavior |
# | dtype families | yes | float, half, int64, bool |
# | tensor shape boundaries | yes | scalar, empty, and small dense shapes |
# | requires_grad True / False | partial | floating-point Parameters use default `True`; integer/bool Parameters use `False` |
# | normal input | yes | NPU-backed Parameter values |
# | abnormal input | partial | constructor error on non-Tensor input is covered; `itemsize` itself has no meaningful runtime error path |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | direct `itemsize` error cases | `itemsize` is a read-only tensor attribute and does not expose a separate failure mode for valid Parameters. |
# | CPU backend behavior | this file is constrained to NPU execution and uses NPU-backed Parameters only. |
# | layout / stride variants | `itemsize` is independent of layout and stride, so varying them would not add meaningful coverage here. |

import pytest
import torch
import torch_npu  # noqa: F401

from torch.nn import Parameter


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="torch_npu/NPU is not available in this environment.",
)


@pytest.mark.parametrize(
    "dtype, expected_itemsize, requires_grad",
    [
        (torch.float32, 4, True),
        (torch.float16, 2, True),
        (torch.int64, 8, False),
        (torch.bool, 1, False),
    ],
)
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_parameter_itemsize_matches_dtype_width_on_npu(dtype, expected_itemsize, requires_grad, shape):
    base = torch.zeros(shape if shape != () else (), device="npu:0", dtype=dtype)
    param = Parameter(base, requires_grad=requires_grad)

    assert param.device.type == "npu"
    assert tuple(param.shape) == tuple(shape)
    assert param.itemsize == expected_itemsize
    assert param.element_size() == expected_itemsize


def test_parameter_itemsize_reflects_tensor_attribute_on_npu():
    tensor = torch.ones((1,), device="npu:0", dtype=torch.float32)
    param = Parameter(tensor, requires_grad=True)

    assert hasattr(param, "itemsize")
    assert isinstance(param.itemsize, int)
    assert param.itemsize == tensor.itemsize


def test_parameter_non_tensor_constructor_input_raises_type_error():
    with pytest.raises(AttributeError):
        Parameter(123)  # type: ignore[arg-type]
