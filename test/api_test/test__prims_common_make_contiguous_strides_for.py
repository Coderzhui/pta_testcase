# Test purpose: validate torch._prims_common.make_contiguous_strides_for behavior on NPU-era
# shape inputs, covering row-major/column-major stride generation, boundary shapes, and
# predictable input validation failures.
# API name: torch._prims_common.make_contiguous_strides_for
#
# Covered parameter dimensions:
#
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | shape passed / not passed | yes | default row_major path and explicit row_major=False |
# | shape container type | yes | `torch.Size`, tuple, and list inputs |
# | shape boundary values | yes | scalar, zero-sized dims, and mixed zero/non-zero dims |
# | normal input | yes | shapes sourced from NPU-backed tensors and literal shapes |
# | abnormal input | yes | `None`, integer, and negative dimensions |
# | row_major True / False | yes | both stride orderings are exercised |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | numeric tensor data validation | this API is shape/stride-only and does not operate on tensor values. |
# | backend-specific execution paths | the function is pure Python logic; NPU tensors are only used to source shapes. |
# | exotic shape element types | float-like values are coerced in this build and do not provide a stable error contract here. |

import pytest
import torch
import torch_npu  # noqa: F401

from torch._prims_common import make_contiguous_strides_for


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="torch_npu/NPU is not available in this environment.",
)


@pytest.mark.parametrize(
    "shape, expected_row_major, expected_col_major",
    [
        (torch.Size([]), (), ()),
        (torch.Size([0]), (1,), (1,)),
        (torch.Size([1]), (1,), (1,)),
        (torch.Size([2, 3]), (3, 1), (1, 2)),
        (torch.Size([2, 0, 4]), (4, 4, 1), (4, 1, 1)),
    ],
)
def test_make_contiguous_strides_for_boundary_shapes(shape, expected_row_major, expected_col_major):
    row_major = make_contiguous_strides_for(shape)
    col_major = make_contiguous_strides_for(shape, row_major=False)

    assert row_major == expected_row_major
    assert col_major == expected_col_major


def test_make_contiguous_strides_for_accepts_npu_tensor_shape():
    npu_tensor = torch.empty((2, 3, 0), device="npu:0")
    shape = npu_tensor.shape

    assert make_contiguous_strides_for(shape) == (3, 1, 1)
    assert make_contiguous_strides_for(shape, row_major=False) == (3, 1, 3)


@pytest.mark.parametrize("shape", [[], [2, 3], (2, 0, 4)])
def test_make_contiguous_strides_for_container_shape_inputs(shape):
    strides = make_contiguous_strides_for(shape)

    assert isinstance(strides, tuple)
    assert len(strides) == len(shape)
    assert all(isinstance(item, int) for item in strides)


@pytest.mark.parametrize("bad_shape", [None, 1])
def test_make_contiguous_strides_for_rejects_non_shape_inputs(bad_shape):
    with pytest.raises(AssertionError):
        make_contiguous_strides_for(bad_shape)


def test_make_contiguous_strides_for_rejects_negative_dimensions():
    with pytest.raises(RuntimeError):
        make_contiguous_strides_for((2, -1))
