# Test purpose: validate `torch.nn.Parameter.stride` behavior on NPU, covering
#               tuple/int return modes, boundary shapes, non-contiguous layout,
#               negative-dimension indexing, and reliable error handling for
#               out-of-range dimensions.
# API name: torch.nn.Parameter.stride
#
# Covered parameter dimensions
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | no-arg / dim-arg | Covered | Both `stride()` and `stride(dim)` are exercised. |
# | positive / negative dim | Covered | Positive dims and negative indexing are exercised. |
# | contiguous / non-contiguous parameter | Covered | A transposed parameter is used to verify non-contiguous stride values. |
# | empty / scalar / normal shapes | Covered | Boundary and normal parameter shapes are exercised. |
# | valid / invalid dim range | Covered | In-range queries and out-of-range `IndexError` are exercised. |
#
# Uncovered items / reasons
# | Item | Reason |
# | --- | --- |
# | Exhaustive storage/layout matrix | `stride` is a read-only metadata query; representative contiguous and non-contiguous NPU cases are sufficient. |
# | Error cases beyond invalid dimension indexing | The API surface is small and the reliable error path here is out-of-range indexing. |

import pytest
import torch
import torch_npu  # noqa: F401


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


def test_parameter_stride_contiguous_noncontiguous_and_boundaries_on_npu():
    _require_npu()

    contiguous = torch.nn.Parameter(torch.randn(2, 3, device="npu"))
    noncontiguous = torch.nn.Parameter(torch.randn(2, 3, device="npu").transpose(0, 1))
    empty = torch.nn.Parameter(torch.empty(0, device="npu"))
    scalar = torch.nn.Parameter(torch.tensor(3.0, device="npu"))

    assert contiguous.device.type == "npu"
    assert noncontiguous.device.type == "npu"
    assert empty.device.type == "npu"
    assert scalar.device.type == "npu"

    assert contiguous.stride() == (3, 1)
    assert contiguous.stride(0) == 3
    assert contiguous.stride(1) == 1
    assert contiguous.stride(-1) == 1

    assert noncontiguous.stride() == (1, 3)
    assert noncontiguous.stride(0) == 1
    assert noncontiguous.stride(1) == 3
    assert noncontiguous.stride(-2) == 1

    assert empty.stride() == (1,)
    assert empty.stride(0) == 1
    assert scalar.stride() == ()


def test_parameter_stride_out_of_range_dim_raises():
    _require_npu()

    param = torch.nn.Parameter(torch.randn(2, 3, device="npu"))

    with pytest.raises(IndexError, match="Dimension out of range"):
        param.stride(2)


def test_parameter_stride_negative_dim_matches_last_dimension():
    _require_npu()

    param = torch.nn.Parameter(torch.randn(4, 5, device="npu").transpose(0, 1))

    assert param.stride(-1) == param.stride(1)
