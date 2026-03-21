# Test purpose: validate `torch._from_functional_tensor` on NPU for converting
#               functional tensors back to regular tensors, covering normal
#               round-trips, boundary shapes, and reliable invalid-input errors.
# API name: torch._from_functional_tensor
#
# Covered parameter dimensions
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | functional tensor input / non-functional tensor input | Covered | A real functionalized tensor succeeds; a plain tensor triggers the internal assert path. |
# | valid tensor / invalid non-tensor input | Covered | Tensor round-trip and invalid Python object inputs are both exercised. |
# | normal / boundary shapes | Covered | Non-empty, empty, and scalar tensors are exercised. |
# | contiguous / non-contiguous layout | Covered | A transposed input is included to verify shape/stride preservation. |
# | NPU execution | Covered | All positive cases use NPU tensors. |
#
# Uncovered items / reasons
# | Item | Reason |
# | --- | --- |
# | Exhaustive dtype matrix | The API is metadata-preserving; representative float NPU tensors are sufficient here. |
# | Behavior on exotic tensor subclasses | This test targets the standard tensor path and functional wrapper behavior. |

import pytest
import torch
import torch_npu  # noqa: F401


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: torch.arange(4, device="npu").reshape(2, 2),
        lambda: torch.empty(0, device="npu"),
        lambda: torch.tensor(3.0, device="npu"),
        lambda: torch.arange(6, device="npu").reshape(2, 3).transpose(0, 1),
    ],
)
def test_from_functional_tensor_roundtrip_on_npu(factory):
    _require_npu()
    base = factory()

    functional = torch._to_functional_tensor(base)
    restored = torch._from_functional_tensor(functional)

    assert restored.device.type == "npu"
    assert tuple(restored.shape) == tuple(base.shape)
    assert restored.layout == base.layout
    assert restored.stride() == base.stride()


def test_from_functional_tensor_rejects_plain_tensor_with_internal_assert():
    _require_npu()

    plain = torch.tensor([1.0, 2.0], device="npu")

    with pytest.raises(RuntimeError, match="assert_functional"):
        torch._from_functional_tensor(plain)


@pytest.mark.parametrize("bad_input", [1, None, "x"])
def test_from_functional_tensor_rejects_non_tensor_inputs(bad_input):
    _require_npu()

    with pytest.raises(TypeError, match="incompatible function arguments"):
        torch._from_functional_tensor(bad_input)
