# Test purpose: verify that `Tensor.untyped_storage` returns an NPU-backed
# storage object and behaves correctly for normal, boundary, and error inputs.
# API name: `torch.Tensor.untyped_storage`
# Covered parameter dimensions:
# | Dimension | Covered cases |
# | --- | --- |
# | receiver tensor device | NPU tensor |
# | receiver tensor shape | non-empty tensor, zero-size tensor, non-contiguous view |
# | receiver tensor dtype | int64, float32, bool |
# | method arguments | no extra args, extra positional arg error |
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | CPU behavior | this repository requires NPU execution and the target API is being validated on NPU tensors |
# | exact storage byte contents | the test targets interface/device behavior, not numerical content verification |
# | keyword-argument error variants | the method accepts no arguments; one representative argument error is sufficient |

import pytest
import torch
import torch_npu  # noqa: F401


if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Tensor.untyped_storage tests require an available NPU device", allow_module_level=True)


@pytest.fixture(autouse=True)
def _use_npu0():
    torch.npu.set_device("npu:0")


def _assert_npu_storage(tensor: torch.Tensor, expected_nbytes: int) -> None:
    storage = tensor.untyped_storage()

    assert storage is not None
    assert storage.device.type == "npu"
    assert storage.device.index == 0
    assert len(storage) == expected_nbytes
    assert storage.nbytes() == expected_nbytes
    assert storage.data_ptr() != 0


def _make_empty_float32():
    return torch.empty(0, device="npu", dtype=torch.float32), 0, None


def _make_noncontig_float32_view():
    base_tensor = torch.arange(6, device="npu", dtype=torch.float32).reshape(2, 3)
    return base_tensor.t(), 6 * torch.empty((), dtype=torch.float32).element_size(), base_tensor


def _make_bool_tensor():
    return torch.tensor([True, False], device="npu", dtype=torch.bool), 2, None


def test_untyped_storage_returns_expected_npu_storage_for_normal_tensor():
    tensor = torch.tensor([1, 2, 3, 4], device="npu", dtype=torch.int64)
    _assert_npu_storage(tensor, tensor.numel() * tensor.element_size())


@pytest.mark.parametrize(
    "tensor_factory",
    [
        _make_empty_float32,
        _make_noncontig_float32_view,
        _make_bool_tensor,
    ],
)
def test_untyped_storage_handles_boundary_and_view_cases(tensor_factory):
    tensor, expected_nbytes, base_tensor = tensor_factory()
    storage = tensor.untyped_storage()

    assert storage.device.type == "npu"
    assert storage.device.index == 0
    assert len(storage) == expected_nbytes
    assert storage.nbytes() == expected_nbytes

    if tensor.numel() == 0:
        assert storage.data_ptr() == 0
    else:
        assert storage.data_ptr() != 0

    if not tensor.is_contiguous() and tensor.numel() > 0:
        assert storage.data_ptr() == base_tensor.untyped_storage().data_ptr()
        assert storage.nbytes() == base_tensor.numel() * base_tensor.element_size()


def test_untyped_storage_rejects_extra_positional_argument():
    tensor = torch.arange(4, device="npu")

    with pytest.raises(TypeError, match=r"takes no arguments"):
        tensor.untyped_storage(1)
