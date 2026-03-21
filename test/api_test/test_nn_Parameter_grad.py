# Test purpose: verify `Parameter.grad` read/write behavior on NPU-backed
# parameters, including default state, autograd population, manual assignment,
# and representative error paths.
# API name: `torch.nn.Parameter.grad`
# Covered parameter dimensions:
# | Dimension | Covered cases |
# | --- | --- |
# | receiver parameter device | NPU |
# | receiver parameter shape | 1D size-1 parameter, 1D size-2 parameter |
# | grad state | default `None`, autograd-populated tensor, manually assigned tensor |
# | grad device | NPU tensor, CPU tensor error |
# | grad shape | matching shape, mismatched shape error |
# | grad type | Tensor, `None`, non-Tensor error |
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | CPU parameter behavior | repository rules require NPU execution, so tests target NPU-backed parameters |
# | exhaustive dtype matrix | one representative floating-point dtype is sufficient for property semantics |
# | in-place mutation semantics of existing grad buffers | property semantics are covered by read/write and autograd cases without relying on fragile mutation details |

import pytest
import torch
import torch_npu  # noqa: F401


if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("Parameter.grad tests require an available NPU device", allow_module_level=True)


@pytest.fixture(autouse=True)
def _use_npu0():
    torch.npu.set_device("npu:0")


def test_parameter_grad_defaults_to_none_and_accepts_npu_tensor_assignment():
    param = torch.nn.Parameter(torch.tensor([1.0], device="npu"))

    assert param.device.type == "npu"
    assert param.device.index == 0
    assert param.grad is None

    assigned_grad = torch.tensor([2.0], device="npu")
    param.grad = assigned_grad

    assert param.grad is not None
    assert param.grad.device.type == "npu"
    assert param.grad.device.index == 0
    assert param.grad.shape == assigned_grad.shape
    assert param.grad.dtype == assigned_grad.dtype
    assert torch.equal(param.grad, assigned_grad)


def test_parameter_grad_is_populated_on_backward_with_npu_tensor():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0], device="npu"))

    loss = (param * 2).sum()
    loss.backward()

    assert param.grad is not None
    assert param.grad.device.type == "npu"
    assert param.grad.device.index == 0
    assert param.grad.shape == param.shape
    assert param.grad.dtype == param.dtype


@pytest.mark.parametrize(
    "bad_grad, expected_exc, match",
    [
        (torch.tensor([2.0]), RuntimeError, r"device type 'cpu'"),
        (torch.tensor([1.0, 2.0, 3.0], device="npu"), RuntimeError, r"same size"),
        (3, TypeError, r"expected to be a Tensor or None"),
    ],
)
def test_parameter_grad_rejects_invalid_assignment(bad_grad, expected_exc, match):
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0], device="npu"))

    with pytest.raises(expected_exc, match=match):
        param.grad = bad_grad

    assert param.grad is None


def test_parameter_grad_accepts_reset_to_none_after_assignment():
    param = torch.nn.Parameter(torch.tensor([1.0], device="npu"))
    param.grad = torch.tensor([4.0], device="npu")
    assert param.grad is not None

    param.grad = None
    assert param.grad is None
