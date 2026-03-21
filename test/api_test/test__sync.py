# Test purpose: validate torch._sync behavior on NPU functional tensors, covering callability,
# tensor-shape boundary cases, argument validation, and the unsupported raw-tensor path.
# API name: torch._sync
#
# Covered parameter dimensions:
#
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | input tensor passed / not passed | yes | valid tensor input and missing-argument error |
# | input tensor type functional / non-functional | partial | functional tensor succeeds; raw tensor is xfail because backend behavior is an internal assert path |
# | tensor device | yes | NPU tensor path only |
# | tensor shape boundary values | yes | scalar, zero-length, and small dense shapes |
# | non-tensor input | yes | `None` and integer inputs raise `TypeError` |
# | normal input | yes | functional NPU tensor |
# | abnormal input | yes | missing argument, `None`, integer |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | raw NPU tensor success path | `torch._sync` is implemented for functional tensors here; raw tensor invocation hits an internal assert and is not a stable contract for a reliable passing test. |
# | CPU tensor path | This file is constrained to NPU execution and must not switch the primary path off-device. |
# | keyword/optional arguments | `torch._sync` only accepts a single tensor argument in this build. |

import pytest
import torch
import torch_npu  # noqa: F401


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="torch_npu/NPU is not available in this environment.",
)


def _make_functional_npu_tensor(shape):
    numel = int(torch.Size(shape).numel())
    if numel == 0:
        base = torch.empty(shape, device="npu:0", dtype=torch.float32)
    else:
        base = torch.arange(numel, device="npu:0", dtype=torch.float32).reshape(shape if shape != () else ())
    return torch._to_functional_tensor(base)


@pytest.mark.parametrize("shape", [(), (0,), (2, 1)])
def test_torch_sync_functional_npu_tensor(shape):
    t = _make_functional_npu_tensor(shape)

    result = torch._sync(t)
    assert result is None

    restored = torch._from_functional_tensor(t)
    assert restored.device.type == "npu"
    assert tuple(restored.shape) == tuple(shape)
    assert restored.dtype == torch.float32


def test_torch_sync_missing_argument_raises_type_error():
    with pytest.raises(TypeError):
        torch._sync()


@pytest.mark.parametrize("bad_input", [None, 1])
def test_torch_sync_rejects_non_tensor_inputs(bad_input):
    with pytest.raises(TypeError):
        torch._sync(bad_input)


def test_torch_sync_raw_npu_tensor_is_xfail():
    pytest.xfail("Raw NPU tensors hit a functionalization internal assert in this build; only functional tensors are a stable success path.")

    t = torch.ones(1, device="npu:0")
    torch._sync(t)
