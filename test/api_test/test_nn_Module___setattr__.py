# Test purpose: validate `torch.nn.Module.__setattr__` routing and validation
#               on NPU, including parameter/module/buffer registration,
#               plain attribute assignment, `None` clearing, and reliable
#               type-error cases for incompatible replacements.
# API name: torch.nn.Module.__setattr__
#
# Covered parameter dimensions
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | plain attribute / registered parameter / registered buffer / submodule | Covered | Each assignment path is exercised on an NPU-backed module. |
# | tensor / Parameter / Module / None values | Covered | Representative values are assigned through `__setattr__`. |
# | replace existing entry / new entry | Covered | New registration and replacement of existing slots are both exercised. |
# | valid / invalid replacement | Covered | Compatible assignments succeed; incompatible replacements raise `TypeError`. |
# | NPU execution | Covered | All tensor/module objects are created on NPU. |
#
# Uncovered items / reasons
# | Item | Reason |
# | --- | --- |
# | Exhaustive registry edge cases | The API behavior is sufficiently covered by representative parameter, buffer, and submodule cases. |
# | Serialization / optimizer interaction | This test targets attribute routing only, not downstream training utilities. |

import pytest
import torch
import torch_npu  # noqa: F401


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


class _Child(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = torch.nn.ReLU()


def test_module_setattr_routes_new_assignments_on_npu():
    _require_npu()

    module = torch.nn.Module().npu()
    param = torch.nn.Parameter(torch.tensor([1.0], device="npu"))
    child = _Child().npu()
    plain = torch.tensor([3.0], device="npu")

    module.p = param
    module.child = child
    module.t = plain

    assert list(module._parameters.keys()) == ["p"]
    assert list(module._modules.keys()) == ["child"]
    assert list(module._buffers.keys()) == []
    assert module.p is param
    assert module.child is child
    assert module.t is plain
    assert module.p.device.type == "npu"
    assert module.child.inner is child.inner
    assert module.t.device.type == "npu"


def test_module_setattr_updates_buffer_and_clears_registered_slots():
    _require_npu()

    module = torch.nn.Module().npu()
    module.register_buffer("b", torch.tensor([2.0], device="npu"))
    module.register_parameter("p", torch.nn.Parameter(torch.tensor([1.0], device="npu")))
    module.child = torch.nn.ReLU().npu()

    replacement_buffer = torch.tensor([4.0], device="npu")

    module.b = replacement_buffer
    module.p = None
    module.child = None

    assert list(module._buffers.keys()) == ["b"]
    assert list(module._parameters.keys()) == ["p"]
    assert list(module._modules.keys()) == ["child"]
    assert module.b is replacement_buffer
    assert module.b.device.type == "npu"
    assert module._parameters["p"] is None
    assert module._modules["child"] is None
    assert getattr(module, "p") is None
    assert getattr(module, "child") is None


def test_module_setattr_rejects_incompatible_replacements():
    _require_npu()

    module = torch.nn.Module().npu()
    module.p = torch.nn.Parameter(torch.tensor([1.0], device="npu"))
    module.child = torch.nn.ReLU().npu()

    with pytest.raises(TypeError, match="cannot assign 'torch\\.npu\\..*Tensor' as parameter 'p'"):
        module.p = torch.tensor([2.0], device="npu")

    with pytest.raises(TypeError, match="cannot assign 'torch\\.npu\\..*Tensor' as child module 'child'"):
        module.child = torch.tensor([3.0], device="npu")


def test_module_setattr_allows_plain_non_tensor_attributes():
    _require_npu()

    module = torch.nn.Module().npu()
    module.note = "plain attribute"
    module.count = 3

    assert module.note == "plain attribute"
    assert module.count == 3
    assert "note" not in module._parameters
    assert "note" not in module._modules
    assert "note" not in module._buffers
