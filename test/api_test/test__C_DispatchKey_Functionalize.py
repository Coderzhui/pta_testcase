# Test purpose: verify `torch._C.DispatchKey.Functionalize` behaves as a stable
# enum member and interacts correctly with NPU tensor dispatch-key inspection.
# API name: `torch._C.DispatchKey.Functionalize`
# Covered parameter dimensions:
# | Dimension | Covered cases |
# | --- | --- |
# | enum access | direct attribute access |
# | enum identity / stability | exact member identity, hashability, name/value/repr |
# | tensor boundary context | NPU tensor dispatch-key set inspection |
# | membership-style usage | `DispatchKeySet.has(...)` |
# | invalid access | missing enum attribute, unsupported membership syntax |
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | full `DispatchKey` enum traversal | the task is specific to the Functionalize member, so broader enum enumeration is unnecessary |
# | CPU tensor dispatch-key behavior | repository rules require NPU execution, so the boundary case is validated on an NPU tensor |
# | functionalization runtime transforms | this file validates the enum/interface surface, not the full functionalization subsystem |

import pytest
import torch
import torch_npu  # noqa: F401


if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("DispatchKey.Functionalize tests require an available NPU device", allow_module_level=True)


@pytest.fixture(autouse=True)
def _use_npu0():
    torch.npu.set_device("npu:0")


def test_dispatchkey_functionalize_has_expected_enum_properties():
    key = torch._C.DispatchKey.Functionalize

    assert key.name == "Functionalize"
    assert key.value == 16
    assert repr(key) == "<DispatchKey.Functionalize: 16>"
    assert str(key) == "DispatchKey.Functionalize"
    assert hash(key) == hash(torch._C.DispatchKey.Functionalize)
    assert key is torch._C.DispatchKey.Functionalize


def test_dispatchkey_functionalize_is_absent_from_npu_tensor_dispatch_set():
    tensor = torch.tensor([1], device="npu")
    dispatch_keys = torch._C._dispatch_keys(tensor)

    assert isinstance(dispatch_keys, torch.DispatchKeySet)
    assert dispatch_keys.has(torch._C.DispatchKey.PrivateUse1)
    assert dispatch_keys.has(torch._C.DispatchKey.AutogradPrivateUse1)
    assert dispatch_keys.has(torch._C.DispatchKey.AutocastPrivateUse1)
    assert dispatch_keys.has(torch._C.DispatchKey.ADInplaceOrView)
    assert not dispatch_keys.has(torch._C.DispatchKey.Functionalize)


def test_dispatchkey_functionalize_rejects_invalid_access_patterns():
    dispatch_keys = torch._C._dispatch_keys(torch.tensor([1], device="npu"))

    with pytest.raises(AttributeError):
        getattr(torch._C.DispatchKey, "NoSuchKey")

    with pytest.raises(TypeError, match=r"not iterable"):
        torch._C.DispatchKey.Functionalize in dispatch_keys
