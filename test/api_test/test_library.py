# Test purpose: verify `torch.library` registration and dispatch behavior on
# NPU-backed tensors, including library definition, implementation binding,
# default arguments, boundary inputs, and duplicate-registration errors.
# API name: `torch.library`
# Covered parameter dimensions:
# | Dimension | Covered cases |
# | --- | --- |
# | library namespace | unique per test to avoid cross-test collisions |
# | library kind | `DEF` |
# | operator schema | tensor input with default integer argument |
# | implementation dispatch key | `NPU` |
# | call style | omitted default argument, explicit argument |
# | input boundary | empty NPU tensor |
# | error path | duplicate `define` registration |
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | `FRAGMENT` / `IMPL` library kinds | the `DEF` path exercises schema definition and NPU dispatch, which is the relevant behavior for this API slice |
# | `custom_op` / `register_fake` / `register_autograd` | this file focuses on the core `Library.define` and `Library.impl` registration flow |
# | CPU dispatch behavior | repository rules require NPU execution, so the test validates NPU-backed calls only |

import itertools

import pytest
import torch
import torch_npu  # noqa: F401
from torch.library import Library


if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("torch.library tests require an available NPU device", allow_module_level=True)


_NS_COUNTER = itertools.count()


@pytest.fixture(autouse=True)
def _use_npu0():
    torch.npu.set_device("npu:0")


def _unique_namespace() -> str:
    return f"codex_library_{next(_NS_COUNTER)}"


def _new_library() -> tuple[str, Library]:
    ns = _unique_namespace()
    return ns, Library(ns, "DEF")


def test_library_define_impl_dispatches_on_npu_with_default_and_explicit_args():
    ns, lib = _new_library()
    lib.define("identity_or_copy(Tensor x, int offset=3) -> Tensor")
    lib.impl("identity_or_copy", lambda x, offset=3: x.clone(), "NPU")

    x = torch.tensor([1.0, 2.0], device="npu")
    op = getattr(torch.ops, ns).identity_or_copy

    y_default = op(x)
    y_explicit = op(x, 5)

    for result in (y_default, y_explicit):
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "npu"
        assert result.device.index == 0
        assert result.shape == x.shape
        assert result.dtype == x.dtype


def test_library_define_impl_handles_zero_size_npu_tensor():
    ns, lib = _new_library()
    lib.define("echo(Tensor x, int scale=1) -> Tensor")
    lib.impl("echo", lambda x, scale=1: x.clone(), "NPU")

    x = torch.empty(0, device="npu", dtype=torch.float32)
    y = getattr(torch.ops, ns).echo(x)

    assert isinstance(y, torch.Tensor)
    assert y.device.type == "npu"
    assert y.device.index == 0
    assert y.shape == x.shape
    assert y.numel() == 0


def test_library_rejects_duplicate_define_registration():
    ns, lib = _new_library()
    lib.define("foo(Tensor x) -> Tensor")

    with pytest.raises(RuntimeError, match=r"same name and overload name multiple times"):
        lib.define("foo(Tensor x) -> Tensor")
