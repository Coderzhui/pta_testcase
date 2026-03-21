# Test purpose: validate `torch.nn.Module.register_forward_pre_hook` behavior on
#               NPU, including hook registration, ordering, input modification,
#               keyword-argument delivery, handle removal, and reliable failure
#               when a non-callable hook is invoked.
# API name: torch.nn.Module.register_forward_pre_hook
#
# Covered parameter dimensions
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | hook callable / non-callable | Covered | Callable hooks work; a non-callable hook is registered and fails on forward. |
# | `prepend` False / True | Covered | Both insertion modes are exercised and checked by call order. |
# | `with_kwargs` False / True | Covered | Positional-only and keyword-aware hooks are both exercised. |
# | hook return / mutation | Covered | Hooks that return modified inputs are exercised. |
# | handle removal | Covered | The removable handle is removed and the hook stops firing. |
# | NPU execution | Covered | Forward runs on NPU and the hook sees NPU tensors. |
#
# Uncovered items / reasons
# | Item | Reason |
# | --- | --- |
# | Exhaustive hook return-shape failures | These are brittle and depend on downstream forward semantics, so they are not forced here. |
# | Global module forward pre-hook registration | This file targets the per-module API only. |

import pytest
import torch
import torch_npu  # noqa: F401


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


class _EchoModule(torch.nn.Module):
    def forward(self, x):
        return x


class _KwargModule(torch.nn.Module):
    def forward(self, x, scale=1):
        return x * scale


def test_register_forward_pre_hook_modifies_input_and_handle_removal():
    _require_npu()
    module = _EchoModule().npu()
    seen = []

    def hook(mod, args):
        seen.append((type(mod).__name__, args[0].device.type, tuple(args[0].shape)))
        return (args[0][:1],)

    handle = module.register_forward_pre_hook(hook)

    out1 = module(torch.ones(2, device="npu"))
    handle.remove()
    out2 = module(torch.ones(2, device="npu"))

    assert seen == [("_EchoModule", "npu", (2,))]
    assert out1.device.type == "npu"
    assert tuple(out1.shape) == (1,)
    assert out2.device.type == "npu"
    assert tuple(out2.shape) == (2,)


def test_register_forward_pre_hook_with_kwargs_and_prepend_order():
    _require_npu()
    module = _KwargModule().npu()
    order = []

    def first(mod, args, kwargs):
        order.append("first")
        assert kwargs["scale"] == 3
        return args, kwargs

    def second(mod, args, kwargs):
        order.append("second")
        assert kwargs["scale"] == 3
        return (args[0] + 1,), kwargs

    module.register_forward_pre_hook(first, with_kwargs=True)
    module.register_forward_pre_hook(second, with_kwargs=True, prepend=True)

    out = module(torch.ones(2, device="npu"), scale=3)

    assert order == ["second", "first"]
    assert out.device.type == "npu"
    assert tuple(out.shape) == (2,)


def test_register_forward_pre_hook_non_callable_fails_on_forward():
    _require_npu()
    module = _EchoModule().npu()
    module.register_forward_pre_hook(123)

    with pytest.raises(TypeError, match="object is not callable"):
        module(torch.ones(1, device="npu"))
