# Test purpose: validate `torch.nn.Module.modules` traversal behavior on NPU,
#               including ordering, deduplication, empty-module boundaries, and
#               iterator semantics for module enumeration.
# API name: torch.nn.Module.modules
#
# Covered parameter dimensions
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | empty vs non-empty module tree | Covered | A bare module and a nested module tree are both exercised. |
# | self included / child modules included | Covered | `self` is yielded first and children are yielded afterward. |
# | duplicate submodule references | Covered | Aliased modules are returned only once. |
# | iterator behavior | Covered | The returned value is consumed as an iterator and compared as a list. |
# | NPU device placement | Covered | Child parameters are verified to remain on NPU after `.npu()`. |
#
# Uncovered items / reasons
# | Item | Reason |
# | --- | --- |
# | Reliable input-driven error case | `modules()` has no arguments and no stable, supported failure path in this environment. |

import pytest
import torch
import torch_npu  # noqa: F401


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


class _NestedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3, 4)
        self.seq = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(4, 2))
        self.alias = self.l1


def test_module_modules_returns_self_then_children_on_npu():
    _require_npu()
    module = _NestedModule().npu()

    mods = list(module.modules())

    assert mods[0] is module
    assert [type(m).__name__ for m in mods] == ["_NestedModule", "Linear", "Sequential", "ReLU", "Linear"]
    assert sum(m is module.l1 for m in mods) == 1
    assert sum(m is module.seq for m in mods) == 1
    assert sum(m is module.alias for m in mods) == 1
    assert all(
        any(param.device.type == "npu" for param in m.parameters(recurse=False))
        if any(True for _ in m.parameters(recurse=False))
        else True
        for m in mods
    )


def test_module_modules_empty_tree_yields_only_self():
    _require_npu()
    module = torch.nn.Module().npu()

    mods = list(module.modules())

    assert mods == [module]
    assert mods[0].__class__.__name__ == "Module"


def test_module_modules_iterator_can_be_consumed_once_and_recreated():
    _require_npu()
    module = _NestedModule().npu()

    first_pass = list(module.modules())
    second_pass = list(module.modules())

    assert first_pass == second_pass
    assert len(first_pass) == 5
    assert len(second_pass) == 5


def test_module_modules_invalid_self_raises_attribute_error():
    _require_npu()

    with pytest.raises(AttributeError, match="named_modules"):
        list(torch.nn.Module.modules(None))
