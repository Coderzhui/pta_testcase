# Test purpose: verify `Module.named_modules` on NPU-backed modules, including
# traversal order, prefix handling, duplicate suppression, memo handling, and
# a representative error path.
# API name: `torch.nn.Module.named_modules`
# Covered parameter dimensions:
# | Dimension | Covered cases |
# | --- | --- |
# | receiver module topology | root-only module, nested module tree, shared submodule |
# | receiver module device | NPU-backed module and buffers |
# | `memo` | omitted, prefilled set, invalid non-iterable error |
# | `prefix` | default empty prefix, non-empty string prefix |
# | `remove_duplicate` | `True`, `False` |
# | return shape | root entry only, nested entries, duplicate entries omitted or preserved |
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | CPU-only traversal behavior | this repository requires NPU execution, so the tests validate NPU-backed modules |
# | exhaustive invalid-type matrix for `prefix` / `remove_duplicate` | current implementation is duck-typed for some inputs; one representative error case is sufficient |
# | non-iterable memo variants beyond one failing example | the API contract is covered by one reliable `pytest.raises` path |

import pytest
import torch
import torch_npu  # noqa: F401


if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("torch.nn.Module.named_modules tests require an available NPU device", allow_module_level=True)


@pytest.fixture(autouse=True)
def _use_npu0():
    torch.npu.set_device("npu:0")


def _build_shared_module() -> torch.nn.Module:
    class SharedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.nn.Linear(2, 2)
            self.a = shared
            self.b = shared
            self.seq = torch.nn.Sequential(torch.nn.ReLU())

    return SharedModule().npu()


def test_named_modules_returns_expected_order_and_prefix_on_npu():
    module = _build_shared_module()

    assert next(module.parameters()).device.type == "npu"
    assert next(module.parameters()).device.index == 0

    default_items = list(module.named_modules())
    assert [name for name, _ in default_items] == ["", "a", "seq", "seq.0"]
    assert default_items[0][1] is module
    assert default_items[1][1] is module.a
    assert default_items[2][1] is module.seq
    assert default_items[3][1] is module.seq[0]

    prefixed_items = list(module.named_modules(prefix="root"))
    assert [name for name, _ in prefixed_items] == ["root", "root.a", "root.seq", "root.seq.0"]
    assert prefixed_items[0][1] is module
    assert prefixed_items[1][1] is module.a
    assert prefixed_items[2][1] is module.seq
    assert prefixed_items[3][1] is module.seq[0]


@pytest.mark.parametrize(
    "scenario",
    [
        "keep_duplicates",
        "prefilled_memo",
        "root_only",
    ],
)
def test_named_modules_boundary_cases_for_duplicates_memo_and_empty_root(scenario):
    if scenario == "keep_duplicates":
        module = _build_shared_module()
        items = list(module.named_modules(remove_duplicate=False))
        assert [name for name, _ in items] == ["", "a", "b", "seq", "seq.0"]
        assert items[1][1] is module.a
        assert items[2][1] is module.b
        assert items[3][1] is module.seq
        assert items[4][1] is module.seq[0]
        return

    if scenario == "prefilled_memo":
        module = _build_shared_module()
        items = list(module.named_modules(memo={module.a}))
        assert [name for name, _ in items] == ["", "seq", "seq.0"]
        assert items[0][1] is module
        assert items[1][1] is module.seq
        assert items[2][1] is module.seq[0]
        return

    root_only = torch.nn.Module()
    root_only.register_buffer("marker", torch.tensor([1.0], device="npu"))
    items = list(root_only.named_modules())
    assert items == [("", root_only)]
    assert root_only.marker.device.type == "npu"
    assert root_only.marker.device.index == 0


def test_named_modules_rejects_non_iterable_memo():
    module = torch.nn.Linear(2, 2).npu()

    with pytest.raises(TypeError, match=r"iterable"):
        list(module.named_modules(memo=1))
