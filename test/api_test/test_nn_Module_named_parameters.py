"""
Test purpose: validate torch.nn.Module.named_parameters behavior on an NPU-backed module.
API name: torch.nn.Module.named_parameters

Coverage table:
| Parameter dimension | Coverage |
| --- | --- |
| Parameter passed / not passed | Covered: default call and explicit keyword arguments are both exercised |
| None / non-None | Covered: prefix is exercised as empty string and non-empty string; boolean flags are non-None |
| Enum options | Uncovered: the API does not expose enum parameters |
| Multiple types | Uncovered: the API mainly accepts strings and booleans, so no heterogeneous type matrix applies |
| Normal input | Covered: recursive traversal, prefix handling, duplicate suppression, and NPU device placement |
| Error input | Covered: too many positional arguments raise TypeError |
| Boundary / equivalence classes | Covered: recurse=True vs recurse=False, remove_duplicate=True vs False, default vs explicit prefix |

Uncovered items and reasons:
- Enum options: not applicable because named_parameters has no enum-style arguments.
- Multiple types: not applicable because the callable only accepts a string prefix and boolean flags.
- Numeric boundary checks: not applicable because this API is an iterator over parameters, not a numeric operator.
"""

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    torch_npu = None
    _TORCH_NPU_IMPORT_ERROR = exc
else:  # pragma: no cover - simple import guard
    _TORCH_NPU_IMPORT_ERROR = None


def _require_npu():
    if torch_npu is None:
        pytest.skip(f"torch_npu is unavailable: {_TORCH_NPU_IMPORT_ERROR}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


class SharedParamModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.top = torch.nn.Parameter(torch.arange(2, dtype=torch.float32))
        shared = torch.nn.Linear(2, 2, bias=False)
        self.branch1 = shared
        self.branch2 = shared


def _build_module_on_npu():
    module = SharedParamModule()
    module = module.npu()
    return module


def test_named_parameters_default_and_explicit_keywords_on_npu():
    _require_npu()

    module = _build_module_on_npu()
    default_items = list(module.named_parameters())
    explicit_items = list(module.named_parameters(prefix="", recurse=True, remove_duplicate=True))

    assert len(default_items) == len(explicit_items)
    assert [name for name, _ in default_items] == [name for name, _ in explicit_items]
    assert all(param.device.type == "npu" for _, param in default_items)
    assert all(isinstance(param, torch.nn.Parameter) for _, param in default_items)


def test_named_parameters_prefix_and_recurse_false_on_npu():
    _require_npu()

    module = _build_module_on_npu()

    prefixed_names = [name for name, _ in module.named_parameters(prefix="root")]
    non_recursive_names = [name for name, _ in module.named_parameters(recurse=False)]

    assert prefixed_names == [
        "root.top",
        "root.branch1.weight",
    ]
    assert non_recursive_names == ["top"]


def test_named_parameters_remove_duplicate_controls_shared_module_entries_on_npu():
    _require_npu()

    module = _build_module_on_npu()

    deduplicated_names = [name for name, _ in module.named_parameters(remove_duplicate=True)]
    all_names = [name for name, _ in module.named_parameters(remove_duplicate=False)]

    assert deduplicated_names == [
        "top",
        "branch1.weight",
    ]
    assert all_names == [
        "top",
        "branch1.weight",
        "branch2.weight",
    ]


def test_named_parameters_rejects_too_many_positional_arguments():
    _require_npu()

    module = _build_module_on_npu()

    with pytest.raises(TypeError):
        list(module.named_parameters("root", True, True, "extra"))
