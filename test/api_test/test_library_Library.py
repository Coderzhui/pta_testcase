# Test purpose: validate torch.library.Library construction and registration behavior on NPU,
# covering namespace/kind validation, operator definition, NPU kernel registration, fallback
# error handling, and reliable duplicate-registration failures.
# API name: torch.library.Library
#
# Covered parameter dimensions:
#
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | kind valid / invalid | yes | DEF, IMPL, FRAGMENT and an invalid kind |
# | reserved namespace / normal namespace | yes | reserved `prim` rejection and normal custom namespaces |
# | schema valid / invalid | yes | valid op schema, malformed schema errors |
# | dispatch_key None / non-None | partial | explicit `NPU` and library-default `NPU` covered; empty default fallback path is error-only |
# | callable / non-callable impl fn | yes | valid callable and invalid integer |
# | duplicate registration | yes | same op + dispatch key triggers RuntimeError |
# | fallback behavior | partial | error paths covered; success path is backend-dependent in this environment |
# | NPU execution | yes | defined custom op executes on NPU input |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | successful global fallback registration | torch_npu already installs a backend fallback in this environment, so a clean success-path registration is not reliable here. |
# | CPU execution path | This file is constrained to run on NPU and should stay NPU-focused. |
# | advanced alias-analysis/tag variants | Not needed to validate the Library constructor and registration interface in this backend-specific test. |

import pytest
import torch
import torch_npu  # noqa: F401

from torch.library import Library


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="torch_npu/NPU is not available in this environment.",
)


def _destroy(*libs):
    for lib in libs:
        lib._destroy()


def test_library_constructor_accepts_valid_kinds_and_rejects_invalid_kind():
    valid = []
    try:
        for kind in ("DEF", "IMPL", "FRAGMENT"):
            lib = Library(f"codex_kind_{kind.lower()}", kind)
            valid.append(lib)
            assert repr(lib).startswith(f"Library(kind={kind}")

        with pytest.raises(ValueError):
            Library("codex_kind_bad", "BAD")
    finally:
        _destroy(*valid)


def test_library_constructor_rejects_reserved_namespace_for_def_and_fragment():
    with pytest.raises(ValueError):
        Library("prim", "DEF")
    with pytest.raises(ValueError):
        Library("prim", "FRAGMENT")


@pytest.mark.parametrize("shape", [(), (0,), (2, 1)])
def test_library_define_and_impl_run_on_npu(shape):
    ns = f"codex_define_{abs(hash(shape))}"
    def_lib = Library(ns, "DEF")
    impl_lib = Library(ns, "IMPL", "NPU")
    try:
        op_name = def_lib.define("foo(Tensor x) -> Tensor")
        assert op_name == "foo"

        def npu_kernel(tensor):
            return tensor.new_empty(tensor.shape)

        impl_lib.impl(op_name, npu_kernel)

        x = torch.ones(shape if shape != () else (), device="npu:0")
        y = torch.ops.__getattr__(ns).foo(x)
        assert y.device.type == "npu"
        assert tuple(y.shape) == tuple(shape)
        assert y.layout == torch.strided
    finally:
        _destroy(def_lib, impl_lib)


def test_library_impl_uses_library_default_dispatch_key():
    ns = "codex_default_dispatch_key"
    def_lib = Library(ns, "DEF")
    impl_lib = Library(ns, "IMPL", "NPU")
    try:
        def_lib.define("baz(Tensor x) -> Tensor")

        def baz_kernel(x):
            return x.new_empty(x.shape)

        impl_lib.impl("baz", baz_kernel)
        out = torch.ops.__getattr__(ns).baz(torch.ones(3, device="npu:0"))
        assert out.device.type == "npu"
        assert tuple(out.shape) == (3,)
    finally:
        _destroy(def_lib, impl_lib)


def test_library_impl_rejects_non_callable_and_duplicate_registration():
    ns = "codex_impl_errors"
    def_lib = Library(ns, "DEF")
    impl_lib = Library(ns, "IMPL", "NPU")
    try:
        def_lib.define("foo(Tensor x) -> Tensor")

        with pytest.raises(TypeError):
            impl_lib.impl("foo", 123)

        def foo_kernel(x):
            return x.new_empty(x.shape)

        impl_lib.impl("foo", foo_kernel)
        with pytest.raises(RuntimeError):
            impl_lib.impl("foo", foo_kernel)
    finally:
        _destroy(def_lib, impl_lib)


@pytest.mark.parametrize("schema", ["foo(Tensor x", "foo(Tensor x) ->", "foo -> Tensor"])
def test_library_define_rejects_malformed_schema(schema):
    lib = Library("codex_bad_schema", "DEF")
    try:
        with pytest.raises(RuntimeError):
            lib.define(schema)
    finally:
        _destroy(lib)


def test_library_fallback_rejects_invalid_usage():
    global_lib = Library("_", "IMPL")
    local_lib = Library("codex_fallback", "IMPL", "NPU")
    try:
        with pytest.raises(AssertionError):
            global_lib.fallback(lambda *args, **kwargs: None)

        with pytest.raises(RuntimeError):
            local_lib.fallback(lambda *args, **kwargs: None, "NPU")
    finally:
        _destroy(global_lib, local_lib)
