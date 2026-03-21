# 测试目的: 验证 torch.library.impl 在 NPU 环境下的注册、调用、重载和异常行为。
# API 名称: torch.library.impl
# 覆盖维度表:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖 decorator 形式、直接调用形式和 Legacy Library-first overload |
# | None / 非 None | 覆盖 | 覆盖 func=None 的 decorator 形式与 func 非 None 的直接注册形式 |
# | 枚举选项 | 覆盖 | 覆盖 types=字符串与 types=序列两种主要输入形态 |
# | 多类型 | 覆盖 | 覆盖 NPU kernel 注册与 NPU Tensor 调用，兼顾空 Tensor 边界 |
# | 正常输入 | 覆盖 | NPU Tensor 正常调用、空 Tensor 调用、legacy overload 注册 |
# | 异常输入 | 覆盖 | 重复注册、非 callable func |
# | 边界值和等价类 | 覆盖 | 空 Tensor、单元素 Tensor、重复注册等价类 |
# 未覆盖项及原因:
# - 其他 dispatch key 候选值: 本文件聚焦 NPU 功能测试，仅覆盖稳定可执行的 NPU 注册路径。
# - 传入非法 qualname 类型: 已有更直接的重复注册/非 callable 异常覆盖，避免引入不稳定的 namespace 解析错误差异。

import pytest
import torch
import torch.library

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - import-time environment guard
    torch_npu = None  # type: ignore[assignment]
    _TORCH_NPU_IMPORT_ERROR = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None


def _require_npu() -> None:
    if _TORCH_NPU_IMPORT_ERROR is not None:
        pytest.skip(f"torch_npu import failed: {_TORCH_NPU_IMPORT_ERROR}")
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu backend is unavailable in this environment")
    if not torch.npu.is_available():
        pytest.skip("NPU device is unavailable in this environment")


def _npu_device() -> torch.device:
    _require_npu()
    return torch.device("npu:0")


def test_impl_decorator_without_explicit_lib_registers_npu_kernel() -> None:
    _require_npu()

    namespace = "api_test_library_impl_decorator"
    lib = torch.library.Library(namespace, "DEF")
    lib.define("scale_add(Tensor x, Tensor y, float alpha=1.0) -> Tensor")

    @torch.library.impl(f"{namespace}::scale_add", "npu")
    def scale_add_npu(x, y, alpha: float = 1.0):
        assert x.device.type == "npu"
        assert y.device.type == "npu"
        return x + y * alpha

    x = torch.ones(4, device=_npu_device(), dtype=torch.float32)
    y = torch.full((4,), 2.0, device=_npu_device(), dtype=torch.float32)
    out = torch.ops.api_test_library_impl_decorator.scale_add(x, y, 0.5)

    assert out.device.type == "npu"
    assert out.shape == x.shape
    assert out.dtype == torch.float32

    empty = torch.empty(0, device=_npu_device(), dtype=torch.float32)
    empty_out = torch.ops.api_test_library_impl_decorator.scale_add(empty, empty, 1.0)

    assert empty_out.device.type == "npu"
    assert empty_out.numel() == 0


def test_impl_direct_call_with_explicit_func_and_type_sequence() -> None:
    _require_npu()

    namespace = "api_test_library_impl_direct"
    lib = torch.library.Library(namespace, "DEF")
    lib.define("shift(Tensor x) -> Tensor")

    def shift_npu(x):
        assert x.device.type == "npu"
        return x + 1

    result = torch.library.impl(
        f"{namespace}::shift",
        ("npu",),
        shift_npu,
        lib=lib,
    )
    assert result is None

    x = torch.ones(1, device=_npu_device(), dtype=torch.float32)
    out = torch.ops.api_test_library_impl_direct.shift(x)

    assert out.device.type == "npu"
    assert out.shape == x.shape
    assert out.dtype == torch.float32


def test_impl_legacy_library_first_overload_registers_npu_kernel() -> None:
    _require_npu()

    namespace = "api_test_library_impl_legacy"
    lib = torch.library.Library(namespace, "DEF")
    lib.define("identity(Tensor x) -> Tensor")

    @torch.library.impl(lib, "identity", "NPU")
    def identity_npu(x):
        assert x.device.type == "npu"
        return x

    x = torch.empty(0, device=_npu_device(), dtype=torch.float32)
    out = torch.ops.api_test_library_impl_legacy.identity(x)

    assert out.device.type == "npu"
    assert out.numel() == 0


def test_impl_rejects_duplicate_registration() -> None:
    _require_npu()

    namespace = "api_test_library_impl_duplicate"
    lib = torch.library.Library(namespace, "DEF")
    lib.define("echo(Tensor x) -> Tensor")

    @torch.library.impl(f"{namespace}::echo", "npu", lib=lib)
    def echo_npu(x):
        return x

    with pytest.raises(RuntimeError):
        @torch.library.impl(f"{namespace}::echo", "npu", lib=lib)
        def echo_npu_again(x):
            return x


def test_impl_rejects_non_callable_func() -> None:
    _require_npu()

    namespace = "api_test_library_impl_non_callable"
    lib = torch.library.Library(namespace, "DEF")
    lib.define("bad(Tensor x) -> Tensor")

    with pytest.raises(TypeError):
        torch.library.impl(f"{namespace}::bad", "npu", 123, lib=lib)
