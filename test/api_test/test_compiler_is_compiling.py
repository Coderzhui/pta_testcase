# 测试目的: 验证 torch.compiler.is_compiling 在 NPU 环境下的默认值、编译态值和错误调用行为。
# API 名称: torch.compiler.is_compiling
# 覆盖维度表:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 该 API 无参数，验证无参调用与错误传参 |
# | None / 非 None | 不适用 | 该 API 无入参 |
# | 枚举选项 | 不适用 | 该 API 无枚举参数 |
# | 多类型 | 不适用 | 该 API 无多类型参数 |
# | 正常输入 | 覆盖 | eager 下返回 False，compile 路径下返回 True |
# | 异常输入 | 覆盖 | 传入多余参数触发 TypeError |
# | 边界值和等价类 | 覆盖 | 普通 NPU Tensor、空 Tensor 作为编译输入边界 |
# 未覆盖项及原因:
# - None / 非 None: 无入参，无法构造该维度。
# - 枚举选项: 无相关参数。
# - 多类型: 无相关参数。

import pytest
import torch

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


def _compile_supported() -> bool:
    if not hasattr(torch, "compile"):
        return False
    return True


def test_is_compiling_default_and_compiled_paths_on_npu() -> None:
    _require_npu()
    assert torch.compiler.is_compiling() is False

    if not _compile_supported():
        pytest.xfail("torch.compile is unavailable in this environment")

    def fn(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        return x + 1, torch.compiler.is_compiling()

    compiled_fn = torch.compile(fn, backend="eager")
    x = torch.ones(4, device=_npu_device(), dtype=torch.float32)

    out, compiling_flag = compiled_fn(x)

    assert out.device.type == "npu"
    assert isinstance(compiling_flag, bool)
    assert compiling_flag is True
    assert torch.compiler.is_compiling() is False


def test_is_compiling_handles_boundary_npu_inputs() -> None:
    _require_npu()

    if not _compile_supported():
        pytest.xfail("torch.compile is unavailable in this environment")

    def fn(x: torch.Tensor) -> bool:
        return torch.compiler.is_compiling()

    compiled_fn = torch.compile(fn, backend="eager")
    empty = torch.empty(0, device=_npu_device(), dtype=torch.float32)

    assert empty.numel() == 0
    assert compiled_fn(empty) is True


def test_is_compiling_rejects_unexpected_args() -> None:
    _require_npu()

    with pytest.raises(TypeError):
        torch.compiler.is_compiling(True)  # type: ignore[call-arg]
