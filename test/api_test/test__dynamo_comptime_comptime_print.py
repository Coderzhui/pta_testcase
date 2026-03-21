# Test purpose: verify torch._dynamo.comptime.comptime.print behavior on NPU
# under TorchDynamo compilation, including compile-time printing, boundary
# tensor shapes, and reliable interface errors.
# API name: torch._dynamo.comptime.comptime.print
#
# Covered parameter dimensions:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖显式传参与缺参 TypeError |
# | None / 非 None | 覆盖 | 打印 None 与非 None Tensor 共同覆盖 |
# | 枚举选项 | 不适用 | 该 API 无枚举参数 |
# | 多类型 | 覆盖 | 覆盖 Tensor、tuple、None、int |
# | 正常输入 | 覆盖 | NPU Tensor 在 torch.compile 中触发 comptime print |
# | 异常输入 | 覆盖 | 缺参、unsupported keyword file 触发 TypeError |
# | 边界值和等价类 | 覆盖 | scalar Tensor、empty Tensor、tuple payload 边界 |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | 枚举分支 | not applicable because the API exposes no enum-style parameters |
# | 其他 Python 复杂对象 | not covered because representative Tensor / tuple / None / int cases already exercise the printer path |
# | 纯运行时直接调用行为 | not covered because the intended API path is inside TorchDynamo compilation |

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
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


def test_comptime_print_emits_compile_time_and_runtime_output_on_npu(capsys) -> None:
    _require_npu()

    from torch._dynamo.comptime import comptime

    def fn(x):
        comptime.print((x, None, 0))
        return x

    result = torch.compile(fn, backend="eager")(torch.ones(1, device=_npu_device()))
    captured = capsys.readouterr().out

    assert result.device.type == "npu"
    assert "(FakeTensor(..., device='npu:0', size=(1,)), None, 0)" in captured
    assert "(tensor([1.], device='npu:0'), None, 0)" in captured


def test_comptime_print_boundary_shapes_on_npu(capsys) -> None:
    _require_npu()

    from torch._dynamo.comptime import comptime

    def scalar_fn(x):
        comptime.print(x)
        return x

    def empty_fn(x):
        comptime.print(x)
        return x

    scalar_result = torch.compile(scalar_fn, backend="eager")(torch.tensor(3.0, device=_npu_device()))
    empty_result = torch.compile(empty_fn, backend="eager")(torch.empty(0, device=_npu_device()))
    captured = capsys.readouterr().out

    assert scalar_result.device.type == "npu"
    assert empty_result.device.type == "npu"
    assert "FakeTensor(..., device='npu:0', size=())" in captured
    assert "FakeTensor(..., device='npu:0', size=(0,))" in captured
    assert "tensor(3., device='npu:0')" in captured
    assert "tensor([], device='npu:0')" in captured


def test_comptime_print_rejects_missing_argument_and_unsupported_keyword_on_npu() -> None:
    _require_npu()

    from torch._dynamo.comptime import comptime

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        def missing_arg_fn(x):
            comptime.print()
            return x

        torch.compile(missing_arg_fn, backend="eager")(torch.ones(1, device=_npu_device()))

    with pytest.raises(TypeError, match="unexpected keyword argument 'file'"):
        def file_kw_fn(x):
            comptime.print(x, file=123)
            return x

        torch.compile(file_kw_fn, backend="eager")(torch.ones(1, device=_npu_device()))
