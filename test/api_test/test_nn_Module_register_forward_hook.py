# 测试目的: 验证 torch.nn.Module.register_forward_hook 在 NPU 环境下的注册顺序、kwargs 传递、always_call 和异常行为。
# API 名称: torch.nn.Module.register_forward_hook
# 覆盖维度表:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖 hook 注册、prepend=False/True、with_kwargs=False/True、always_call=False/True |
# | None / 非 None | 不适用 | 该 API 无 None 型参数设计点 |
# | 枚举选项 | 不适用 | 该 API 无枚举参数 |
# | 多类型 | 覆盖 | 覆盖 hook callable、非 callable 异常与 NPU Tensor 输入 |
# | 正常输入 | 覆盖 | 正常 forward、kwargs forward、空 Tensor 输入 |
# | 异常输入 | 覆盖 | non-callable hook、forward 抛错时 always_call 回调 |
# | 边界值和等价类 | 覆盖 | 空 Tensor、hook 顺序边界、handle.remove() 边界 |
# 未覆盖项及原因:
# - None / 非 None: 该 API 没有自然的 None 参数分支。
# - 枚举选项: API 只提供布尔开关，不存在枚举参数。
# - 其它 hook 体系（例如全局 hook）: 这些属于不同 API，不在本文件职责范围内。

import pytest
import torch
import torch.nn as nn

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


class _ScaleModule(nn.Module):
    def forward(self, x, scale=1.0):
        return x * scale


class _FailingModule(nn.Module):
    def forward(self, x):
        _ = x + 1
        raise RuntimeError("forward failed intentionally")


def test_register_forward_hook_prepend_and_remove_on_npu() -> None:
    _require_npu()

    module = _ScaleModule()
    call_order: list[str] = []

    def hook_a(mod, args, output):
        call_order.append("a")
        assert mod is module
        assert len(args) == 1
        assert args[0].device.type == "npu"
        assert output.device.type == "npu"
        return output

    def hook_b(mod, args, output):
        call_order.append("b")
        assert mod is module
        assert len(args) == 1
        assert args[0].device.type == "npu"
        assert output.device.type == "npu"
        return output

    handle_a = module.register_forward_hook(hook_a)
    handle_b = module.register_forward_hook(hook_b, prepend=True)

    x = torch.empty(0, device=_npu_device(), dtype=torch.float32)
    out = module(x)

    assert out.device.type == "npu"
    assert out.numel() == 0
    assert call_order == ["b", "a"]

    handle_b.remove()
    call_order.clear()
    out_again = module(x)

    assert out_again.device.type == "npu"
    assert out_again.numel() == 0
    assert call_order == ["a"]

    handle_a.remove()


def test_register_forward_hook_with_kwargs_and_always_call_on_npu() -> None:
    _require_npu()

    module = _ScaleModule()
    kwargs_seen: list[dict[str, object]] = []

    def hook_with_kwargs(mod, args, kwargs, output):
        assert mod is module
        assert len(args) == 1
        assert args[0].device.type == "npu"
        assert output.device.type == "npu"
        kwargs_seen.append(dict(kwargs))
        return output

    module.register_forward_hook(hook_with_kwargs, with_kwargs=True)

    x = torch.ones(1, device=_npu_device(), dtype=torch.float32)
    out = module(x, scale=3.0)

    assert out.device.type == "npu"
    assert kwargs_seen == [{"scale": 3.0}]

    failing_module = _FailingModule()
    always_called: list[str] = []

    def always_hook(mod, args, output):
        assert mod is failing_module
        assert len(args) == 1
        assert args[0].device.type == "npu"
        assert output is None
        always_called.append("called")
        return None

    failing_module.register_forward_hook(always_hook, always_call=True)

    with pytest.raises(RuntimeError, match="forward failed intentionally"):
        failing_module(torch.ones(1, device=_npu_device(), dtype=torch.float32))

    assert always_called == ["called"]


def test_register_forward_hook_rejects_non_callable_hook_on_call_on_npu() -> None:
    _require_npu()

    module = _ScaleModule()
    module.register_forward_hook(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        module(torch.ones(1, device=_npu_device(), dtype=torch.float32))
