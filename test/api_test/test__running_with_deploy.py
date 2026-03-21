# 测试目的: 验证 torch._running_with_deploy 在 NPU 环境下的默认返回值、无状态行为和异常输入处理。
# API 名称: torch._running_with_deploy
# 覆盖维度表:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖无参调用与错误传参 |
# | None / 非 None | 不适用 | 该 API 无入参 |
# | 枚举选项 | 不适用 | 该 API 无枚举参数 |
# | 多类型 | 覆盖 | 覆盖 NPU Tensor 参与的上下文与错误参数类型 |
# | 正常输入 | 覆盖 | 默认返回 False，NPU Tensor 不改变返回值 |
# | 异常输入 | 覆盖 | 传入多余参数触发 TypeError |
# | 边界值和等价类 | 覆盖 | NPU 单元素 Tensor、空 Tensor 作为上下文边界 |
# 未覆盖项及原因:
# - None / 非 None: 该 API 无入参，不存在该维度。
# - 枚举选项: 该 API 无枚举参数。
# - 其他状态分支: 该实现当前固定返回 False，未发现可稳定切换的 deploy 状态。

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


def test_running_with_deploy_default_false_on_npu() -> None:
    _require_npu()

    assert torch._running_with_deploy() is False

    x = torch.ones(1, device=_npu_device(), dtype=torch.float32)
    assert x.device.type == "npu"
    assert torch._running_with_deploy() is False

    empty = torch.empty(0, device=_npu_device(), dtype=torch.float32)
    assert empty.device.type == "npu"
    assert empty.numel() == 0
    assert torch._running_with_deploy() is False


def test_running_with_deploy_rejects_unexpected_args() -> None:
    _require_npu()

    with pytest.raises(TypeError):
        torch._running_with_deploy(True)  # type: ignore[call-arg]
