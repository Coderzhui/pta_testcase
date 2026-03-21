# 测试目的: 验证 torch.fx.node.has_side_effect 在 NPU 环境下对 FX 图死代码消除的保留效果、边界输入和接口错误行为。
# API 名称: torch.fx.node.has_side_effect
# 覆盖维度表:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖传入 callable 的正常注册与缺失参数的错误调用 |
# | None / 非 None | 不适用 | 该 API 需要 callable，不存在 None 分支设计点 |
# | 枚举选项 | 不适用 | 该 API 无枚举参数 |
# | 多类型 | 覆盖 | 覆盖普通 Tensor、空 NPU Tensor、FX node target callable |
# | 正常输入 | 覆盖 | 侧效函数被保留，纯函数被 DCE 删除 |
# | 异常输入 | 覆盖 | 缺失必需参数触发 TypeError |
# | 边界值和等价类 | 覆盖 | 空 Tensor、单元素 Tensor、无用户节点等价类 |
# 未覆盖项及原因:
# - None / 非 None: 该 API 的第一个参数必须是 callable，没有自然的 None 分支。
# - 枚举选项: 该 API 不存在枚举型参数。
# - 非 callable 参数的类型校验: 该函数本身不做输入类型校验，无法构造稳定的错误分支。

import pytest
import torch
import torch.fx as fx
from torch.fx.node import has_side_effect

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


_SIDE_EFFECT_LOG: list[tuple[str, int]] = []


def _pure_passthrough(x: torch.Tensor) -> torch.Tensor:
    return x


@has_side_effect
def _record_npu_side_effect(x: torch.Tensor) -> torch.Tensor:
    assert x.device.type == "npu"
    _SIDE_EFFECT_LOG.append((x.device.type, x.numel()))
    return x


def _make_graph_module() -> tuple[fx.GraphModule, fx.Node, fx.Node]:
    graph = fx.Graph()
    x = graph.placeholder("x")
    pure_node = graph.call_function(_pure_passthrough, (x,))
    side_node = graph.call_function(_record_npu_side_effect, (x,))
    graph.output(x)
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    return gm, pure_node, side_node


def test_has_side_effect_preserves_side_effectful_node_on_npu() -> None:
    _require_npu()
    _SIDE_EFFECT_LOG.clear()

    assert has_side_effect(_record_npu_side_effect) is _record_npu_side_effect

    gm, pure_node, side_node = _make_graph_module()
    x = torch.ones(1, device=_npu_device(), dtype=torch.float32)

    gm.graph.eliminate_dead_code()
    gm.recompile()

    node_targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert _pure_passthrough not in node_targets
    assert _record_npu_side_effect in node_targets
    assert pure_node not in gm.graph.nodes
    assert side_node in gm.graph.nodes

    out = gm(x)

    assert out.device.type == "npu"
    assert out.shape == torch.Size([1])
    assert _SIDE_EFFECT_LOG == [("npu", 1)]


def test_has_side_effect_boundary_empty_npu_tensor() -> None:
    _require_npu()
    _SIDE_EFFECT_LOG.clear()

    gm, _, _ = _make_graph_module()
    x = torch.empty(0, device=_npu_device(), dtype=torch.float32)

    gm.graph.eliminate_dead_code()
    gm.recompile()

    out = gm(x)

    assert out.device.type == "npu"
    assert out.numel() == 0
    assert _SIDE_EFFECT_LOG == [("npu", 0)]


def test_has_side_effect_requires_callable_argument() -> None:
    _require_npu()

    with pytest.raises(TypeError):
        has_side_effect()  # type: ignore[call-arg]
