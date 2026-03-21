# 测试目的: 验证 torch.utils._pytree.tree_unflatten 在 NPU Tensor 叶子场景下的重建、边界和异常行为。
# API 名称: torch.utils._pytree.tree_unflatten
# 覆盖维度表:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖标准顺序调用；错误顺序调用会抛 TypeError |
# | None / 非 None | 覆盖 | 覆盖 None 叶子与非 None NPU Tensor 叶子 |
# | 枚举选项 | 不适用 | 该 API 无枚举参数 |
# | 多类型 | 覆盖 | 覆盖 list / tuple / dict 组合与 NPU Tensor、None 混合叶子 |
# | 正常输入 | 覆盖 | 嵌套 pytree、空结构、生成器 leaves 输入 |
# | 异常输入 | 覆盖 | leaves 数量不匹配、treespec 类型非法 |
# | 边界值和等价类 | 覆盖 | 空 pytree、空 Tensor、None 叶子、生成器叶子 |
# 未覆盖项及原因:
# - 枚举选项: 该 API 没有枚举型参数。
# - 其他容器变体: 本文件已覆盖 list/tuple/dict 等主要结构，未继续扩展更多等价容器以避免冗余。
# - treespec/leaves 交换调用: 当前运行时实现不支持该兼容分支，会直接抛 TypeError。

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - import-time environment guard
    torch_npu = None  # type: ignore[assignment]
    _TORCH_NPU_IMPORT_ERROR = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None

try:
    from torch.utils import _pytree as pytree
except Exception as exc:  # pragma: no cover - optional dependency/environment guard
    pytree = None  # type: ignore[assignment]
    _PYTREE_IMPORT_ERROR = exc
else:
    _PYTREE_IMPORT_ERROR = None


def _require_env() -> None:
    if _TORCH_NPU_IMPORT_ERROR is not None:
        pytest.skip(f"torch_npu import failed: {_TORCH_NPU_IMPORT_ERROR}")
    if _PYTREE_IMPORT_ERROR is not None:
        pytest.skip(f"torch.utils._pytree import failed: {_PYTREE_IMPORT_ERROR}")
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu backend is unavailable in this environment")
    if not torch.npu.is_available():
        pytest.skip("NPU device is unavailable in this environment")


def _npu_device() -> torch.device:
    _require_env()
    return torch.device("npu:0")


def test_tree_unflatten_reconstructs_nested_npu_pytree() -> None:
    _require_env()

    tree = [
        torch.ones(2, device=_npu_device(), dtype=torch.float32),
        (None, torch.empty(0, device=_npu_device(), dtype=torch.float32)),
        {"right": torch.zeros(1, device=_npu_device(), dtype=torch.float32)},
    ]
    leaves, spec = pytree.tree_flatten(tree)

    rebuilt = pytree.tree_unflatten((leaf for leaf in leaves), spec)

    assert isinstance(rebuilt, list)
    assert len(rebuilt) == 3
    assert rebuilt[0].device.type == "npu"
    assert rebuilt[0].shape == torch.Size([2])
    assert rebuilt[1][0] is None
    assert rebuilt[1][1].device.type == "npu"
    assert rebuilt[1][1].numel() == 0
    assert rebuilt[2]["right"].device.type == "npu"
    assert rebuilt[2]["right"].shape == torch.Size([1])


def test_tree_unflatten_supports_empty_structure_and_rejects_swapped_arguments() -> None:
    _require_env()

    empty_tree = {"left": [], "right": ()}
    leaves, spec = pytree.tree_flatten(empty_tree)

    rebuilt = pytree.tree_unflatten(leaves, spec)

    assert rebuilt == empty_tree
    assert pytree.tree_unflatten([], pytree.tree_flatten([])[1]) == []

    with pytest.raises(TypeError, match="Expected `treespec`"):
        pytree.tree_unflatten(spec, leaves)


def test_tree_unflatten_raises_on_mismatch_and_invalid_spec() -> None:
    _require_env()

    tree = [torch.ones(1, device=_npu_device(), dtype=torch.float32), None]
    leaves, spec = pytree.tree_flatten(tree)

    with pytest.raises(ValueError, match="leaves.*length"):
        pytree.tree_unflatten(leaves[:-1], spec)

    with pytest.raises(TypeError, match="Expected `treespec`"):
        pytree.tree_unflatten([], 123)
