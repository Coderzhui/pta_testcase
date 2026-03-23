"""
测试目的：验证 `torch.utils._pytree.tree_unflatten` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.utils._pytree.tree_unflatten`
覆盖的入参维度：
- 参数传参与不传参：覆盖合法 leaves/spec 和非法叶子数量。
- 参数为 None / 非 None：覆盖 leaves 中的 `None`。
- 枚举/多类型：覆盖 dict/list/tuple/Tensor。
- 正常输入：覆盖 NPU Tensor 结构重建。
- 异常输入：覆盖 leaves 数量不匹配。
- 边界值和等价类：覆盖最小一层嵌套。
未覆盖项及原因：
- 未覆盖自定义 pytree node 的反序列化。
"""

import contextlib
import io
import logging

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")

def test_tree_unflatten_rebuilds_nested_npu_structure():
    _require_npu()

    tree = {"a": [torch.ones(1, device="npu")], "b": (3, None)}
    leaves, spec = torch.utils._pytree.tree_flatten(tree)
    restored = torch.utils._pytree.tree_unflatten(leaves, spec)

    assert restored["a"][0].device.type == "npu"
    assert restored["b"] == (3, None)


def test_tree_unflatten_mismatched_leaves_raises():
    _require_npu()

    leaves, spec = torch.utils._pytree.tree_flatten([1, 2])
    with pytest.raises(ValueError):
        torch.utils._pytree.tree_unflatten(leaves[:1], spec)
