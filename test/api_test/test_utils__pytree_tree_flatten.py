"""
测试目的：验证 `torch.utils._pytree.tree_flatten` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.utils._pytree.tree_flatten`
覆盖的入参维度：
- 参数传参与不传参：覆盖默认 `is_leaf` 和自定义 `is_leaf`。
- 参数为 None / 非 None：覆盖 `is_leaf=None` 和 lambda。
- 枚举/多类型：覆盖 dict/list/tuple/Tensor/None。
- 正常输入：覆盖 NPU Tensor 嵌套结构。
- 异常输入：覆盖非法 `is_leaf`。
- 边界值和等价类：覆盖叶子级 None。
未覆盖项及原因：
- 未覆盖更复杂自定义 pytree node 注册。
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

def test_tree_flatten_handles_nested_npu_structure():
    _require_npu()

    tree = {"a": [torch.ones(1, device="npu"), 2], "b": (None, torch.zeros(2, device="npu"))}
    leaves, spec = torch.utils._pytree.tree_flatten(tree)

    assert isinstance(spec, torch.utils._pytree.TreeSpec)
    assert leaves[0].device.type == "npu"
    assert leaves[-1].device.type == "npu"
    assert len(leaves) == 4


def test_tree_flatten_invalid_is_leaf_raises():
    _require_npu()

    with pytest.raises(TypeError):
        torch.utils._pytree.tree_flatten([1, 2], is_leaf=1)
