"""
测试目的：验证 `torch.utils._pytree.tree_map` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.utils._pytree.tree_map`
覆盖的入参维度：
- 参数传参与不传参：覆盖单树与默认 `is_leaf`。
- 参数为 None / 非 None：覆盖结构中的 `None`。
- 枚举/多类型：覆盖 Tensor/int/None。
- 正常输入：覆盖 NPU Tensor 映射。
- 异常输入：覆盖不可调用 mapper。
- 边界值和等价类：覆盖保持 `None` 不变。
未覆盖项及原因：
- 未覆盖多树 zip 映射场景。
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

def test_tree_map_transforms_nested_npu_tensors():
    _require_npu()

    tree = {"a": [torch.ones(1, device="npu"), 2], "b": (None, torch.zeros(2, device="npu"))}
    mapped = torch.utils._pytree.tree_map(
        lambda x: x + 1 if isinstance(x, torch.Tensor) else (x + 1 if isinstance(x, int) else x),
        tree,
    )

    assert mapped["a"][0].device.type == "npu"
    assert mapped["a"][1] == 3
    assert mapped["b"][0] is None
    assert mapped["b"][1].device.type == "npu"


def test_tree_map_invalid_mapper_raises():
    _require_npu()

    with pytest.raises(TypeError):
        torch.utils._pytree.tree_map(None, [1, 2])
