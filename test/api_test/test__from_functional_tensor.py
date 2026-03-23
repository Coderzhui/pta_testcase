"""
测试目的：验证 `torch._from_functional_tensor` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch._from_functional_tensor`
覆盖的入参维度：
- 参数传参与不传参：覆盖 functional tensor 正常路径和普通 Tensor 异常路径。
- 参数为 None / 非 None：覆盖合法 functional Tensor，以及非法 `None`/普通 Tensor。
- 枚举/多类型：不适用。
- 正常输入：覆盖 NPU functional tensor 还原。
- 异常输入：覆盖 `None` 和普通 Tensor。
- 边界值和等价类：覆盖最小一维张量。
未覆盖项及原因：
- 若环境缺少 `_to_functional_tensor` helper，则子测试会 skip。
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

def test_from_functional_tensor_round_trip_on_npu():
    _require_npu()
    if not hasattr(torch, "_to_functional_tensor"):
        pytest.skip("当前环境缺少 functional tensor helper，无法稳定验证 torch._from_functional_tensor。")

    base = torch.ones(2, device="npu")
    functional = torch._to_functional_tensor(base)
    restored = torch._from_functional_tensor(functional)

    assert isinstance(restored, torch.Tensor)
    assert restored.device.type == "npu"
    assert restored is base


@pytest.mark.parametrize("case", ["none", "plain_tensor"])
def test_from_functional_tensor_invalid_inputs_raise(case):
    _require_npu()
    value = None if case == "none" else torch.ones(1, device="npu")

    with pytest.raises((RuntimeError, TypeError)):
        torch._from_functional_tensor(value)
