"""
测试目的：验证 `torch.nn.Parameter.size` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Parameter.size`
覆盖的入参维度：
- 参数传参与不传参：覆盖无参返回完整形状和单维索引。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖整体 `torch.Size` 与整型维度索引。
- 正常输入：覆盖 NPU Parameter 大小读取。
- 异常输入：覆盖非法维度类型和越界维度。
- 边界值和等价类：覆盖 2D Parameter。
未覆盖项及原因：
- 未覆盖更高维形状，因为 2D 已覆盖接口行为。
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

def test_parameter_size_on_npu():
    _require_npu()

    parameter = torch.nn.Parameter(torch.ones(2, 3, device="npu"))
    assert parameter.size() == torch.Size([2, 3])
    assert parameter.size(1) == 3


@pytest.mark.parametrize("dim", ["bad", 5])
def test_parameter_size_invalid_dim_raises(dim):
    _require_npu()

    parameter = torch.nn.Parameter(torch.ones(2, 3, device="npu"))
    with pytest.raises((RuntimeError, IndexError, TypeError)):
        parameter.size(dim)
