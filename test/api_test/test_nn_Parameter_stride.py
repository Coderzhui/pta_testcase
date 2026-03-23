"""
测试目的：验证 `torch.nn.Parameter.stride` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Parameter.stride`
覆盖的入参维度：
- 参数传参与不传参：覆盖默认调用和错误 self 调用。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖连续与转置后的 stride。
- 正常输入：覆盖 NPU Parameter stride 读取。
- 异常输入：当前实例方法缺少稳定且有意义的接口级异常路径。
- 边界值和等价类：覆盖 2D Parameter。
未覆盖项及原因：
- 未覆盖更复杂 memory format。
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

def test_parameter_stride_on_npu():
    _require_npu()

    contiguous = torch.nn.Parameter(torch.ones(2, 3, device="npu"))
    transposed = torch.nn.Parameter(torch.ones(2, 3, device="npu").t())

    assert contiguous.stride() == (3, 1)
    assert transposed.stride() == (1, 3)
