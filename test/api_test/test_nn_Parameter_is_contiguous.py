"""
测试目的：验证 `torch.nn.Parameter.is_contiguous` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Parameter.is_contiguous`
覆盖的入参维度：
- 参数传参与不传参：覆盖默认调用和错误 self 调用。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖连续与非连续 NPU Parameter。
- 正常输入：覆盖 contiguous 状态读取。
- 异常输入：当前实例方法缺少稳定且有意义的接口级异常路径。
- 边界值和等价类：覆盖转置得到的非连续张量。
未覆盖项及原因：
- 未覆盖 memory_format 变体。
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

def test_parameter_is_contiguous_on_npu():
    _require_npu()

    contiguous = torch.nn.Parameter(torch.ones(2, 3, device="npu"))
    non_contiguous = torch.nn.Parameter(torch.ones(2, 3, device="npu").t())

    assert contiguous.is_contiguous() is True
    assert non_contiguous.is_contiguous() is False
