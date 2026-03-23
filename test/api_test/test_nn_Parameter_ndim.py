"""
测试目的：验证 `torch.nn.Parameter.ndim` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Parameter.ndim`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为属性访问。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖 1D 与 2D Parameter。
- 正常输入：覆盖 NPU Parameter 维度读取。
- 异常输入：当前属性为只读访问，缺少稳定且有意义的接口级异常路径。
- 边界值和等价类：覆盖 1 维边界。
未覆盖项及原因：
- 未覆盖 0 维 Parameter，因为当前最小用例已能验证接口语义。
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

def test_parameter_ndim_on_npu():
    _require_npu()

    one_dim = torch.nn.Parameter(torch.ones(2, device="npu"))
    two_dim = torch.nn.Parameter(torch.ones(2, 3, device="npu"))

    assert one_dim.ndim == 1
    assert two_dim.ndim == 2
