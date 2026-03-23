"""
测试目的：验证 `torch._C.DispatchKey.Functionalize` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch._C.DispatchKey.Functionalize`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为枚举值访问。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖枚举对象和整数值。
- 正常输入：覆盖 NPU 上下文中的枚举读取。
- 异常输入：覆盖缺失成员访问。
- 边界值和等价类：覆盖单枚举项。
未覆盖项及原因：
- 未覆盖完整 DispatchKey 枚举空间。
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

def test_dispatchkey_functionalize_is_accessible():
    _require_npu()

    tensor = torch.ones(1, device="npu")
    key = torch._C.DispatchKey.Functionalize

    assert tensor.device.type == "npu"
    assert isinstance(key, torch.DispatchKey)
    assert int(key) >= 0


def test_dispatchkey_missing_member_raises():
    _require_npu()

    with pytest.raises(AttributeError):
        getattr(torch._C.DispatchKey, "NotARealKey")
