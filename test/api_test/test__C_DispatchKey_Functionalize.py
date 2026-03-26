"""
测试目的：验证 `torch._C.DispatchKey.Functionalize` 在 NPU 环境下的完整功能行为。
API 名称：`torch._C.DispatchKey.Functionalize`

覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为枚举值访问。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：已覆盖多个枚举对象（Functionalize、CPU、CUDA、AutogradCPU、PrivateUse1）和整数值。
- 正常输入：已覆盖 NPU 上下文中的枚举读取。
- 异常输入：已覆盖缺失成员访问。
- 边界值和等价类：已覆盖多个枚举项和枚举比较。

已补充覆盖项：
- ✓ 多个 DispatchKey 枚举值测试
- ✓ 枚举值比较操作

未覆盖项及原因：
- 无，已满足 API 功能一致性标准。
"""

import contextlib
import io
import logging

import pytest

import torch
import torch_npu  # noqa: F401


def _require_npu():
    """检查 NPU 是否可用，不可用则跳过测试。"""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("当前环境未检测到可用 NPU，无法验证该 API 的 NPU 行为。")


def test_dispatchkey_functionalize_is_accessible():
    """
    测试用例：验证 Functionalize 枚举值可访问
    覆盖场景：
    - 在 NPU 上下文中访问枚举
    - 验证枚举类型正确
    - 验证可转换为整数
    - 验证整数值非负
    """
    _require_npu()

    tensor = torch.ones(1, device="npu")
    key = torch._C.DispatchKey.Functionalize

    assert tensor.device.type == "npu"
    assert isinstance(key, torch.DispatchKey)
    assert int(key) >= 0


def test_dispatchkey_missing_member_raises():
    """
    测试用例：验证访问不存在的枚举成员抛出异常
    覆盖场景：
    - 访问不存在的枚举成员
    - 验证抛出 AttributeError
    """
    _require_npu()

    with pytest.raises(AttributeError):
        getattr(torch._C.DispatchKey, "NotARealKey")


@pytest.mark.parametrize("key_name", ["CPU", "CUDA", "AutogradCPU", "PrivateUse1"])
def test_dispatchkey_multiple_keys_accessible(key_name):
    """
    测试用例：验证多个 DispatchKey 枚举值可访问
    覆盖场景：
    - 参数化测试多个枚举值
    - CPU：CPU 设备 dispatch key
    - CUDA：CUDA 设备 dispatch key
    - AutogradCPU：CPU autograd dispatch key
    - PrivateUse1：自定义设备 dispatch key（NPU 使用）
    - 验证类型和整数转换
    """
    _require_npu()

    key = getattr(torch._C.DispatchKey, key_name)
    assert isinstance(key, torch.DispatchKey)
    assert int(key) >= 0


def test_dispatchkey_comparison():
    """
    测试用例：验证枚举值比较操作
    覆盖场景：
    - 相同枚举值相等
    - 不同枚举值不相等
    - 测试 == 和 != 运算符
    """
    _require_npu()

    key1 = torch._C.DispatchKey.Functionalize
    key2 = torch._C.DispatchKey.Functionalize
    key3 = torch._C.DispatchKey.CPU

    assert key1 == key2
    assert key1 != key3
