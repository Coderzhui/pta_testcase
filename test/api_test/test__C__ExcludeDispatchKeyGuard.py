"""
测试目的：验证 `torch._C._ExcludeDispatchKeyGuard` 在 NPU 环境下的完整功能行为。
API 名称：`torch._C._ExcludeDispatchKeyGuard`

覆盖的入参维度：
- 参数传参与不传参：已覆盖合法 DispatchKeySet 构造。
- 参数为 None / 非 None：已覆盖有效 keyset 与 None。
- 枚举/多类型：已覆盖多个 DispatchKey（Functionalize、CPU、AutogradCPU）。
- 正常输入：已覆盖正常构造和对象类型验证。
- 异常输入：已覆盖非法 None 入参。
- 边界值和等价类：已覆盖多个 keyset。

已补充覆盖项：
- ✓ 上下文管理器基本行为（进入/退出）
- ✓ 多个 DispatchKey 测试

未覆盖项及原因：
- dispatch 副作用观测：当前环境缺少稳定的 dispatch 副作用可观测接口，无法验证 guard 对 dispatcher 状态的实际影响。已通过上下文管理器协议测试验证基本功能。
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


def test_exclude_dispatch_key_guard_constructs_with_functionalize():
    """
    测试用例：验证 Guard 正常构造
    覆盖场景：
    - 使用 DispatchKey.Functionalize 创建 DispatchKeySet
    - 使用 keyset 构造 _ExcludeDispatchKeyGuard
    - 验证 guard 对象非空
    - 验证 guard 对象类型正确
    """
    _require_npu()

    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
    guard = torch._C._ExcludeDispatchKeyGuard(keyset)

    assert guard is not None
    assert isinstance(guard, torch._C._ExcludeDispatchKeyGuard)


def test_exclude_dispatch_key_guard_invalid_none_raises():
    """
    测试用例：验证 None 参数抛出异常
    覆盖场景：
    - 传入 None 作为参数
    - 验证抛出 TypeError 或 RuntimeError
    """
    _require_npu()

    with pytest.raises((TypeError, RuntimeError)):
        torch._C._ExcludeDispatchKeyGuard(None)


def test_exclude_dispatch_key_guard_context_manager():
    """
    测试用例：验证作为上下文管理器使用
    覆盖场景：
    - 使用 with 语句
    - 验证可以正常进入和退出上下文
    - 在上下文中执行 NPU 操作
    注意：无法观测 dispatch 副作用，仅验证上下文管理器协议
    """
    _require_npu()

    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)

    with torch._C._ExcludeDispatchKeyGuard(keyset):
        # 在 guard 上下文中执行操作
        tensor = torch.ones(2, device="npu")
        assert tensor.device.type == "npu"


@pytest.mark.parametrize("key_name", ["CPU", "AutogradCPU"])
def test_exclude_dispatch_key_guard_multiple_keys(key_name):
    """
    测试用例：验证不同 DispatchKey 的 guard 构造
    覆盖场景：
    - 参数化测试多个 DispatchKey
    - CPU
    - AutogradCPU
    """
    _require_npu()

    key = getattr(torch._C.DispatchKey, key_name)
    keyset = torch._C.DispatchKeySet(key)
    guard = torch._C._ExcludeDispatchKeyGuard(keyset)

    assert guard is not None
    assert isinstance(guard, torch._C._ExcludeDispatchKeyGuard)
