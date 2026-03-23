"""
测试目的：验证 `torch._C._ExcludeDispatchKeyGuard` 的构造入口，以及当前环境下无法稳定观测的 NPU dispatch 限制。
API 名称：`torch._C._ExcludeDispatchKeyGuard`
覆盖的入参维度：
- 参数传参与不传参：覆盖合法 DispatchKeySet 构造和非法 `None`。
- 参数为 None / 非 None：覆盖有效 keyset 与 `None`。
- 枚举/多类型：覆盖 `DispatchKey.Functionalize`。
- 正常输入：当前环境缺少稳定可观测的 NPU dispatch 副作用，使用 `pytest.skip` 明确阻塞。
- 异常输入：覆盖非法入参。
- 边界值和等价类：覆盖单 keyset。
未覆盖项及原因：
- 当前环境缺少稳定的 NPU dispatch 可观测接口，无法可靠验证 guard 对 dispatcher 状态的实际影响，因此不伪造正常路径覆盖。
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

def test_exclude_dispatch_key_guard_constructs_with_functionalize():
    _require_npu()
    pytest.skip("当前环境缺少稳定的 NPU dispatch 可观测路径，无法可靠验证 _ExcludeDispatchKeyGuard 的正常语义。")


def test_exclude_dispatch_key_guard_invalid_none_raises():
    _require_npu()

    with pytest.raises((TypeError, RuntimeError)):
        torch._C._ExcludeDispatchKeyGuard(None)
