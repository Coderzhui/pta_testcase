"""
测试目的：验证 `torch._running_with_deploy` 在 NPU 环境下的完整功能行为。
API 名称：`torch._running_with_deploy`

覆盖的入参维度：
- 参数传参与不传参：已覆盖无参调用和多余参数异常。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：已覆盖布尔返回值。
- 正常输入：已覆盖正常调用和返回值验证。
- 异常输入：已覆盖多余参数异常。
- 边界值和等价类：已覆盖默认返回值（False）。

已补充覆盖项：
- ✓ 正常调用返回值类型验证
- ✓ 默认返回值验证

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


def test_flag_like_api_returns_bool_in_npu_eager_mode():
    """
    测试用例：验证 API 返回布尔类型
    覆盖场景：
    - 在 NPU eager 模式下调用 API
    - 验证返回值类型为 bool
    """
    _require_npu()

    result = torch._running_with_deploy()
    assert isinstance(result, bool)


def test_running_with_deploy_returns_false_by_default():
    """
    测试用例：验证 API 默认返回 False
    覆盖场景：
    - 在非部署环境下调用
    - 验证返回值为 False
    """
    _require_npu()

    result = torch._running_with_deploy()
    assert result is False


def test_running_with_deploy_rejects_extra_args():
    """
    测试用例：验证 API 拒绝多余参数
    覆盖场景：
    - 传入非法参数
    - 验证抛出 TypeError 异常
    """
    _require_npu()

    with pytest.raises(TypeError):
        torch._running_with_deploy("bad")
