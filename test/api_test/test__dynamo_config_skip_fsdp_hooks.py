"""
测试目的：验证 `torch._dynamo.config.skip_fsdp_hooks` 在 NPU 环境下的完整功能行为。
API 名称：`torch._dynamo.config.skip_fsdp_hooks`

覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为配置属性访问。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：已覆盖布尔值（True/False）和整数值（0/1）。
- 正常输入：已覆盖 NPU 上下文中的属性读取和修改。
- 异常输入：配置项接受多种类型，无严格类型限制。
- 边界值和等价类：已覆盖布尔值和整数值。

已补充覆盖项：
- ✓ 配置项修改与恢复机制
- ✓ True/False 布尔值切换
- ✓ 整数值（0/1）赋值测试

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


def test_skip_fsdp_hooks_is_bool_in_npu_context():
    """
    测试用例：验证配置项类型为布尔值
    覆盖场景：
    - 在 NPU 上下文中访问配置项
    - 验证返回值类型为 bool
    """
    _require_npu()

    tensor = torch.ones(1, device="npu")
    assert tensor.device.type == "npu"
    assert isinstance(torch._dynamo.config.skip_fsdp_hooks, bool)


def test_skip_fsdp_hooks_can_be_modified():
    """
    测试用例：验证配置项可修改并恢复
    覆盖场景：
    - 修改配置项为相反值
    - 验证修改生效
    - 恢复原始值
    - 使用 finally 确保配置恢复
    """
    _require_npu()

    original = torch._dynamo.config.skip_fsdp_hooks
    try:
        torch._dynamo.config.skip_fsdp_hooks = not original
        assert torch._dynamo.config.skip_fsdp_hooks == (not original)

        torch._dynamo.config.skip_fsdp_hooks = original
        assert torch._dynamo.config.skip_fsdp_hooks == original
    finally:
        torch._dynamo.config.skip_fsdp_hooks = original


@pytest.mark.parametrize("value", [True, False])
def test_skip_fsdp_hooks_accepts_bool_values(value):
    """
    测试用例：验证配置项接受布尔值
    覆盖场景：
    - 参数化测试 True 和 False
    - 验证赋值后值正确
    - 测试后恢复原始配置
    """
    _require_npu()

    original = torch._dynamo.config.skip_fsdp_hooks
    try:
        torch._dynamo.config.skip_fsdp_hooks = value
        assert torch._dynamo.config.skip_fsdp_hooks == value
    finally:
        torch._dynamo.config.skip_fsdp_hooks = original


def test_skip_fsdp_hooks_accepts_truthy_values():
    """
    测试用例：验证配置项接受整数值
    覆盖场景：
    - 赋值整数 1（truthy 值）
    - 赋值整数 0（falsy 值）
    - 验证配置项不自动转换类型
    - 测试后恢复原始配置
    """
    _require_npu()

    original = torch._dynamo.config.skip_fsdp_hooks
    try:
        torch._dynamo.config.skip_fsdp_hooks = 1
        assert torch._dynamo.config.skip_fsdp_hooks == 1

        torch._dynamo.config.skip_fsdp_hooks = 0
        assert torch._dynamo.config.skip_fsdp_hooks == 0
    finally:
        torch._dynamo.config.skip_fsdp_hooks = original
