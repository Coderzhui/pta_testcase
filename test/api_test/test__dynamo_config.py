"""
测试目的：验证 `torch._dynamo.config` 模块在 NPU 环境下的完整功能行为。
API 名称：`torch._dynamo.config`

覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为模块对象访问。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：已覆盖多个布尔配置项访问。
- 正常输入：已覆盖模块属性读取和修改。
- 异常输入：已覆盖缺失属性访问。
- 边界值和等价类：已覆盖多个已有配置项。

已补充覆盖项：
- ✓ 配置项修改与恢复机制
- ✓ 多个常用配置项测试（skip_fsdp_hooks、verbose、suppress_errors）
- ✓ verbose 配置项访问测试

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


def test_dynamo_config_module_is_accessible_from_npu_context():
    """
    测试用例：验证 config 模块在 NPU 上下文中可访问
    覆盖场景：
    - 在 NPU 设备上创建 tensor
    - 访问 skip_fsdp_hooks 配置项
    - 验证配置项类型为 bool
    - 验证 verbose 属性存在
    """
    _require_npu()

    tensor = torch.ones(1, device="npu")
    assert tensor.device.type == "npu"
    assert isinstance(torch._dynamo.config.skip_fsdp_hooks, bool)
    assert hasattr(torch._dynamo.config, "verbose")


def test_dynamo_config_missing_attr_raises():
    """
    测试用例：验证访问不存在的属性抛出异常
    覆盖场景：
    - 访问不存在的配置项
    - 验证抛出 AttributeError
    """
    _require_npu()

    with pytest.raises(AttributeError):
        getattr(torch._dynamo.config, "not_a_real_config")


def test_dynamo_config_verbose_is_accessible():
    """
    测试用例：验证 verbose 配置项可访问
    覆盖场景：
    - 检查 verbose 属性存在
    - 读取 verbose 值
    - 验证值非空
    """
    _require_npu()

    assert hasattr(torch._dynamo.config, "verbose")
    verbose_value = torch._dynamo.config.verbose
    assert verbose_value is not None


def test_dynamo_config_can_modify_and_restore():
    """
    测试用例：验证配置项可修改并恢复
    覆盖场景：
    - 保存原始 verbose 值
    - 修改为相反值
    - 验证修改生效
    - 恢复原始值
    """
    _require_npu()

    original_verbose = torch._dynamo.config.verbose
    try:
        torch._dynamo.config.verbose = not original_verbose
        assert torch._dynamo.config.verbose == (not original_verbose)
    finally:
        torch._dynamo.config.verbose = original_verbose


@pytest.mark.parametrize("attr", ["skip_fsdp_hooks", "verbose", "suppress_errors"])
def test_dynamo_config_common_attrs_exist(attr):
    """
    测试用例：验证常用配置项存在
    覆盖场景：
    - 参数化测试多个配置项
    - skip_fsdp_hooks：FSDP hooks 跳过配置
    - verbose：详细输出配置
    - suppress_errors：错误抑制配置
    """
    _require_npu()

    assert hasattr(torch._dynamo.config, attr)
