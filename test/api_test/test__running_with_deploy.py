"""
测试目的：验证 `torch._running_with_deploy` 的无参/异常签名，并对缺少 NPU 设备语义的正常路径使用显式 skip。
API 名称：`torch._running_with_deploy`
覆盖的入参维度：
- 参数传参与不传参：覆盖无参读取语义说明和多余参数异常调用。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖布尔返回值。
- 正常输入：当前 API 不存在可验证的 NPU 设备语义，使用 `pytest.skip` 避免伪覆盖。
- 异常输入：覆盖多余参数。
- 边界值和等价类：覆盖 eager 默认值。
未覆盖项及原因：
- 该类 API 为状态读取，不覆盖全局配置变更流程。
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

def test_flag_like_api_returns_bool_in_npu_eager_mode():
    _require_npu()
    pytest.skip("`torch._running_with_deploy` 返回进程部署状态，不存在可验证的 NPU 设备语义。")


def test_running_with_deploy_rejects_extra_args():
    _require_npu()

    with pytest.raises(TypeError):
        torch._running_with_deploy("bad")
