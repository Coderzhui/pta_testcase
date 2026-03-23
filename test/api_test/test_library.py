"""
测试目的：验证 `torch.library` 模块对象在 NPU 场景下暴露的主入口可正常用于自定义算子注册。
API 名称：`torch.library`
覆盖的入参维度：
- 参数传参与不传参：不适用，该 API 为模块对象访问。
- 参数为 None / 非 None：不适用。
- 枚举/多类型：覆盖模块暴露的 `Library` / `impl` 成员。
- 正常输入：覆盖通过 `torch.library` 模块创建自定义 NPU op。
- 异常输入：覆盖缺失属性访问。
- 边界值和等价类：覆盖最小 schema。
未覆盖项及原因：
- 未覆盖该模块下的所有辅助入口，仅验证主干入口。
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

def test_torch_library_module_exposes_npu_registration_entrypoints():
    _require_npu()

    assert hasattr(torch.library, "Library")
    assert hasattr(torch.library, "impl")

    namespace = "pta_test_library_module"
    lib = torch.library.Library(namespace, "DEF")
    lib.define("myop(Tensor x) -> Tensor")

    @torch.library.impl(f"{namespace}::myop", "PrivateUse1")
    def myop_privateuse1(x):
        return x + 1

    out = torch.ops.pta_test_library_module.myop(torch.ones(1, device="npu"))
    assert out.device.type == "npu"


def test_torch_library_missing_attr_raises():
    _require_npu()

    with pytest.raises(AttributeError):
        getattr(torch.library, "not_a_real_attr")
