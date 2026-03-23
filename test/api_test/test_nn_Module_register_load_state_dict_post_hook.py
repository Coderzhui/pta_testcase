"""
测试目的：验证 `torch.nn.Module.register_load_state_dict_post_hook` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Module.register_load_state_dict_post_hook`
覆盖的入参维度：
- 参数传参与不传参：覆盖 hook 注册和 load_state_dict 触发。
- 参数为 None / 非 None：覆盖合法 callable 与 `None` 异常。
- 枚举/多类型：覆盖 missing_keys 可原地修改。
- 正常输入：覆盖 NPU Parameter 模块。
- 异常输入：覆盖非法 hook。
- 边界值和等价类：覆盖空 state_dict。
未覆盖项及原因：
- 未覆盖 strict=True 下更复杂的不兼容键组合。
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

class _LoadStateModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, device="npu"))


def test_register_load_state_dict_post_hook_observes_missing_keys():
    _require_npu()

    module = _LoadStateModule()
    seen = []

    def hook(mod, incompatible_keys):
        seen.append((list(incompatible_keys.missing_keys), list(incompatible_keys.unexpected_keys)))
        incompatible_keys.missing_keys.clear()

    handle = module.register_load_state_dict_post_hook(hook)
    result = module.load_state_dict({}, strict=False)

    assert type(handle).__name__ == "RemovableHandle"
    assert seen == [(["weight"], [])]
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    handle.remove()


def test_register_load_state_dict_post_hook_invalid_hook_raises():
    _require_npu()

    module = _LoadStateModule()
    handle = module.register_load_state_dict_post_hook(None)

    with pytest.raises(TypeError):
        module.load_state_dict({}, strict=False)

    handle.remove()
