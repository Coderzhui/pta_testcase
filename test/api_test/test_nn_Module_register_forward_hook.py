"""
测试目的：验证 `torch.nn.Module.register_forward_hook` 在 NPU 环境下的基础功能行为、返回对象类型和异常分支。
API 名称：`torch.nn.Module.register_forward_hook`
覆盖的入参维度：
- 参数传参与不传参：覆盖默认 hook 与 `with_kwargs=True`。
- 参数为 None / 非 None：覆盖合法 callable 与 `None` 异常。
- 枚举/多类型：覆盖输出变换返回值。
- 正常输入：覆盖 NPU forward 后置 hook。
- 异常输入：覆盖非法 hook。
- 边界值和等价类：覆盖单输入单输出。
未覆盖项及原因：
- 未覆盖多个 hook 的 prepend/alway_call 顺序组合。
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

class _HookModule(torch.nn.Linear):
    def forward(self, input, scale=1):
        return super().forward(input) * scale


def test_module_register_forward_hook_on_npu():
    _require_npu()

    module = _HookModule(2, 2).to("npu")
    with torch.no_grad():
        module.weight.zero_()
        module.bias.zero_()
    seen = []

    def hook(mod, args, kwargs, output):
        seen.append((len(args), kwargs["scale"], output.device.type))
        return output + 1

    handle = module.register_forward_hook(hook, with_kwargs=True)
    out = module(torch.ones(2, device="npu"), scale=1)

    assert type(handle).__name__ == "RemovableHandle"
    assert seen == [(1, 1, "npu")]
    assert out.device.type == "npu"
    assert torch.equal(out.cpu(), torch.ones(2))
    handle.remove()


def test_module_register_forward_hook_invalid_hook_raises():
    _require_npu()

    module = torch.nn.Linear(1, 1).to("npu")
    handle = module.register_forward_hook(None)

    with pytest.raises(TypeError):
        module(torch.ones(1, device="npu"))

    handle.remove()


def test_module_register_forward_hook_prepend_orders_hooks():
    _require_npu()

    module = _HookModule(1, 1).to("npu")
    order = []
    handle_first = module.register_forward_hook(lambda *args: order.append("first"))
    handle_prepend = module.register_forward_hook(lambda *args: order.append("prepend"), prepend=True)

    module(torch.ones(1, device="npu"))

    assert order == ["prepend", "first"]
    handle_first.remove()
    handle_prepend.remove()
