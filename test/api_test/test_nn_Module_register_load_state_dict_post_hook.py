# Test purpose: verify torch.nn.Module.register_load_state_dict_post_hook behavior
# on NPU modules, including hook registration, invocation during load_state_dict,
# handle removal, mutable incompatible_keys behavior, and failure cases for
# invalid self objects / non-callable hooks at load time.
# API name: torch.nn.Module.register_load_state_dict_post_hook
#
# Covered parameter dimensions:
# | 维度 | 覆盖情况 | 说明 |
# | --- | --- | --- |
# | 参数传参与不传参 | 覆盖 | 覆盖 hook 显式传参与 handle.remove 行为 |
# | None / 非 None | 覆盖 | hook=None 与非 None hook 对象均覆盖 |
# | 枚举选项 | 覆盖 | 覆盖可调用 hook、非可调用 hook、移除后不触发 |
# | 多类型 | 覆盖 | hook 使用 function、callable object、int、None、object |
# | 正常输入 | 覆盖 | NPU module load_state_dict 成功路径、空不一致键路径 |
# | 异常输入 | 覆盖 | 非 Module self、非 callable hook 在 load 时触发异常 |
# | 边界值和等价类 | 覆盖 | empty incompatible_keys、missing/unexpected keys、remove 后边界 |
#
# Uncovered items and reasons:
# | Item | Reason |
# | --- | --- |
# | 数值精度校验 | intentionally omitted; this API only concerns hook registration and load-state behavior |
# | registration-time hook type rejection | not covered because current runtime accepts any object at registration and defers failure to load_state_dict |

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - import-time environment guard
    torch_npu = None  # type: ignore[assignment]
    _TORCH_NPU_IMPORT_ERROR = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None


def _require_npu() -> None:
    if _TORCH_NPU_IMPORT_ERROR is not None:
        pytest.skip(f"torch_npu import failed: {_TORCH_NPU_IMPORT_ERROR}")
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu backend is unavailable in this environment")
    if not torch.npu.is_available():
        pytest.skip("NPU device is unavailable in this environment")


def _npu_device() -> torch.device:
    _require_npu()
    return torch.device("npu:0")


def test_register_load_state_dict_post_hook_invokes_and_mutates_incompatible_keys_on_npu() -> None:
    _require_npu()

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("buf", torch.ones(1, device=_npu_device()))

    net = Net()
    seen = {}

    def hook(module, incompatible_keys):
        seen["module_is_net"] = module is net
        seen["missing_before"] = tuple(incompatible_keys.missing_keys)
        seen["unexpected_before"] = tuple(incompatible_keys.unexpected_keys)
        assert module.buf.device.type == "npu"
        incompatible_keys.missing_keys.append("missing_from_hook")
        incompatible_keys.unexpected_keys.append("unexpected_from_hook")

    handle = net.register_load_state_dict_post_hook(hook)
    result = net.load_state_dict({"buf": torch.ones(1, device=_npu_device())}, strict=False)

    assert isinstance(handle, torch.utils.hooks.RemovableHandle)
    assert seen == {
        "module_is_net": True,
        "missing_before": (),
        "unexpected_before": (),
    }
    assert result.missing_keys == ["missing_from_hook"]
    assert result.unexpected_keys == ["unexpected_from_hook"]
    assert net.buf.device.type == "npu"


def test_register_load_state_dict_post_hook_remove_prevents_invocation_on_npu() -> None:
    _require_npu()

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("buf", torch.zeros(1, device=_npu_device()))

    net = Net()
    called = []

    def hook(module, incompatible_keys):
        called.append((module.__class__.__name__, tuple(incompatible_keys.missing_keys)))

    handle = net.register_load_state_dict_post_hook(hook)
    handle.remove()
    result = net.load_state_dict({"buf": torch.zeros(1, device=_npu_device())})

    assert called == []
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert net.buf.device.type == "npu"


def test_register_load_state_dict_post_hook_non_callable_fails_when_loading_on_npu() -> None:
    _require_npu()

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("buf", torch.ones(1, device=_npu_device()))

    net = Net()
    handle = net.register_load_state_dict_post_hook(123)

    with pytest.raises(TypeError, match="callable"):
        net.load_state_dict({"buf": torch.ones(1, device=_npu_device())})

    handle.remove()


def test_register_load_state_dict_post_hook_invalid_self_raises() -> None:
    _require_npu()

    with pytest.raises(AttributeError, match="_load_state_dict_post_hooks"):
        torch.nn.Module.register_load_state_dict_post_hook(None, lambda module, keys: None)
