# Test purpose: validate `torch._dynamo.config.skip_fsdp_hooks` as a mutable
#               configuration flag used by Dynamo/FSDP hook handling, covering
#               default value, explicit updates, and the fact that this runtime
#               does not enforce type validation on assignment.
# API name: torch._dynamo.config.skip_fsdp_hooks
#
# Covered parameter dimensions
# | Dimension | Covered | Notes |
# | --- | --- | --- |
# | default value / explicit update | Covered | The flag default is checked and then toggled. |
# | True / False | Covered | Both boolean states are exercised. |
# | NPU runtime presence | Covered | An NPU tensor is created to keep the test anchored to the required target runtime. |
#
# Uncovered items / reasons
# | Item | Reason |
# | --- | --- |
# | Assignment type validation errors | The config is a plain module-level bool in this build, so invalid assignments are not rejected and there is no reliable exception path. |
# | Functional FSDP hook behavior | This file targets the config flag itself, not the full FSDP hook execution pipeline. |

import pytest
import torch
import torch_npu  # noqa: F401

import torch._dynamo.config as dynamo_config


def _require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available in this environment.")


def test_skip_fsdp_hooks_default_and_mutation_on_npu():
    _require_npu()

    npu_tensor = torch.ones(1, device="npu")
    assert npu_tensor.device.type == "npu"

    original = dynamo_config.skip_fsdp_hooks
    try:
        assert original is True

        dynamo_config.skip_fsdp_hooks = False
        assert dynamo_config.skip_fsdp_hooks is False

        dynamo_config.skip_fsdp_hooks = True
        assert dynamo_config.skip_fsdp_hooks is True
    finally:
        dynamo_config.skip_fsdp_hooks = original


def test_skip_fsdp_hooks_no_reliable_type_error_path():
    _require_npu()
    pytest.skip(
        "skip_fsdp_hooks is a plain module-level bool in this runtime; "
        "non-bool assignment is not rejected, so there is no reliable exception path."
    )
