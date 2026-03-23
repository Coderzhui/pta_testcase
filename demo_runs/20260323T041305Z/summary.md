# Pipeline Summary: 20260323T041305Z

- Input: `apis.txt`
- Manifest progress CSV: `runs/20260323T041305Z/manifest.csv`
- Fix mode: `tests`
- Command: `/usr/local/python3.11.14/bin/python -m scripts.pipeline run --input apis.txt --fix-mode tests --report-dir /home/l00913161/projects/pta_testcase/runs`
- Total APIs: `53`
- Results JSON: `runs/20260323T041305Z/results.json`
- Results CSV: `runs/20260323T041305Z/results.csv`
- Summary Table CSV: `runs/20260323T041305Z/summary_table.csv`
- Generation Summary: `runs/20260323T041305Z/generation_summary.md`
- Analysis Summary: `runs/20260323T041305Z/analysis_summary.md`

## Status Counts
- `analyzed`: 3
- `fixed`: 11
- `pytest_failed`: 1
- `pytest_passed`: 37
- `skipped`: 1

## Failure Categories
- `NONE`: 48
- `PYTORCH_BUG`: 1
- `UNKNOWN`: 3
- `UNSUPPORTED_ON_NPU`: 1

## Fixed APIs
- `Tensor.register_hook`: initial `TEST_BUG` -> `test/api_test`; rerun `pytest_passed`; changed: test/api_test/test_Tensor_register_hook.py
- `torch.nn.Module.buffers`: initial `TEST_BUG` -> `test/api_test`; rerun `pytest_passed`; changed: test/api_test/test_nn_Module_buffers.py
- `torch.nn.Module.modules`: initial `TEST_BUG` -> `test/api_test`; rerun `pytest_passed`; changed: test/api_test/test_nn_Module_modules.py
- `torch.nn.Module.named_modules`: initial `TEST_BUG` -> `test/api_test`; rerun `pytest_passed`; changed: test/api_test/test_nn_Module_named_modules.py
- `torch.nn.Module.named_parameters`: initial `TEST_BUG` -> `test/api_test`; rerun `pytest_passed`; changed: test/api_test/test_nn_Module_named_parameters.py
- `torch.nn.Module.register_forward_hook`: initial `TEST_BUG` -> `test/api_test`; rerun `pytest_passed`; changed: test/api_test/test_nn_Module_register_forward_hook.py
- `torch.nn.Module.register_forward_pre_hook`: initial `TEST_BUG` -> `test/api_test`; rerun `pytest_passed`; changed: test/api_test/test_nn_Module_register_forward_pre_hook.py
- `torch.nn.Module.register_load_state_dict_post_hook`: initial `TEST_BUG` -> `test/api_test`; rerun `pytest_passed`; changed: test/api_test/test_nn_Module_register_load_state_dict_post_hook.py
- `torch.utils.swap_tensors`: initial `TEST_BUG` -> `test/api_test`; rerun `pytest_passed`; changed: test/api_test/test_utils_swap_tensors.py
- `torch.autograd._unsafe_preserve_version_counter`: initial `TEST_BUG` -> `test/api_test`; rerun `pytest_passed`; changed: test/api_test/test_autograd__unsafe_preserve_version_counter.py
- `torch.autograd.graph._MultiHandle`: initial `TEST_BUG` -> `test/api_test`; rerun `pytest_passed`; changed: test/api_test/test_autograd_graph__MultiHandle.py

## Remaining Failures
- `torch.library.Library`: `PYTORCH_BUG`; RuntimeError: !dispatch_key_.has_value() INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/core/library.cpp":87, please report a bug to PyTorch. (Error occurred while processing TORCH_LIBRARY_FRAGMENT block at /dev/null:166)
def test_library_construct_and_define_for_npu_dispatch():
        _require_npu()
    
        namespace = "pta_test_library_ctor"
        lib = torch.library.Library(namespace, "DEF")
>       frag = torch.library.Library(namespace, "FRAGMENT", "PrivateUse1")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

test/api_test/test_library_Library.py:34: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <[AttributeError("'Library' object has no attribute 'kind'") raised in repr()] Library object at 0xfffdd80a3190>
ns = 'pta_test_library_ctor', kind = 'FRAGMENT', dispatch_key = 'PrivateUse1'

    def __init__(self, ns, kind, dispatch_key=""):
        if kind not in ("IMPL", "DEF", "FRAGMENT"):
            raise ValueError("Unsupported kind: ", kind)
    
        if ns in _reserved_namespaces and (kind == "DEF" or kind == "FRAGMENT"):
            raise ValueError(
                ns,
                " is a reserved namespace. Please try creating a library with another name.",
            )
        if torch._running_with_deploy():
            _library.utils.warn_deploy()
            return
    
        frame = traceback.extract_stack(limit=3)[0]
        filename, lineno = frame.filename, frame.lineno
>       self.m: Optional[Any] = torch._C._dispatch_library(
            kind, ns, dispatch_key, filename, lineno
        )
E       RuntimeError: !dispatch_key_.has_value() INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/core/library.cpp":87, please report a bug to PyTorch. (Error occurred while processing TORCH_LIBRARY_FRAGMENT block at /dev/null:166)

/usr/local/python3.11.14/lib/python3.11/site-packages/torch/library.py:109: RuntimeError

## Skipped APIs
- `torch.fx.node.has_side_effect`: `skipped`; `torch.fx.node.has_side_effect` 缺少可验证的 NPU 设备语义，避免通过无关 NPU Tensor 伪造覆盖。
/home/l00913161/projects/pta_testcase/test/api_test/test_fx_node_has_side_effect.py:31: `torch.fx.node.has_side_effect` 缺少可验证的 NPU 设备语义，避免通过无关 NPU Tensor 伪造覆盖。

## Passing APIs
- Count: 48
