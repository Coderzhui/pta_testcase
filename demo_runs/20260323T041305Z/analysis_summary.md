# Analysis Summary: 20260323T041305Z

- Fix mode: `tests`
- Inputs: `runs/20260323T041305Z/analysis_inputs.json`
- Triage JSON: `runs/20260323T041305Z/analysis_triage.json`
- Codex notes: `runs/20260323T041305Z/analysis_codex.md`

## Auto-Fix Candidates
- `Tensor.register_hook`: `TEST_BUG` -> `adjust_test`; The test expects register_hook(None) to fail at registration time, but current PyTorch accepts the hook and only raises TypeError when backward later tries to call the stored None hook.
- `torch.nn.Module.buffers`: `TEST_BUG` -> `adjust_test`; The invalid-self test is too strict: calling the unbound method with a string deterministically raises AttributeError from self.named_buffers, not TypeError.
- `torch.nn.Module.modules`: `TEST_BUG` -> `adjust_test`; The invalid-self test is too strict: calling the unbound method with a string deterministically raises AttributeError from self.named_modules, not TypeError.
- `torch.nn.Module.named_modules`: `TEST_BUG` -> `adjust_test`; The invalid-self test is too strict: calling the unbound method with a string deterministically raises AttributeError when the implementation accesses self._modules, not TypeError.
- `torch.nn.Module.named_parameters`: `TEST_BUG` -> `adjust_test`; The invalid-self test is too strict: calling the unbound method with a string deterministically raises AttributeError from self._named_members, not TypeError.
- `torch.nn.Module.register_forward_hook`: `TEST_BUG` -> `adjust_test`; The test expects register_forward_hook(None) to fail during registration, but current PyTorch stores None and raises TypeError only when forward later invokes the hook.
- `torch.nn.Module.register_forward_pre_hook`: `TEST_BUG` -> `adjust_test`; The test expects register_forward_pre_hook(None) to fail during registration, but current PyTorch stores None and raises TypeError only when forward later invokes the hook.
- `torch.nn.Module.register_load_state_dict_post_hook`: `TEST_BUG` -> `adjust_test`; The test expects register_load_state_dict_post_hook(None) to fail during registration, but current PyTorch stores None and raises TypeError only when load_state_dict later invokes the hook.
- `torch.utils.swap_tensors`: `TEST_BUG` -> `adjust_test`; The invalid-input test is too strict: passing a non-Tensor reaches t._use_count() and raises AttributeError, while the test only accepts TypeError or RuntimeError.
- `torch.autograd._unsafe_preserve_version_counter`: `TEST_BUG` -> `adjust_test`; The invalid-input test is too strict: passing None hits the constructor assertion that tensors must be a tuple and raises AssertionError, while the test only accepts TypeError or RuntimeError.
- `torch.autograd.graph._MultiHandle`: `TEST_BUG` -> `adjust_test`; The test expects _MultiHandle(None) to fail in the constructor, but the constructor accepts None and TypeError appears only if remove() later iterates the handles.

## Report-Only Failures
- `torch.library.Library`: `PYTORCH_BUG`; Creating torch.library.Library with kind="FRAGMENT" and a non-empty dispatch key triggers an internal assert in /pytorch/aten/src/ATen/core/library.cpp:87; the same failure reproduces with CPU and PrivateUse1, so this points to PyTorch core registration logic.
- `torch.fx.node.has_side_effect`: `INSUFFICIENT_COVERAGE`; The case is skipped because this API has no meaningful NPU device behavior to validate, so the generated file does not provide a runnable NPU-focused coverage scenario.
