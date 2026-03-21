已按 `api_manifest.csv` 的 53 个 `pending` API 完整跑完生成、审查、最小修复和 pytest。只新增/修改了 [test/api_test](/home/l00913161/projects/pta_testcase/test/api_test)，共 53 个测试文件，每个 API 1 个文件，文件名均与 CSV 一致。

`pytest` 最终结果是：`222 passed, 8 skipped, 3 xfailed`。没有仍然失败的 API。修复后通过的 11 个 API 是：`Tensor.new_empty`、`Tensor.new_zeros`、`torch.nn.Module.modules`、`torch.utils._pytree.tree_map`、`torch.nn.Parameter.device.type`、`torch._from_functional_tensor`、`torch.Event`、`torch.nn.Parameter.size`、`torch._C._ExcludeDispatchKeyGuard`、`torch.nn.Module._parameters`、`torch.autograd.graph._MultiHandle`。其余 42 个 API 生成后无需额外修复即通过。

`skip/xfail` 明细如下：
- `torch._sync`：`xfail`，原生 NPU tensor 路径会命中当前构建里的 functionalization internal assert；functional tensor 成功路径已覆盖。
- `torch.Event`：`xfail`，当前 NPU 后端不支持 `elapsed_time`。
- `torch._dynamo.compiled_autograd.compiled_autograd_enabled`：`xfail`，compiled backward 在当前构建不稳定。
- `torch.nn.Parameter.is_contiguous`：`skip`，`channels_last` 在当前 NPU 构建不可靠。
- `torch._dynamo.compiled_autograd.compiled_autograd_enabled_force_eager`：`skip`，该 API 在当前构建里不是 context manager。
- `torch._dynamo.config.skip_fsdp_hooks`：`skip`，当前只是普通模块级布尔值，没有可靠异常路径可测。
- `torch._C.DispatchKeySet`：`skip`，当前构建没有 `DispatchKey.NPU`。

明显未覆盖项主要是这些受构建/后端限制的路径：CPU 输出路径、非稳定 layout 或 dispatch-key 组合、compiled-autograd 完整后向图、部分 backend-specific fallback 行为，以及若干属性类 API 的穷举 dtype 矩阵。这些都已在对应文件头注释里标明原因，没有伪造覆盖。