# PyTorch NPU API 用例生成

## 用法

将所有API放到`api.txt`，每行一个API，格式为`torch.xxx.yyy`。然后运行：

```bash
python process_api_manifest.py api.txt
```

得到`api_manifest.csv`，包含待处理的API列表和相关信息。接着运行：

```bash
bash scripts/run_api_batch.sh api_manifest.csv
```

该脚本会自动执行以下步骤：
1. 读取`api_manifest.csv`中状态为`pending`的API。
2. 并行生成对应的测试文件到`test/api_test/`。
3. 并行审查生成的测试文件，修复不通过的项。
4. 运行`pytest`验证测试文件。
5. 输出最终汇总结果。
