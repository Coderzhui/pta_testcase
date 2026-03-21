# PyTorch NPU API 批处理流水线 🚀

这个仓库用于把一批 PyTorch API 自动转成 NPU 上可运行的 `pytest` 用例，并继续完成运行、结果分析、报告输出，以及低风险自动修复。

目标不是只“生成测试文件”，而是把整条链路尽量自动化，减少用户手工切换步骤。理想使用方式是：用户准备一个 `apis.txt`，执行一条命令，最后拿到测试文件、运行结果、失败分析和报告。

## 仓库能做什么 ✨

给定一批 API，这个仓库可以自动完成：

1. 把 `apis.txt` 转成 `api_manifest.csv`
2. 为每个 API 在 `test/api_test/` 下生成 1 个测试文件
3. 审查并对测试文件做最小修复
4. 运行 `pytest`
5. 解析 `pass/fail/skip/xfail`
6. 输出结构化结果和 Markdown 摘要
7. 在允许的范围内尝试自动修复简单问题
8. 对修复后的结果做定向复跑

## 适用对象 👥

- 需要批量补 PyTorch API NPU 功能测试的开发者
- 需要快速知道“哪些 API 已通过、哪些失败、为什么失败”的维护者
- 需要收敛手工操作步骤、把批处理流程脚本化的使用者

## 核心约束 📌

测试生成和修复必须遵守仓库规则，核心约束如下：

- 每个 API 只对应 1 个测试文件
- 测试文件统一放在 `test/api_test/`
- 文件名必须使用 manifest 中的 `file_name`
- 测试框架必须是 `pytest`
- 测试必须跑在 NPU 上，依赖 `torch_npu`
- 关注 API 功能行为和接口覆盖，不做数值精度比对
- 当前环境不支持的场景，必须用 `pytest.skip` 或 `pytest.xfail` 明确说明原因

更完整的规则见 [AGENTS.md](/home/l00913161/projects/pta_testcase/AGENTS.md)。

## 流程总览 🔄

整条流水线分成 5 个阶段：

### 1. Manifest 阶段 🧾

输入可以是：

- `apis.txt`
- 已存在的 `api_manifest.csv`

如果输入是 `apis.txt`，流水线会自动生成 manifest。manifest 当前至少包含这些列：

- `raw_api_name`
- `canonical_name`
- `file_name`
- `status`
- `notes`

其中：

- `status=pending` 表示该 API 会参与当前批次处理
- `file_name` 是目标测试文件名，最终必须落到 `test/api_test/`

### 2. 生成阶段 🛠️

流水线会调用 Codex 批量生成测试文件，并要求：

- 只处理 `status=pending` 的 API
- 只修改 `test/api_test/` 下对应目标文件
- 不重写与当前 API 无关的文件
- 优先保证文件可 import、pytest 可收集、命名正确

### 3. 执行阶段 ▶️

生成完成后，流水线统一运行 `pytest`，并把原始命令、stdout/stderr、JUnit XML 落盘，避免“只看到最终一句总结，看不到原始上下文”。

### 4. 分析阶段 🔍

流水线会把 pytest 结果映射成结构化字段，例如：

- `final_status`
- `pytest_outcome`
- `failure_category`
- `root_cause_summary`
- `fix_recommendation`

失败分类使用统一 taxonomy，见 [docs/failure_taxonomy.md](/home/l00913161/projects/pta_testcase/docs/failure_taxonomy.md)。

### 5. 自动修复阶段 🩹

如果启用了自动修复，流水线会针对失败、`skip`、`xfail` 或未生成成功的 API，尝试做低风险修复，然后自动复跑对应测试。

自动修复是分级的，不是无边界自由修改：

- `off`: 不自动修复
- `tests`: 只允许修测试文件
- `safe`: 在 `tests` 基础上，允许修 `pytorch/` 或 `ascend-pytorch/` 的局部低风险问题

## 目录说明 🗂️

几个关键目录的职责如下：

- `test/api_test/`: 生成出的 API 测试文件
- `scripts/`: 外层 pipeline 和兼容入口
- `runs/`: 每次批处理的结果工件
- `docs/`: failure taxonomy 等补充文档
- `pytorch/`: PyTorch 源码树
- `ascend-pytorch/`: torch_npu / NPU 相关源码树

## 快速开始 ⚡

### 方式 1：从 `apis.txt` 直接运行 📝

先准备一个文本文件，每行一个 API，例如：

```text
Tensor.new_empty
torch.Event
torch.nn.Module.named_parameters
```

然后执行：

```bash
python -m scripts.pipeline run --input apis.txt --fix-mode tests
```

这是推荐用法。流水线会自动生成 manifest，然后完成生成、运行、分析和报告。

### 方式 2：从现成 manifest 运行 📄

如果你已经准备好了 `api_manifest.csv`：

```bash
python -m scripts.pipeline run --input api_manifest.csv --fix-mode tests
```

这适合你需要手动维护 `status`、`notes` 或 `file_name` 的场景。

## 常用命令 💻

### 1. 只生成 manifest 🧱

```bash
python -m scripts.pipeline build-manifest --input apis.txt --output api_manifest.csv
```

兼容旧入口：

```bash
python process_api_manifest.py apis.txt api_manifest.csv
```

### 2. 跑完整流水线 🏃

```bash
python -m scripts.pipeline run --input apis.txt --fix-mode tests
```

### 3. 指定 run artifact 目录 📦

```bash
python -m scripts.pipeline run --input apis.txt --fix-mode off --report-dir /tmp/pta_runs
```

### 4. 复用已有 run 目录 ♻️

```bash
python -m scripts.pipeline run --input api_manifest.csv --resume runs/20260321T091734Z
```

### 5. 跳过生成，只做执行和分析 ⏭️

这个模式适合你已经有现成测试文件，或者只想重跑 pytest 和重新产出报告：

```bash
python -m scripts.pipeline run --input api_manifest.csv --skip-generate --fix-mode off
```

### 6. 兼容旧 shell 入口 🧰

```bash
bash scripts/run_api_batch.sh apis.txt
bash scripts/run_api_batch.sh api_manifest.csv --fix-mode safe
```

## `run` 命令参数说明 🧭

当前 `run` 子命令支持：

- `--input`: 输入文件，支持 `apis.txt` 或 `api_manifest.csv`
- `--report-dir`: 指定 run artifact 根目录，默认是 `runs/`
- `--resume`: 复用已有 run 目录
- `--fix-mode`: 自动修复范围，支持 `off/tests/safe`
- `--skip-generate`: 跳过生成阶段，复用已有测试文件
- `--max-workers`: 给内层生成阶段的并发预算提示

查看命令帮助：

```bash
python -m scripts.pipeline run --help
```

## 推荐使用方式 ✅

大多数情况下，建议按下面的策略使用：

- 第一次批量跑新 API：`--fix-mode tests`
- 只想观察当前质量，不希望自动改文件：`--fix-mode off`
- 你确认允许对 `pytorch/` 或 `ascend-pytorch/` 做小范围修复：`--fix-mode safe`
- 已经有测试文件，只想重跑和产报告：`--skip-generate`

## 输入文件格式 🧮

### `apis.txt` 📃

每行一个 API，空行和 `#` 注释行会被忽略。例如：

```text
# tensor methods
Tensor.new_empty
Tensor.new_zeros

# torch namespace
torch.Event
torch.utils.swap_tensors
```

### `api_manifest.csv` 🗃️

需要至少包含这几列：

```csv
raw_api_name,canonical_name,file_name,status,notes
Tensor.new_empty,Tensor.new_empty,test_Tensor_new_empty.py,pending,
torch.Event,torch.Event,test_Event.py,pending,
```

说明：

- `raw_api_name`: 原始 API 名称
- `canonical_name`: 当前批次的标准 API 名称
- `file_name`: 目标测试文件名
- `status`: 当前是否参与处理
- `notes`: 额外说明，可作为分析上下文

当前默认处理 `status=pending` 的条目。如果 manifest 里没有 `pending`，流水线会退化为处理全部条目。

## 输出工件说明 📤

每次运行都会生成一个独立目录，例如：

```text
runs/20260321T091734Z/
```

其中常见文件如下：

- `manifest.csv`: 本次运行使用的 manifest 快照
- `generation_summary.md`: 生成阶段摘要
- `results.json`: 面向程序消费的结构化结果
- `results.csv`: 面向表格查看的结果汇总
- `summary.md`: 面向人工阅读的批次摘要
- `pytest_raw/*.command.txt`: pytest 执行命令
- `pytest_raw/*.stdout.log`: pytest 标准输出
- `pytest_raw/*.stderr.log`: pytest 错误输出
- `pytest_raw/*_junit.xml`: JUnit XML
- `fixes/*.md`: 自动修复摘要
- `fixes/*.stdout.log`: 修复阶段日志
- `fixes/*.stderr.log`: 修复阶段错误日志

## 如何阅读结果 👀

### 优先看 `summary.md` 🧠

适合快速回答这些问题：

- 本批次一共跑了多少个 API
- 多少通过、多少失败、多少 skip/xfail
- 哪些 API 被自动修复了
- 还有哪些失败没有解决

### 再看 `results.csv` 或 `results.json` 📊

适合做统计、筛选和后处理。几个关键字段：

- `final_status`: API 的最终状态
- `pytest_outcome`: pytest 侧的结果概括
- `failure_category`: 失败分类
- `root_cause_summary`: 当前总结的根因摘要
- `fix_recommendation`: 建议修复动作
- `fix_applied`: 是否实际做了修复
- `fix_target`: 修复目标位置
- `changed_files`: 本次修复触达的文件
- `rerun_status`: 修复后复跑结果

### 常见 `final_status` 🏷️

- `pytest_passed`: 测试通过
- `pytest_failed`: pytest 运行失败
- `skipped`: 该 API 当前被 skip
- `xfailed`: 该 API 当前被 xfail
- `review_failed`: 预期测试文件没有成功生成
- `fixed`: 经过自动修复后通过

## 自动修复边界 🧱

自动修复不是无限制的。当前设计原则是：

- 优先修测试，不优先改源码
- 只修当前 API 直接相关的问题
- 不做大范围重构
- 不静默改源码后不复跑
- 所有源码修复都应该是局部、低风险、可被当前批次验证的问题

### `tests` 模式下通常会处理的问题 🔧

- 测试文件 import/collect 问题
- 明显错误的参数或异常断言
- 应该 `skip/xfail` 但没有正确标注的场景
- 文件头覆盖说明明显缺失

### `safe` 模式下额外允许的问题 🧪

- 小范围、直接可验证的 `pytorch/` 修复
- 小范围、直接可验证的 `ascend-pytorch/` 修复

如果问题需要架构层决策、影响面不清晰，当前流程应当停留在“报告问题”，而不是自动修改。

## 失败分类说明 🧬

当前失败分类包括：

- `NONE`
- `TEST_BUG`
- `UNSUPPORTED_ON_NPU`
- `ENVIRONMENT_MISSING`
- `API_BEHAVIOR_MISMATCH`
- `PYTORCH_BUG`
- `TORCH_NPU_BUG`
- `FLAKY_OR_UNSTABLE`
- `INSUFFICIENT_COVERAGE`
- `UNKNOWN`

详细定义和默认修复映射见：

- [docs/failure_taxonomy.md](/home/l00913161/projects/pta_testcase/docs/failure_taxonomy.md)

## 典型使用场景 🎯

### 场景 1：新批次 API，全流程自动跑 🚀

```bash
python -m scripts.pipeline run --input apis.txt --fix-mode tests
```

适合第一次处理一批新 API。

### 场景 2：只想拿报告，不希望自动改任何东西 📋

```bash
python -m scripts.pipeline run --input apis.txt --fix-mode off
```

适合做基线统计或人工 review 前的扫描。

### 场景 3：测试文件已经存在，只重跑结果 🔁

```bash
python -m scripts.pipeline run --input api_manifest.csv --skip-generate --fix-mode off
```

适合调试已有批次。

### 场景 4：允许低风险源码修复 🧯

```bash
python -m scripts.pipeline run --input api_manifest.csv --fix-mode safe
```

适合你明确接受对 `pytorch/` 或 `ascend-pytorch/` 的小范围自动 patch。

## 常见问题 ❓

### 为什么某些 API 是 `skip` 或 `xfail`？ 🤔

因为当前环境、当前 NPU 后端或当前构建并不稳定支持该路径。对于这类场景，流程要求显式写明原因，而不是伪造通过。

### 为什么有的 API 是 `review_failed`？ 🧩

这通常表示目标测试文件没有成功生成，或者生成后没有通过最基本的收集要求。此时优先看：

- `generation_summary.md`
- `results.csv`
- `pytest_raw/`

### 为什么 summary 通过了，但仍然有 `skip/xfail`？ 📎

`summary.md` 中的“通过”是批次整体意义上的通过情况，不代表每个 API 都是纯 `passed`。`skip/xfail` 仍然会被单独列出。

### 什么时候适合用 `safe`？ ⚠️

只有当你接受本地工作区中的 `pytorch/` 或 `ascend-pytorch/` 可能被修改，并且希望把低风险简单问题直接自动修掉时，才建议使用。

## 当前限制 🚧

当前实现仍有这些现实边界：

- 失败分类是启发式的，不是严格根因证明
- 自动修复优先面向低风险、小范围问题
- 对复杂源码问题，当前更适合输出报告而不是强行自动修改
- `safe` 模式下的源码修复能力仍然依赖具体失败模式是否足够明确

## 建议的日常工作流 🛤️

如果你要长期使用这个仓库，建议采用下面的节奏：

1. 编辑 `apis.txt`
2. 执行 `python -m scripts.pipeline run --input apis.txt --fix-mode tests`
3. 先看 `runs/<run_id>/summary.md`
4. 再看 `results.csv` 做过滤和统计
5. 对剩余失败项决定是否继续人工处理，或切换到 `--fix-mode safe`

## 相关文件 🔗

- [scripts/pipeline.py](/home/l00913161/projects/pta_testcase/scripts/pipeline.py)
- [scripts/run_api_batch.sh](/home/l00913161/projects/pta_testcase/scripts/run_api_batch.sh)
- [process_api_manifest.py](/home/l00913161/projects/pta_testcase/process_api_manifest.py)
- [docs/failure_taxonomy.md](/home/l00913161/projects/pta_testcase/docs/failure_taxonomy.md)
- [AGENTS.md](/home/l00913161/projects/pta_testcase/AGENTS.md)
