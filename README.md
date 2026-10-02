# 大理大学医学生 AI 问诊陪练助手

面向医学教育问诊（病史采集）技能训练场景的 AI 陪练应用。通过构建高拟真的 **AI 虚拟标准化病人（VSP）**，让医学生随时随地开展反复、安全、可量化的问诊训练，并由**智能评分引擎**提供标准化、个性化的即时反馈，形成「训练—评分—反馈—成长」的完整教学闭环。

- **技术底座**：Python + Streamlit + 大语言模型（OpenAI 兼容接口）
- **核心机制**：角色引擎 + 病例事实约束 + 低温度采样
- **评分体系**：临床问诊评分量表（Rubric）——问诊完整性 80 分 + 综合表现 20 分

---

## 一、功能特性

| 编号 | 功能模块 | 说明 |
| --- | --- | --- |
| F1 | 智能问诊陪练 | AI 虚拟标准化病人基于病例进行多轮自然语言问诊对话，回复**流式输出**（打字机效果） |
| F2 | 多学科病例库 | 覆盖 16 个科室（17 例，P01–P17，易/中/难三档），支持按**科室、难度**筛选 |
| F3 | 考试模拟 | 限轮次（10 问）+ 计时，问满自动交卷并生成评分 |
| F4 | 智能评分反馈 | 规则评分（离线基线）+ LLM 结构化评分，输出总分/分维度得分/优点/不足/建议；接口调用含**限流退避重试** |
| F5 | 训练记录持久化 | 每次评分自动落盘为 JSON，形成个人成长档案 |
| F6 | 会话持久化 | 训练中途刷新/重开页面可**自动恢复**上次问诊进度 |
| F7 | 导出 | 评分报告一键下载（Markdown / JSON）+ 完整问诊对话成稿（Markdown） |
| F8 | 诊断剧透防护 | 自动检测虚拟病人是否泄露诊断结论，并在报告中标记 |
| F9 | 个人成长档案 | 训练概览统计（次数/平均/最高/最低分）+ 历史记录表 + 总分趋势 + 能力雷达图 |
| F10 | 实验评测可视化 | benchmark 评测指标柱状图（`eval/visualize.py`，懒加载 matplotlib） |
| F11 | 标准化评测数据集 | 6 套分科室 MCQ（各 40 题，共 240 题，A/B/C/D 四选项 + 答案） |
| F12 | 一键评测脚本 | `run_benchmark_all.py` 串联多模型 × 多数据集评测与零样本迁移衰减矩阵 |

## 二、目录结构

```
AI/
├── src/
│   ├── app.py                    # Streamlit 主应用（串联各模块）
│   ├── requirements.txt          # 依赖清单
│   ├── core/                     # 核心算法包
│   │   ├── config.py             # 全局配置与多模型注册
│   │   ├── case_loader.py        # 病例库加载与校验
│   │   ├── role_engine.py        # 角色引擎（VSP）
│   │   ├── scoring_engine.py     # 评分引擎（规则 + LLM）
│   │   └── records.py            # 训练记录持久化
│   ├── eval/                     # 实验评测包
│   │   ├── metrics.py            # 评估指标
│   │   ├── benchmark.py          # 多模型评测与零样本迁移
│   │   ├── visualize.py          # 评测指标可视化（懒加载 matplotlib）
│   │   └── imbalance.py          # 不平衡样本处理实验
│   ├── cases/                    # 病例库（JSON，P01–P17）
│   ├── tests/                    # 单元/集成测试（pytest，56 项）
│   └── .streamlit/secrets.toml.example  # 密钥配置模板
├── data/                         # 标准化 MCQ 评测数据集（6 套 × 40 题）
├── run_benchmark_all.py          # 一键多模型 × 多数据集评测脚本
├── pyproject.toml                # 项目元数据 + ruff/mypy/coverage 配置
├── pytest.ini                    # pytest 配置
├── requirements-dev.txt          # 开发/测试依赖（pytest、pytest-cov、ruff、mypy）
├── .pre-commit-config.yaml       # 提交前 ruff 检查与格式化
├── Dockerfile                    # 容器化运行镜像
├── .github/workflows/ci.yml      # GitHub Actions CI（lint + 测试 + 覆盖率）
├── gen_docx.py                   # 生成《设计方案》docx
├── gen_report.py                 # 生成《实验测试报告》docx
├── 大理大学医学生AI问诊陪练助手_设计方案_v2.docx
└── 大理大学医学生AI问诊陪练助手_实验测试报告.docx
```

## 三、快速开始

### 1. 环境准备

```bash
# 建议 Python 3.11+，以项目根目录（src/ 的上一级）为基准
cd src
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.streamlit/secrets.toml.example` 为 `.streamlit/secrets.toml`，填入真实密钥（至少一个模型即可）：

```toml
DEEPSEEK_API_KEY = "your-deepseek-key"
DASHSCOPE_API_KEY = "your-qwen-key"
ZHIPU_API_KEY = "your-glm-key"
OPENAI_API_KEY = "your-openai-key"
```

> 密钥文件请勿提交到公开仓库。未配置 API Key 时仅「规则评分」可用，问答对话会提示配置密钥。默认调用 `deepseek-v4-flash`（DeepSeek 经济档），可在 `src/core/config.py` 的 `DEFAULT_MODEL` / `MODEL_REGISTRY` 中切换。

### 3. 启动应用

```bash
cd src
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

### 4. 运行测试

```bash
cd src
pip install -r ../requirements-dev.txt   # 开发/测试依赖（pytest、pytest-cov）
python -m pytest tests/ -v
```

> 项目根目录已配置 `pytest.ini` 与 GitHub Actions CI（`.github/workflows/ci.yml`），提交后自动运行测试并输出覆盖率。

---

## 四、用户手册

### 4.1 技能训练模式

1. 在左侧边栏选择**训练模式 = 技能训练**；
2. 按**科室 / 难度**筛选并选择病例，查看患者基本信息与主诉；
3. 在底部输入框以自然语言向「患者」提问，AI 患者口语化作答（**流式输出**、保守病情细节）；
4. 问诊结束后点击**生成评分报告**，查看总分、分维度得分、优点、不足与改进建议；
5. 评分报告可**下载为 Markdown / JSON**，并自动写入**个人成长档案**；
6. 训练中途刷新或关闭页面，重开应用时会**自动恢复**上次未完成的问诊进度。

### 4.2 考试模拟模式

1. 选择**训练模式 = 考试模拟**，界面显示已提问数与已用时间；
2. 最多提问 **10 轮**，问满自动交卷并生成评分；
3. 也可随时点击**交卷并生成评分**提前结束；
4. 交卷后不再接受提问，可查看评分报告与成长档案。

### 4.3 个人成长档案

页面底部的「个人成长档案」折叠区展示：

- **历史训练记录**：时间、病例、模式、总分列表；
- **总分趋势**：历次得分折线图；
- **能力雷达图**：最近一次评分各维度得分雷达图。

### 4.4 评分说明

- **规则评分（默认）**：基于病例病史信息点覆盖度的启发式打分，可离线运行；
- **LLM 智能评分**：调用大模型按 Rubric 结构化打分（需勾选并配置 API Key），输出更细致的优点/不足/建议。

---

## 五、API 文档

核心模块对外接口摘要（详细参数与返回值见各模块 docstring）。

### 5.1 core.config

| 函数/常量 | 说明 |
| --- | --- |
| `MODEL_REGISTRY` | `{key: ModelConfig}` 多模型注册表（deepseek/qwen/glm/gpt） |
| `get_model_config(key)` | 按注册名获取 `ModelConfig`，未注册抛 `KeyError` |
| `TEMPERATURE` / `SCORING_TEMPERATURE` | 角色/评分采样温度 |

### 5.2 core.case_loader

| 函数 | 说明 |
| --- | --- |
| `load_case(path) -> Case` | 加载并校验单个病例 JSON，失败抛 `CaseLoadError` |
| `load_all_cases(dir) -> dict[str, Case]` | 加载目录下全部病例 |
| `case_to_facts(case) -> str` | 生成注入角色引擎的事实描述（**不含诊断**） |

`Case` 关键字段：`case_id`、`chief_complaint`、`scoring_category`、`department`、`difficulty`、`patient_profile`、`history`、`diagnosis`、`scoring_points`。

### 5.3 core.role_engine

| 函数 | 说明 |
| --- | --- |
| `build_system_prompt(case) -> str` | 组装角色 System Prompt |
| `build_messages(case, history) -> list[dict]` | 组装 system + 历史对话消息 |
| `create_client(model_key, api_key=None) -> OpenAI` | 创建 OpenAI 兼容客户端 |
| `generate_patient_reply(client, case, ...) -> str` | 生成患者回复 |
| `stream_patient_reply(client, case, history=None, ...) -> Iterator[str]` | 以流式逐段生成患者回复（打字机效果） |
| `diagnosis_terms(case) -> list[str]` | 提取诊断关键词（用于剧透检测） |
| `detect_diagnosis_leak(case, history=None) -> list[str]` | 检测对话中的诊断泄露关键词 |

### 5.4 core.scoring_engine

| 函数 | 说明 |
| --- | --- |
| `rule_score(case, inquiry_text) -> ScoringResult` | 规则启发式评分（离线） |
| `llm_score(client, case, history) -> ScoringResult` | LLM 结构化评分 |
| `build_scoring_prompt(case, history) -> str` | 构建评分 Prompt |
| `parse_llm_score(text) -> ScoringResult` | 解析 LLM 评分 JSON |

`ScoringResult` 字段：`total`、`dimensions`、`strengths`、`weaknesses`、`suggestions`、`method`；方法 `to_dict()` / `to_json()` / `from_dict()` / `to_markdown()`（`to_markdown` 用于导出 Markdown 评分报告）。

### 5.5 core.records

| 函数 | 说明 |
| --- | --- |
| `save_record(result, case, *, mode="技能训练") -> Path` | 保存一次评分记录到 `RECORDS_DIR` |
| `load_history() -> list[dict]` | 按时间顺序读取全部历史记录 |
| `save_session(state) -> None` | 保存当前会话状态（刷新续答） |
| `load_session() -> dict \| None` | 读取当前会话状态 |
| `clear_session() -> None` | 清除当前会话 |

### 5.6 eval（实验评测）

| 模块 | 说明 |
| --- | --- |
| `eval.metrics.compute_metrics(y_true, y_pred, labels)` | 计算 accuracy/macro/micro/weighted-F1、混淆矩阵 |
| `eval.benchmark.run_benchmark(...)` | 多模型 × 多数据集评测与迁移矩阵 |
| `eval.benchmark.migration_decay(src, tgt)` | 计算零样本迁移衰减率 |
| `eval.imbalance.run_imbalance_experiment(...)` | 不平衡样本处理对比实验 |
| `eval.visualize.metrics_to_chart_data(metrics)` | 评测指标转图表数据 |
| `eval.visualize.plot_metrics_bar(metrics, ...)` | 生成评测指标柱状图（懒加载 matplotlib） |

---

## 六、开发说明

### 6.1 架构设计

采用分层架构，各层松耦合、可插拔：

- **应用层**：`app.py`（Streamlit 界面、会话管理、结果可视化）；
- **业务逻辑层**：`core/`（病例加载、Prompt 组装、角色引擎、评分引擎、记录持久化）；
- **模型层**：OpenAI 兼容接口多模型热切换（`core/config.py`）；
- **数据/知识层**：病例库 JSON、评分量表（`scoring_engine` 内嵌 Rubric）、训练记录库。

### 6.2 设计模式与约定

- **紧凑的 `from __future__ import annotations` + dataclass**：数据对象（`Case`、`ScoringResult`、`ModelConfig`）使用 dataclass 保证类型安全；
- **惰性导入**：`openai`、`matplotlib` 仅在真正调用时导入，保证核心逻辑与评测模块离线可测试；
- **规则优先、LLM 增强**：评分提供离线基线，LLM 评分失败时回退规则评分，保证可用性；
- **中英文字段别名兼容**：病例 JSON 同时支持中文字段（`主诉`/`科室`）与英文别名（`chief_complaint`/`department`）。

### 6.3 扩展指南

- **新增模型**：在 `core/config.py` 的 `MODEL_REGISTRY` 增加 `ModelConfig` 并在 secrets 中配置对应 Key；
- **新增病例**：在 `src/cases/` 下新增 JSON（含 `case_id`、`主诉`、`科室`、`难度`、`患者画像`、各病史字段、`诊断`、`评分维度`）；
- **调整评分量表**：修改 `core/scoring_engine.py` 中 `RUBRIC_COMPLETENESS` / `RUBRIC_PERFORMANCE`；
- **调整考试轮数**：修改 `app.py` 中 `EXAM_MAX_QUESTIONS`。

### 6.4 测试与 CI

测试使用 pytest（共 **73 项**），位于 `src/tests/`，覆盖病例加载、评分引擎、评估指标、不平衡处理、多模型评测解析、训练记录/会话持久化、评分报告导出、诊断剧透检测、评测可视化等模块。开发依赖位于 `requirements-dev.txt`（pytest、pytest-cov、ruff、mypy）：

```bash
cd src && python -m pytest tests/ -v                        # 全量测试
python -m pytest --cov=src/core --cov=src/eval tests/        # 附覆盖率报告
```

> GitHub Actions CI（`.github/workflows/ci.yml`）在每次 push / PR 时自动执行 lint（ruff + mypy）与带覆盖率阈值的测试。
>
> 评测与不平衡实验入口（需在 `src/` 目录下执行）：`python -m eval.benchmark --dataset x.json`、`python -m eval.imbalance`。
>
> 一键跑完 6 套 MCQ 数据集：`python run_benchmark_all.py`（需配置对应模型 API Key）。