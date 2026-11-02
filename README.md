# MixLoc / Q-Locate 代码发布包（v1）

**MixLoc: A Multilingual Benchmark for Mixed Human–AI Authorship with Provenance-Supervised Localization**

本项目包含两部分成果：（1）**MixLoc**——大规模多语言混合作者身份检测与字符级定位基准；（2）**Q-Locate**——在其上构建的统一检测定位框架（冻结 Qwen3-4B + SVD 适配 + 字符级来源证据 + 边界感知结构化解码）。本包提供 Q-Locate 的完整源码与全套评测分析脚本。

## 一、数据集下载

| 项 | 内容 |
|---|---|
| 数据集名称 | MixLoc（v2 冻结版） |
| 规模 | 601,846 篇文档 / 20 种语言 / 6 个测试专属生成器 |
| 切分 | train / val / test = 366,618 / 32,466 / 202,762（附 SHA-256 清单） |
| 标注 | 三级来源标注：构造期字符级 span 记录 / 外部继承+编译器校验 / 文档级隐式 |
| **下载链接** | **百度网盘：https://pan.baidu.com/s/11UcWdeXMJ54fgFcW6Yrpkg?pwd=nbk5** |

> 数据集及其他材料（数据准备脚本、`runtime.py` 运行时工具、schema 文档等）：如网盘链接失效或有任何问题，请联系 ____________（邮箱）。

### 数据字段说明（核心字段）

| 字段 | 含义 |
|---|---|
| `row_id_v2` | 文档唯一 ID |
| `text` | 文档全文（Unicode，字符坐标即标注坐标） |
| `label` | HUMAN / PARTIAL_AI / FULL_AI |
| `machine_spans` | AI 生成内容的字符区间列表 `[s, e)`（半开，Unicode 码点） |
| `span_completeness` | 标注完备性标记（如 EXHAUSTIVE_BY_CONSTRUCTION） |
| `operation` | 构造操作类型（sentence_replacement / sentence_insertion / multi_sentence_insertion / style_constrained_rewrite / continuation / full_document_generation） |
| `language` / `generator` | 语言 / 生成器标识 |
| `source_group_hash` | 来源分组哈希（防泄漏切分键） |
| `raw_lineage_locator` | 溯源定位（指向构造/来源记录） |

## 二、目录结构

```
MixLoc_Q-Locate_code_v1/
├── README.md                       本文件
├── core_model/                     Q-Locate v1.1 引擎源码（与论文方法章对应）
│   ├── model.py                    模型架构：字节级输入（257 符号嵌入）→ 冻结 Qwen3-4B
│   │                                 （144 attention 投影 SVD rank-16 适配，FP32 尾部合并）
│   │                                 → token 概率 / 128 维投影 / 字符特征（708 维）
│   │                                 → source 头（α）+ boundary 头（b、相对边界场 a）
│   │                                 → 多尺度 Top-4 证据扫描 → 文档双分支 logits
│   ├── losses.py                   五项损失：L = ce + loc + 0.2·dice + 0.1·boundary_bce
│   │                                 + 0.025·boundary_rank（类归一化 CE；PARTIAL 内外
│   │                                 分组 BCE；Dice；端点位置 BCE ±8 字符软排序）
│   ├── decode.py                   两态（人/机）DP 结构化解码：O(n)、float64、β=0.25、
│   │                                 机器态平局规则；输入 α 与边界场 a，输出精确区间
│   ├── engine.py                   DDP 训练引擎（显式梯度同步、紧凑重放、显存分析）
│   ├── evaluate.py                 评测：raw / decoded / gated / e2e 四路径指标
│   │                                 + 逐篇 FP32 数组导出（供独立复算）
│   ├── train.py                    训练入口（exact-once 调度、原子 checkpoint、分组梯度记录）
│   └── supervise.py                训练监督工具（进度/健康检查）
└── analysis/                       评测与分析脚本（全部 CPU 可运行，输入为数据集+导出数组）
    ├── independent_recompute.py    指标独立复算（验证论文全部数字）
    ├── stratified_analysis.py      coverage / generator / language 分层评测
    ├── decode_lib.py               共享加载与度量库（字符计数、IoU 贪心匹配、MAE）
    ├── decodecmp.py                解码阶梯对照（raw/句界吸附/平滑/常数惩罚 DP/学习边界 DP/gated）
    ├── p1a2_recon.py               解码实现一致性核查（抽样对比引擎输出）
    ├── p1a2_full.py                全量解码复核 + 修复版阶梯（Val 选参冻结 → Test）
    ├── audit3.py                   区间数审计 / 句界匹配困难负例 / 边界场扰动 / 单区间 Kadane
    ├── audit3b.py                  单区间解码去边界场对照 + 终点通道标点后负例
    ├── logmine.py                  训练分项损失轨迹 + 边界场 AUC/P@k 校准（三检查点）
    ├── dataset_evidence.py         数据集统计（逐语言×切分 / 逐操作覆盖率与定位难度 / 生成器构成）
    ├── s3_s4_analysis.py           生成内容多样性统计 + 语言错配分布与敏感性
    ├── binary_metrics.py           二分类口径（HUMAN vs AI-involved：AUROC/BACC/F1(AI)/Spec）
    └── a0_runner.py                SVD 文档导向基线（matched-budget 系统对照）
```

## 三、环境与依赖

- Python ≥ 3.11；PyTorch ≥ 2.x（训练需 CUDA，约 4×24GB 显存；分析脚本仅需 CPU + numpy）
- 模型底座：Qwen3-4B（HuggingFace，bf16；hidden 2560 / 36 层 / vocab 151,936）
- 可训练参数：**12,400,666**（0.3%；SVD 残差因子 + 轻量头 + 字节嵌入）
- `train.py` 依赖的 `runtime.py`（checkpoint/加载工具）与数据准备脚本：见上文联系方式获取

## 四、训练与复现

**训练合同（完全固定，可复现）**：

| 项 | 值 |
|---|---|
| 数据 | MixLoc train 366,618 篇，exact-once 单 epoch（1,432 步 × 256 篇 + 26 篇收尾批） |
| 优化器 | AdamW，lr 1e-4，warmup 42 步 + 余弦退火至 0，梯度裁剪 1.0 |
| 精度 | bf16 autocast + 非重入梯度检查点；SVD 尾部因子与损失 FP32；TF32 关闭 |
| 硬件 | 4×RTX 4090，约 24 小时 |
| checkpoint | 单滚动原子保存；**最终模型 = 固定训练终点**（全程无验证集挑选） |

**复现路径**：
1. `core_model/train.py` —— 训练；
2. `core_model/evaluate.py` —— 评测（四路径指标 + 逐篇数组导出）；
3. `analysis/independent_recompute.py` —— 对导出数组独立重算全部指标（论文数字的验证器）；
4. `analysis/` 其余脚本 —— 分层评测、解码对照与机制分析（对应论文实验章各小节）。


## 六、许可

- 代码：仅供研究用途（research, non-commercial）
- 数据集：随网盘发布附最终许可条款；外部继承子集受上游语料许可约束

## 七、联系

数据集、代码或补充材料（训练日志、审计文档、`runtime.py` 等）：如需请联系作者
