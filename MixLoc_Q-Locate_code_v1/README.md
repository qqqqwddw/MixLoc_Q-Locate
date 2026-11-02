# MixLoc / Q-Locate 代码发布包（v1）

**MixLoc: A Multilingual Benchmark for Mixed Human–AI Authorship with Provenance-Supervised Localization**

Q-Locate 检测/定位框架的完整源码与评测分析脚本。

> **数据集及其他材料：如需请联系 ____________（邮箱）。**

## 目录结构

```
├── core_model/        Q-Locate v1.1 引擎源码
│   ├── model.py       架构：字节输入 → 冻结 Qwen3-4B（SVD r16 适配）→ token/字符/边界/文档头 → 多尺度证据扫描
│   ├── losses.py      五项损失：L = ce + loc + 0.2·dice + 0.1·boundary_bce + 0.025·boundary_rank
│   ├── decode.py      两态 DP 结构化解码（float64）
│   ├── engine.py      DDP 训练引擎（梯度同步/重放）
│   ├── evaluate.py    评测：raw/decoded/gated/e2e 四路径 + 逐篇数组导出
│   ├── train.py       训练入口（exact-once 单 epoch）
│   └── supervise.py   训练监督工具
└── analysis/          评测与分析脚本（CPU 可运行）
    ├── independent_recompute.py    指标独立复算
    ├── stratified_analysis.py      coverage/generator/language 分层
    ├── decodecmp.py / decode_lib.py  解码阶梯对照
    ├── p1a2_full.py / p1a2_recon.py  解码一致性复核
    ├── audit3.py / audit3b.py      边界场扰动 / 单区间解码 / 句界负例
    ├── logmine.py                  训练动态 + 边界场校准
    ├── dataset_evidence.py         数据集统计（逐语言/逐操作/生成器构成）
    ├── s3_s4_analysis.py           生成多样性 + 语言错配敏感性
    ├── binary_metrics.py           二分类口径（AUROC/BACC/F1/Spec）
    └── a0_runner.py                SVD 文档导向基线
```

## 环境

- Python ≥ 3.11，PyTorch ≥ 2.x（训练需 CUDA；分析脚本仅 CPU + numpy）
- 模型底座：Qwen3-4B（HuggingFace，bf16）；SVD r16 适配，12,400,666 可训练参数
- 训练合同（完全固定）：AdamW lr 1e-4 / warmup 42 + cosine / clip 1.0 / seed 0 / 单滚动 checkpoint / 固定终点（无验证集挑选）

## 复现路径

1. 训练：`core_model/train.py`（exact-once，约 24h @ 4×RTX 4090；`runtime.py` 运行时工具随数据集联系获取）
2. 评测：`core_model/evaluate.py` → 四路径指标 + 逐篇 FP32 数组导出
3. 验证与分析：`analysis/` 各脚本（指标复算、分层、解码对照、机制分析）

## 许可

代码仅供研究用途（research, non-commercial）。

## 联系

数据集、代码或补充材料：如需请联系 ____________（邮箱）
