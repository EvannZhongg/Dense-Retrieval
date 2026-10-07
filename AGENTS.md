# AGENTS.md

## Project Purpose

本项目用于究 研究传统 Dense Retrieval / RAG 中的 Query Representation Calibration。要求：

- **线上只进行一次检索**
- 不采用：
  - 首轮 Top-K → PRF → 第二轮检索
  - rerank 后再检索
  - Top-K centroid feedback 作为主方案
- Pretrained embedding model 尽量冻结
- Document embeddings 冻结
- 已建立的 vector index 不重新构建
- Correction 只发生在 query embedding 侧

本研究旨在探索一种无需修改文档索引、无需依赖初始检索结果的查询表示修正方法。对于给定查询，仅利用其原始 embedding 与冻结知识库的全局几何结构，预测查询在一个共享低维修正子空间中的移动方向与幅度，使修正后的查询表示更接近当前知识库中与其相关的文档区域。

进一步地，我们希望将“如何修正查询”与“具体知识库”解耦：使用一个可跨知识库共享的修正模型，通过感知不同知识库的几何结构，对相同或相似查询产生知识库条件化的修正。因此，模型学习的不是某个固定语料库上的查询偏移，而是一种从“查询表示 + 知识库结构”到“查询修正方向”的通用映射，使其能够迁移到不同知识库而无需重新训练完整检索器或修改已有文档索引。

### 2. Dataset / Model / Evaluation 解耦

保持以下模块独立：

```text
Dataset Adapter
Embedding Adapter
Retrieval
Evaluation
Analysis
```

新增模型或数据集时，应通过 adapter / config 接入，避免在主 pipeline 中增加大量模型名或数据集名判断。

### 3. 不过度抽象

只为已经存在的实验需求建立抽象。

避免：

- 提前设计复杂插件体系；
- 为未来假设增加大量配置；
- 重复包装成熟库；
- 将 retrieval、analysis、plotting 混入单一大型模块。

代码应优先保持简单、可读、可验证。

重构过程无需考虑旧数据兼容，禁止为了旧实现加兼容层。