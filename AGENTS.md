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

当前真正想验证的是：
- 在不修改文档索引、不进行初次文档检索的条件下，利用查询自身表示以及冻结知识库的几何结构，预测查询在共享修正子空间中的移动方向。
- 同一个修正模型面对不同知识库时，因为知识库结构不同，对相同/相似 query 产生不同修正。应该做跨知识库共享模型。

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