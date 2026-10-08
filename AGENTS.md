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

本研究旨在探索：

先在有标注 benchmark 上寻找一种能够从 query–document ranking关系中稳定定义 retrieval error 的 supervision，再研究如何用 synthetic queries 在无标注知识库中重构这种 supervision，从而自动训练 index-preserving query calibrator。

给定 frozen embedding model、已有 document vectors/index 和原始文本，在无人工 qrels 的情况下，能否自动构造有效检索监督，训练一个轻量 query-only adapter，使真实查询的检索质量稳定提升？

注意：

不同 embedding 模型的训练协议、query/document 角色建模方式和空间几何差异，避免把“跨知识库泛化”和“跨 embedding 空间泛化”混在一起。
## 开发要求
- 保持以下模块独立：
- 新增模型或数据集时，避免在主 pipeline 中增加大量模型名或数据集名判断。
- 不过度抽象，只为已经存在的实验需求建立抽象。避免：提前设计复杂插件体系/为未来假设增加大量配置/重复包装成熟库/将 retrieval、analysis、plotting 混入单一大型模块。
- 代码应优先保持简单、可读、可验证。
- 重构过程无需考虑旧数据兼容，禁止为了旧实现加兼容层。