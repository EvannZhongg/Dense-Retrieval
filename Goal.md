## 1. 当前研究目标

当前正在研究传统 Dense Retrieval / RAG 中的 Query Representation Calibration。要求：

- **线上只进行一次检索**
- 不采用：
  - 首轮 Top-K → PRF → 第二轮检索
  - rerank 后再检索
  - Top-K centroid feedback 作为主方案
- Pretrained embedding model 尽量冻结
- Document embeddings 冻结
- 已建立的 vector index 不重新构建
- Correction 只发生在 query embedding 侧

当前真正想验证的是：在不修改文档索引、不进行初次文档检索的条件下，利用查询自身表示以及冻结知识库的压缩几何结构，预测查询在共享修正子空间中的移动方向。

重构过程无需考虑旧数据兼容，禁止为了旧实现加兼容层。