## 1. 当前研究目标

当前正在研究传统 Dense Retrieval / RAG 中的 Query Representation Calibration。

核心系统约束已经确定为：

```text
Query
  ↓
Pretrained Embedding Model
  ↓
q
  ↓
Query-side Correction / Adapter
  ↓
q'
  ↓
Vector DB Retrieval
```

要求：

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

\[
q'=C_\theta(q)
\]

是否能够在保持原 document embedding space 不变的情况下提高 retrieval quality。

---

# 6. 已观察到的 Geometry 现象

这部分目前只是实验事实，不需要急着形成论文故事。

## Positive similarity

FiQA Qwen3：

```text
Hit        0.6831
Near       0.5489
Hard       0.4422
```

FiQA BGE：

```text
Hit        0.6843
Near       0.6022
Hard       0.5153
```

两个模型都有：

\[
PositiveSim(Hit)
>
PositiveSim(Near)
>
PositiveSim(Hard)
\]

而且统计差异明显。

---

## Top1–Top10 margin

Qwen3：

```text
Hit        0.1193
Near       0.0708
Hard       0.0599
```

BGE：

```text
Hit        0.0781
Near       0.0495
Hard       0.0461
```

Margin 对：

```text
Hit vs Failure
```

有一定判别能力。

---

## Local Distance

Qwen3：

```text
Hit        0.3823
Near       0.3900
Hard       0.4123
```

BGE：

```text
Hit        0.3594
Near       0.3578
Hard       0.3790
```

重要现象：

\[
LocalDistance(Hit)
\approx
LocalDistance(Near)
\]

而 Hard-miss 才明显增加。

因此目前没有证据支持简单的：

> Near-miss query 只是整体距离 document manifold 更远。

---

# 7. Deployable Score Features 的结果

结合仅基于 retrieval score geometry 判断：

- FiQA 上 margin / score std 有较强信号
- ArguAna 上基本退化到接近随机
- local distance 基本没有可靠 detection 能力

由于现在已经确定线上架构：

```text
Embedding → Adapter → 一次 Retrieval
```

因此这些需要第一次 retrieval 才能获得的特征**不是主方案输入**。

可以保留作为分析结果，不需要继续投入大量时间扩展手工 score features。

---

# 9. 当前真正考虑的 Adapter 结构

不考虑 PRF / Top-K feedback。

目标：

\[
q=E_q(x)
\]

然后：

\[
q'=C_\theta(q)
\]

一次检索。

最基本版本：

\[
q'=Normalize(q+A_\theta(q))
\]

但纯自由 MLP 与已有 Query Adapter 工作过于接近。

当前更感兴趣的是：

> **让 Document / Query→Document correction geometry 提供一个固定修正子空间，Adapter 只预测低维系数。**

形式：

\[
q'
=
Normalize
\left(
q+
g_\theta(q)
[
\mu_\Delta+B a_\theta(q)
]
\right)
\]

其中：

- \(B\)：固定低维修正 basis
- \(a_\theta(q)\)：query-dependent coefficients
- \(g_\theta(q)\)：可选 gate / correction magnitude
- \(\mu_\Delta\)：平均 correction direction

这样 inference 仍然只是：

```text
Query
 ↓
Embedding
 ↓
Coefficient / Gate Predictor
 ↓
Low-rank Correction
 ↓
Single Retrieval
```
