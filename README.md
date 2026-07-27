# Hypergraph Discovery: HGNN vs Transformer on MIND-Small

A mathematical benchmark comparing **hypergraph neural message passing** with **pairwise self-attention** for the recovery and ranking of higher-order interaction structures in the **MIND-Small** news recommendation dataset.

The central question is:

> Does explicitly representing multi-entity relations as hyperedges recover bundle structure more effectively than reducing the same observations to pairwise attention interactions?

The benchmark evaluates:

* higher-order structural representation;
* top-(K) ranking quality;
* classification performance;
* optimisation behaviour;
* robustness across item-popularity regimes;
* computational reproducibility.

---

## 1. Problem Formulation

Let

[
\mathcal{V}={v_1,\ldots,v_n}
]

denote the set of entities, such as users, news articles, topics, categories, or behavioural states.

A conventional graph represents interactions using pairwise edges

[
\mathcal{E}_{2}\subseteq \mathcal{V}\times\mathcal{V}.
]

This assumes that every relevant relation can be decomposed into independent pairs.

The hypergraph model instead uses

[
\mathcal{H}=(\mathcal{V},\mathcal{E}),
]

where each hyperedge

[
e\in\mathcal{E},
\qquad
e\subseteq\mathcal{V},
]

may connect an arbitrary number of entities simultaneously.

Thus,

[
|e|\geq 2,
]

and potentially

[
|e|>2.
]

A behavioural bundle can therefore be represented directly as

[
e_k
===

{
u_i,
a_{j_1},
a_{j_2},
\ldots,
a_{j_m},
c_r,
t_s
},
]

where:

* (u_i) is a user;
* (a_{j_\ell}) is an interacted article;
* (c_r) is a category or topic;
* (t_s) is a contextual or temporal state.

The modelling objective is to learn a scoring function

[
f_{\theta}:
\mathcal{V}\times\mathcal{C}
\longrightarrow
\mathbb{R},
]

such that, for user (u_i) and candidate item (a_j),

[
s_{ij}
======

f_{\theta}(u_i,a_j)
]

induces a ranking

[
a_{i,(1)}
\succeq
a_{i,(2)}
\succeq
\cdots
\succeq
a_{i,(M)}.
]

The desired ordering satisfies

[
s_{ij}^{+}>s_{ik}^{-}
]

for relevant item (a_j^{+}) and non-relevant item (a_k^{-}).

---

## 2. Hypergraph Representation

### 2.1 Incidence Matrix

The hypergraph structure is encoded by the incidence matrix

[
\mathbf{H}\in{0,1}^{|\mathcal{V}|\times|\mathcal{E}|},
]

with entries

[
H_{ve}
======

\begin{cases}
1, & v\in e,\
0, & v\notin e.
\end{cases}
]

Unlike a graph adjacency matrix, (\mathbf{H}) preserves the identity of each multi-entity relation rather than replacing it with a collection of pairwise edges.

---

### 2.2 Vertex and Hyperedge Degrees

For hyperedge weights

[
\mathbf{W}
==========

\operatorname{diag}(w_1,\ldots,w_{|\mathcal{E}|}),
]

the vertex degree is

[
d(v)
====

\sum_{e\in\mathcal{E}}
w_eH_{ve},
]

and the hyperedge degree is

[
\delta(e)
=========

\sum_{v\in\mathcal{V}}
H_{ve}.
]

The corresponding diagonal degree matrices are

[
\mathbf{D}_v
============

\operatorname{diag}
\bigl(
d(v_1),\ldots,d(v_{|\mathcal{V}|})
\bigr),
]

and

[
\mathbf{D}_e
============

\operatorname{diag}
\bigl(
\delta(e_1),\ldots,\delta(e_{|\mathcal{E}|})
\bigr).
]

---

### 2.3 Normalised Hypergraph Propagation

Let

[
\mathbf{X}^{(0)}
\in
\mathbb{R}^{|\mathcal{V}|\times d_0}
]

be the initial feature matrix.

The normalised hypergraph propagation operator is

[
\mathbf{P}_{\mathcal{H}}
========================

\mathbf{D}_v^{-1/2}
\mathbf{H}
\mathbf{W}
\mathbf{D}_e^{-1}
\mathbf{H}^{\top}
\mathbf{D}_v^{-1/2}.
]

A hypergraph neural layer is then defined by

[
\mathbf{X}^{(\ell+1)}
=====================

\sigma
\left(
\mathbf{P}_{\mathcal{H}}
\mathbf{X}^{(\ell)}
\mathbf{\Theta}^{(\ell)}
+
\mathbf{b}^{(\ell)}
\right),
]

where:

* (\mathbf{X}^{(\ell)}) is the representation at layer (\ell);
* (\mathbf{\Theta}^{(\ell)}) is a trainable transformation;
* (\mathbf{b}^{(\ell)}) is a bias term;
* (\sigma) is a nonlinear activation.

For the implemented two-layer model,

[
\mathbf{X}^{(1)}
================

\operatorname{ReLU}
\left(
\mathbf{P}_{\mathcal{H}}
\mathbf{X}^{(0)}
\mathbf{\Theta}^{(0)}
+
\mathbf{b}^{(0)}
\right),
]

followed by

[
\mathbf{Z}_{\mathcal{H}}
========================

\mathbf{P}_{\mathcal{H}}
\mathbf{X}^{(1)}
\mathbf{\Theta}^{(1)}
+
\mathbf{b}^{(1)}.
]

The operator

[
\mathbf{H}^{\top}
\mathbf{D}_v^{-1/2}
]

aggregates vertex information into hyperedge states, while

[
\mathbf{D}_v^{-1/2}
\mathbf{H}
\mathbf{W}
\mathbf{D}_e^{-1}
]

returns the resulting higher-order information to the participating vertices.

Consequently,

[
\text{vertex}
\longrightarrow
\text{hyperedge}
\longrightarrow
\text{vertex}
]

is performed without destroying the original multi-entity relation.

---

## 3. Hypergraph Laplacian Interpretation

The normalised hypergraph Laplacian is

[
\mathbf{L}_{\mathcal{H}}
========================

## \mathbf{I}

\mathbf{P}_{\mathcal{H}}.
]

Therefore,

[
\mathbf{P}_{\mathcal{H}}
========================

## \mathbf{I}

\mathbf{L}_{\mathcal{H}}.
]

Hypergraph propagation may consequently be interpreted as a structure-aware smoothing operation over higher-order relational geometry.

For an embedding

[
\mathbf{Z}
==========

[\mathbf{z}*1,\ldots,\mathbf{z}*{|\mathcal{V}|}]^{\top},
]

the structural energy is

[
\mathcal{E}_{\mathcal{H}}(\mathbf{Z})
=====================================

\operatorname{Tr}
\left(
\mathbf{Z}^{\top}
\mathbf{L}_{\mathcal{H}}
\mathbf{Z}
\right).
]

Minimising this energy encourages entities participating in the same hyperedge to acquire compatible representations:

[
v_i,v_j\in e
\quad\Longrightarrow\quad
|\mathbf{z}_i-\mathbf{z}_j|_2
\text{ is controlled by the shared relational structure}.
]

This is stronger than assuming that the observed structure consists only of independent pairs.

---

## 4. Transformer Baseline

The Transformer baseline represents an interaction sequence as

[
\mathbf{S}
==========

[\mathbf{s}_1,\ldots,\mathbf{s}_m]^{\top}
\in
\mathbb{R}^{m\times d}.
]

For each attention head,

[
\mathbf{Q}
==========

\mathbf{S}\mathbf{W}_Q,
\qquad
\mathbf{K}
==========

\mathbf{S}\mathbf{W}_K,
\qquad
\mathbf{V}
==========

\mathbf{S}\mathbf{W}_V.
]

Scaled dot-product attention is

[
\operatorname{Attn}
(\mathbf{Q},\mathbf{K},\mathbf{V})
==================================

\operatorname{softmax}
\left(
\frac{\mathbf{Q}\mathbf{K}^{\top}}
{\sqrt{d_k}}
\right)
\mathbf{V}.
]

For (h) heads,

[
\operatorname{MHA}(\mathbf{S})
==============================

\operatorname{Concat}
\left(
\operatorname{head}_1,
\ldots,
\operatorname{head}_h
\right)
\mathbf{W}_O.
]

Each attention coefficient

[
\alpha_{ij}
===========

\frac{
\exp
\left(
\mathbf{q}_i^{\top}\mathbf{k}*j/\sqrt{d_k}
\right)
}{
\sum*{r=1}^{m}
\exp
\left(
\mathbf{q}_i^{\top}\mathbf{k}_r/\sqrt{d_k}
\right)
}
]

models a pairwise interaction between positions (i) and (j).

Although multiple layers can approximate complex dependencies, the primitive relation remains

[
(i,j),
]

whereas a hypergraph layer begins with the higher-order object

[
e={i_1,\ldots,i_r}.
]

The benchmark therefore compares:

[
\boxed{
\text{learned pairwise contextualisation}
}
]

against

[
\boxed{
\text{explicit higher-order relational propagation}
}.
]

---

## 5. Structural Difference

A bundle containing (r) entities is represented by the HGNN as one relation:

[
e
=

{v_1,\ldots,v_r}.
]

A pairwise reduction replaces it with up to

[
\binom{r}{2}
============

\frac{r(r-1)}{2}
]

edges.

However, the set of pairwise projections does not necessarily preserve the identity of the original bundle:

[
e
\not\equiv
\bigcup_{i<j}
(v_i,v_j).
]

The pairwise expansion records which entities co-occur, but may lose the fact that they participated in one common higher-order event.

The HGNN instead retains the factorisation

[
\mathcal{V}
\xrightarrow{\mathbf{H}^{\top}}
\mathcal{E}
\xrightarrow{\mathbf{H}}
\mathcal{V}.
]

This gives the model an explicit inductive bias for discovering structures such as:

[
\text{user}
+
\text{topic}
+
\text{article cluster}
+
\text{interaction context}.
]

---

## 6. Prediction Layer

Let

[
\mathbf{z}_{u_i}
\in
\mathbb{R}^{d}
]

be the learned user representation and

[
\mathbf{z}_{a_j}
\in
\mathbb{R}^{d}
]

the candidate-item representation.

A compatibility score may be written as

[
s_{ij}
======

\mathbf{z}*{u_i}^{\top}
\mathbf{z}*{a_j},
]

or, more generally,

[
s_{ij}
======

\mathbf{w}^{\top}
\phi
\left(
\mathbf{z}*{u_i},
\mathbf{z}*{a_j},
\mathbf{z}*{u_i}\odot\mathbf{z}*{a_j},
|\mathbf{z}*{u_i}-\mathbf{z}*{a_j}|
\right)
+b.
]

The predicted relevance probability is

[
\widehat{p}_{ij}
================

# \sigma(s_{ij})

\frac{1}{1+\exp(-s_{ij})}.
]

For binary relevance labels

[
y_{ij}\in{0,1},
]

the binary cross-entropy objective is

[
\mathcal{L}_{\mathrm{BCE}}
==========================

*

\frac{1}{N}
\sum_{(i,j)}
\left[
y_{ij}\log\widehat{p}*{ij}
+
(1-y*{ij})
\log(1-\widehat{p}_{ij})
\right].
]

With regularisation,

[
\mathcal{L}
===========

\mathcal{L}_{\mathrm{BCE}}
+
\lambda
|\theta|_2^2.
]

A pairwise ranking formulation can alternatively be expressed as

[
\mathcal{L}_{\mathrm{rank}}
===========================

*

\sum_{(i,j,k)}
\log
\sigma
\left(
s_{ij}^{+}-s_{ik}^{-}
\right).
]

---

## 7. Evaluation Metrics

### Hit Rate at (K)

For user (u_i), let

[
\mathcal{R}_i
]

be the relevant-item set and

[
\widehat{\mathcal{R}}_i^{(K)}
]

the top-(K) predicted set.

Then

[
\operatorname{HR@K}
===================

\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}
\left[
\mathcal{R}_i
\cap
\widehat{\mathcal{R}}_i^{(K)}
\neq
\varnothing
\right].
]

---

### Normalised Discounted Cumulative Gain at (K)

[
\operatorname{DCG@K}_i
======================

\sum_{r=1}^{K}
\frac{
2^{\operatorname{rel}_{i,r}}-1
}{
\log_2(r+1)
}.
]

The normalised score is

[
\operatorname{NDCG@K}_i
=======================

\frac{
\operatorname{DCG@K}_i
}{
\operatorname{IDCG@K}_i
}.
]

The reported aggregate is

[
\operatorname{NDCG@K}
=====================

\frac{1}{N}
\sum_{i=1}^{N}
\operatorname{NDCG@K}_i.
]

---

### Precision

[
\operatorname{Precision}
========================

\frac{\mathrm{TP}}
{\mathrm{TP}+\mathrm{FP}}.
]

### Recall

[
\operatorname{Recall}
=====================

\frac{\mathrm{TP}}
{\mathrm{TP}+\mathrm{FN}}.
]

### Accuracy

[
\operatorname{Accuracy}
=======================

\frac{
\mathrm{TP}+\mathrm{TN}
}{
\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}
}.
]

---

## 8. Benchmark Results

| Metric     |      HGNN |     Transformer | Relative HGNN Gain |
| :--------- | --------: | --------------: | -----------------: |
| HR@10      | **0.842** |           0.611 |          **37.8%** |
| NDCG@10    | **0.791** |           0.574 |          **37.8%** |
| Precision  | **0.900** |           0.650 |          **38.5%** |
| Recall     | **0.870** |           0.600 |          **45.0%** |
| Accuracy   | **0.900** |           0.650 |          **38.5%** |
| Final Loss | **0.214** | (\approx 0.750) |    **71.5% lower** |

For any metric (M) where larger values are preferred, the relative improvement is

[
\Delta_{\mathrm{rel}}(M)
========================

\frac{
M_{\mathrm{HGNN}}
-----------------

M_{\mathrm{Transformer}}
}{
M_{\mathrm{Transformer}}
}
\times 100%.
]

Hence,

[
\Delta_{\mathrm{rel}}(\operatorname{HR@10})
===========================================

\frac{0.842-0.611}{0.611}
\times 100%
\approx
37.8%,
]

and

[
\Delta_{\mathrm{rel}}(\operatorname{NDCG@10})
=============================================

\frac{0.791-0.574}{0.574}
\times 100%
\approx
37.8%.
]

For loss, where smaller values are preferred,

[
\Delta_{\mathrm{loss}}
======================

\frac{
0.750-0.214
}{
0.750
}
\times 100%
\approx
71.5%.
]

The benchmark therefore gives

[
\operatorname{HR@10}_{\mathrm{HGNN}}

>

\operatorname{HR@10}_{\mathrm{Transformer}},
]

[
\operatorname{NDCG@10}_{\mathrm{HGNN}}

>

\operatorname{NDCG@10}_{\mathrm{Transformer}},
]

and

[
\mathcal{L}*{\mathrm{HGNN}}
<
\mathcal{L}*{\mathrm{Transformer}}.
]

Full visual comparisons and metric summaries are available in:

[`results/summary.md`](results/summary.md)

---

## 9. Interpretation

The results support the hypothesis that the observations contain relevant structure of order greater than two:

[
\exists e\in\mathcal{E}
\quad\text{such that}\quad
|e|>2,
]

and that explicitly preserving this structure improves ranking quality.

The observed relation is

[
\text{explicit higher-order incidence structure}
\Longrightarrow
\text{improved bundle recovery}.
]

The benchmark does **not** imply that Transformers are universally weaker than hypergraph models.

Instead, it establishes that under this experimental construction,

[
\text{HGNN inductive bias}
]

is better aligned with the bundle-discovery task than the selected Transformer baseline.

The main empirical conclusion is therefore:

[
\boxed{
\text{When the target phenomenon is intrinsically multi-entity,}
\quad
\text{higher-order propagation can outperform pairwise attention.}
}
]

---

## 10. Popularity-Segment Robustness

Let item popularity be

[
\pi(a_j)
========

\sum_{i=1}^{N}
\mathbb{1}
\left[
u_i\text{ interacted with }a_j
\right].
]

Items may be partitioned into popularity strata

[
\mathcal{A}
===========

\mathcal{A}*{\mathrm{head}}
\cup
\mathcal{A}*{\mathrm{mid}}
\cup
\mathcal{A}_{\mathrm{tail}}.
]

Segment-specific performance is

[
M_s
===

M
\left(
\mathcal{A}_s
\right),
\qquad
s\in
{
\mathrm{head},
\mathrm{mid},
\mathrm{tail}
}.
]

This analysis tests whether the learned scoring function merely reproduces popularity:

[
s_{ij}
\propto
\pi(a_j),
]

or whether it recovers structural relevance beyond global frequency:

[
s_{ij}
======

f_{\theta}
\left(
u_i,
a_j,
\mathcal{H}
\right).
]

A structurally meaningful model should preserve useful performance when

[
\pi(a_j)
\downarrow,
]

particularly in the long-tail regime.

---

## 11. Architectural Summary

### Hypergraph Neural Network

[
\mathbf{X}^{(\ell+1)}
=====================

\sigma
\left(
\mathbf{D}_v^{-1/2}
\mathbf{H}
\mathbf{W}
\mathbf{D}_e^{-1}
\mathbf{H}^{\top}
\mathbf{D}_v^{-1/2}
\mathbf{X}^{(\ell)}
\mathbf{\Theta}^{(\ell)}
\right).
]

Primary inductive bias:

[
\text{multi-entity incidence}.
]

### Transformer

[
\mathbf{Z}
==========

\operatorname{softmax}
\left(
\frac{
\mathbf{Q}\mathbf{K}^{\top}
}{
\sqrt{d_k}
}
\right)
\mathbf{V}.
]

Primary inductive bias:

[
\text{learned pairwise contextual dependence}.
]

### Benchmark Contrast

[
\boxed{
\mathcal{H}
===========

(\mathcal{V},\mathcal{E})
}
\qquad
\text{versus}
\qquad
\boxed{
\mathbf{Q}\mathbf{K}^{\top}
}.
]

Equivalently,

[
\boxed{
\text{explicit relational order}
}
\qquad
\text{versus}
\qquad
\boxed{
\text{implicit relational approximation}
}.
]

---

## 12. Relevance to Structured AI and Cybersecurity

Although evaluated on recommendation data, the mathematical architecture is domain-independent.

A cybersecurity hyperedge may represent

[
e_{\mathrm{attack}}
===================

{
\text{account},
\text{device},
\text{process},
\text{IP address},
\text{credential},
\text{resource},
\text{time window}
}.
]

The malicious pattern may exist only at the level of the complete relation:

[
\operatorname{Risk}
\left(
e_{\mathrm{attack}}
\right)
\gg
\sum_{v\in e_{\mathrm{attack}}}
\operatorname{Risk}(v).
]

This occurs when individual entities appear benign but their joint configuration is anomalous.

The same incidence-based machinery can therefore support detection of:

* coordinated account behaviour;
* shared malicious infrastructure;
* campaign-level activity;
* multi-stage attack chains;
* distributed fraud;
* identity-device-resource collusion;
* higher-order agent state;
* relational context for autonomous security systems.

The general transfer is

[
\text{news bundle}
\longrightarrow
\text{entity bundle}
\longrightarrow
\text{coordinated security event}.
]

Thus, the project demonstrates a broader structured-intelligence principle:

[
\boxed{
\text{Do not collapse a higher-order event into isolated pairwise observations}
}
]

when the interaction itself carries the relevant information.

---

## 13. Reproducibility

The project is designed around:

[
\text{fixed data preparation}
+
\text{controlled model comparison}
+
\text{consistent evaluation}.
]

For seed (s), model parameters are initialised as

[
\theta_0
\sim
p(\theta\mid s).
]

A reproducible experiment requires

[
s_{\mathrm{Python}}
===================

# s_{\mathrm{NumPy}}

# s_{\mathrm{PyTorch}}

s.
]

The comparison is meaningful only when both systems use the same:

* training partition;
* validation partition;
* test partition;
* candidate-generation procedure;
* feature space;
* evaluation definitions;
* randomisation controls;
* stopping criteria.

Formally,

[
\mathcal{D}_{\mathrm{train}}^{\mathrm{HGNN}}
============================================

\mathcal{D}_{\mathrm{train}}^{\mathrm{Transformer}},
]

[
\mathcal{D}_{\mathrm{test}}^{\mathrm{HGNN}}
===========================================

\mathcal{D}_{\mathrm{test}}^{\mathrm{Transformer}},
]

and

[
\mathcal{M}_{\mathrm{eval}}^{\mathrm{HGNN}}
===========================================

\mathcal{M}_{\mathrm{eval}}^{\mathrm{Transformer}}.
]

---

## 14. Environment

The benchmark was developed using:

* Python 3.10.11
* PyTorch 2.1.0
* CUDA 12.4, optional
* Windows-native execution
* no Docker dependency
* no WSL dependency
* no Linux dependency

The CPU execution path remains available when CUDA is absent.

---

## 15. Installation

Clone the repository:

```bash
git clone https://github.com/Hars15kth/-HGNN-News.git
cd -HGNN-News
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Verify the PyTorch installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 16. Dataset

The benchmark uses the Microsoft News Dataset, MIND-Small:

[MIND: Microsoft News Dataset](https://msnews.github.io/)

Let the raw dataset be

[
\mathcal{D}
===========

{
(u_i,\mathcal{I}_i,\mathcal{C}*i,y_i)
}*{i=1}^{N},
]

where:

* (u_i) denotes a user;
* (\mathcal{I}_i) denotes interaction history;
* (\mathcal{C}_i) denotes candidate items;
* (y_i) denotes observed relevance labels.

The processed dataset induces a hypergraph

[
\mathcal{D}
\longmapsto
\mathcal{H}
===========

(\mathcal{V},\mathcal{E},\mathbf{X}).
]

The dataset is not redistributed by this repository. Download and use it in accordance with the original dataset terms.

---

## 17. Research Contribution

This repository contributes an empirical test of the proposition

[
\mathfrak{R}*{>2}
\not\subseteq
\mathfrak{R}*{2},
]

where:

* (\mathfrak{R}_{>2}) denotes directly represented higher-order relations;
* (\mathfrak{R}_{2}) denotes structures represented through pairwise interactions.

The practical hypothesis is

[
\mathcal{H}*{1}:
\quad
M
\left(
f*{\mathrm{HGNN}}
\right)

>

M
\left(
f_{\mathrm{Transformer}}
\right)
]

for bundle-sensitive ranking metrics

[
M
\in
{
\operatorname{HR@10},
\operatorname{NDCG@10}
}.
]

The reported results satisfy

[
M
\left(
f_{\mathrm{HGNN}}
\right)

>

M
\left(
f_{\mathrm{Transformer}}
\right)
]

for both primary ranking metrics.

The project therefore provides experimental evidence that:

[
\boxed{
\text{Representational geometry matters independently of model scale.}
}
]

A comparatively compact architecture can outperform a more general sequence model when its structural assumptions more accurately reflect the data-generating process.

---

## 18. Limitations

The results should be interpreted within the scope of the implemented benchmark.

They do not establish

[
f_{\mathrm{HGNN}}
\succ
f_{\mathrm{Transformer}}
]

for every dataset, architecture, or hyperparameter regime.

Performance depends on:

[
M
=

F
\left(
\mathcal{D},
\mathcal{H},
\theta,
\Omega,
\mathcal{S},
\mathcal{B}
\right),
]

where:

* (\mathcal{D}) is the dataset;
* (\mathcal{H}) is the structural construction;
* (\theta) is the parameterisation;
* (\Omega) is the optimisation procedure;
* (\mathcal{S}) is the sampling strategy;
* (\mathcal{B}) is the baseline configuration.

The correct conclusion is therefore benchmark-specific:

[
\boxed{
\text{Under the controlled experimental setup, the HGNN recovered}
}
]

[
\boxed{
\text{bundle-sensitive structure substantially better than the Transformer baseline.}
}
]

---

## 19. Author

**Harshwardhan Singh**

Mathematical AI researcher working on:

* graph and hypergraph learning;
* topological feature extraction;
* spectral optimisation;
* quantum-encoded machine learning;
* structured representations for autonomous decision systems;
* mathematical intelligence layers for cybersecurity agents.

GitHub: [github.com/Hars15kth](https://github.com/Hars15kth)

LinkedIn: [linkedin.com/in/harshwardhan-singh-2b1453318](https://linkedin.com/in/harshwardhan-singh-2b1453318)

---

## 20. Core Result

[
\boxed{
\begin{aligned}
\operatorname{HR@10}_{\mathrm{HGNN}}
&=
0.842

>

# 0.611

\operatorname{HR@10}*{\mathrm{Transformer}},
[4pt]
\operatorname{NDCG@10}*{\mathrm{HGNN}}
&=
0.791

>

# 0.574

\operatorname{NDCG@10}_{\mathrm{Transformer}}.
\end{aligned}
}
]

Therefore,

[
\boxed{
\text{explicit higher-order structure}
\quad

>

\quad
\text{pairwise attention baseline}
}
]

for the bundle-discovery task evaluated in this repository.

