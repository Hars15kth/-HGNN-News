# Hypergraph Discovery: HGNN vs Transformer on MIND-Small

A mathematical benchmark comparing a **Hypergraph Neural Network (HGNN)** with a **Transformer** for higher-order bundle discovery on the [MIND-Small](https://msnews.github.io/) dataset.

The project studies whether explicitly representing multi-entity relations through a hypergraph incidence structure produces better ranking performance than learning relations through pairwise self-attention.

The central hypothesis is:

$$
\text{explicit higher-order structure}
\quad \Longrightarrow \quad
\text{improved bundle discovery}
$$

The benchmark evaluates:

* Hit Rate at 10;
* Normalised Discounted Cumulative Gain at 10;
* precision;
* recall;
* accuracy;
* optimisation loss;
* robustness across item-popularity segments.

---

## 1. Mathematical Problem

Let the entity set be

$$
\mathcal{V} = {v_1,v_2,\ldots,v_n}.
$$

The entities may represent users, articles, topics, categories, or interaction states.

A conventional graph encodes pairwise relations:

$$
\mathcal{E}_2 \subseteq \mathcal{V}\times\mathcal{V}.
$$

Every relation therefore has the form

$$
(v_i,v_j).
$$

A hypergraph is defined as

$$
\mathcal{H} = (\mathcal{V},\mathcal{E}),
$$

where each hyperedge satisfies

$$
e \subseteq \mathcal{V}.
$$

Unlike an ordinary edge, a hyperedge may connect more than two entities:

$$
|e| \geq 2.
$$

For example, a behavioural bundle may be represented as

$$
e_k =
{
u_i,
a_{j_1},
a_{j_2},
\ldots,
a_{j_m},
c_r,
t_s
},
$$

where:

* $u_i$ is a user;
* $a_{j_\ell}$ is an interacted article;
* $c_r$ is a category or topic;
* $t_s$ is a temporal or contextual state.

The learning objective is to construct a scoring function

$$
f_\theta :
\mathcal{U}\times\mathcal{A}
\rightarrow
\mathbb{R},
$$

where

$$
s_{ij} = f_\theta(u_i,a_j)
$$

is the predicted compatibility between user $u_i$ and candidate article $a_j$.

For a relevant candidate $a_j^+$ and a non-relevant candidate $a_k^-$, the desired ordering is

$$
s_{ij}^+ > s_{ik}^-.
$$

---

## 2. Hypergraph Incidence Representation

The hypergraph is encoded by an incidence matrix

$$
\mathbf{H}
\in
{0,1}^{|\mathcal{V}|\times|\mathcal{E}|}.
$$

Its entries are defined by

$$
H_{ve}
$$
======
$$
\begin{cases}
1, & \text{when } v\in e, \
0, & \text{when } v\notin e.
\end{cases}
$$

The incidence matrix preserves the identity of each higher-order relation.

An adjacency matrix records whether two vertices are connected. The incidence matrix instead records which vertices participate in the same multi-entity event.

Let the hyperedge-weight matrix be

$$
\mathbf{W}
==========

\mathrm{diag}(w_1,w_2,\ldots,w_{|\mathcal{E}|}).
$$

The degree of a vertex $v$ is

$$
d(v)
====

\sum_{e\in\mathcal{E}}
w_e H_{ve}.
$$

The degree of a hyperedge $e$ is

$$
\delta(e)
=========

\sum_{v\in\mathcal{V}}
H_{ve}.
$$

The corresponding degree matrices are

$$
\mathbf{D}_v
============

\mathrm{diag}
\bigl(
d(v_1),d(v_2),\ldots,d(v_{|\mathcal{V}|})
\bigr)
$$

and

$$
\mathbf{D}_e
============

\mathrm{diag}
\bigl(
\delta(e_1),\delta(e_2),\ldots,\delta(e_{|\mathcal{E}|})
\bigr).
$$

---

## 3. Hypergraph Propagation Operator

The normalised hypergraph propagation operator is

$$
\mathbf{P}_{\mathcal{H}}
========================

\mathbf{D}_v^{-1/2}
\mathbf{H}
\mathbf{W}
\mathbf{D}_e^{-1}
\mathbf{H}^{\mathsf{T}}
\mathbf{D}_v^{-1/2}.
$$

Let

$$
\mathbf{X}^{(0)}
\in
\mathbb{R}^{|\mathcal{V}|\times d_0}
$$

be the initial entity-feature matrix.

A hypergraph neural layer is defined by

$$
\mathbf{X}^{(\ell+1)}
=====================

\sigma
\left(
\mathbf{P}_{\mathcal{H}}
\mathbf{X}^{(\ell)}
\mathbf{\Theta}^{(\ell)}
+
\mathbf{b}^{(\ell)}
\right).
$$

Here:

* $\mathbf{X}^{(\ell)}$ is the representation at layer $\ell$;
* $\mathbf{\Theta}^{(\ell)}$ is a trainable weight matrix;
* $\mathbf{b}^{(\ell)}$ is a bias vector;
* $\sigma$ is a nonlinear activation.

For the first layer,

$$
\mathbf{X}^{(1)}
================

\mathrm{ReLU}
\left(
\mathbf{P}_{\mathcal{H}}
\mathbf{X}^{(0)}
\mathbf{\Theta}^{(0)}
+
\mathbf{b}^{(0)}
\right).
$$

For the second layer,

$$
\mathbf{Z}_{\mathcal{H}}
========================

\mathbf{P}_{\mathcal{H}}
\mathbf{X}^{(1)}
\mathbf{\Theta}^{(1)}
+
\mathbf{b}^{(1)}.
$$

The propagation process is therefore

$$
\text{vertex}
\rightarrow
\text{hyperedge}
\rightarrow
\text{vertex}.
$$

The term

$$
\mathbf{H}^{\mathsf{T}}
\mathbf{D}_v^{-1/2}
\mathbf{X}^{(\ell)}
$$

aggregates vertex information into hyperedge representations.

The term

$$
\mathbf{D}_v^{-1/2}
\mathbf{H}
\mathbf{W}
\mathbf{D}_e^{-1}
$$

then distributes the hyperedge information back to participating vertices.

This allows information to propagate through a complete multi-entity relation without first decomposing it into independent pairs.

---

## 4. Hypergraph Laplacian

The normalised hypergraph Laplacian is

$$
\mathbf{L}_{\mathcal{H}}
========================

## \mathbf{I}

\mathbf{P}_{\mathcal{H}}.
$$

Equivalently,

$$
\mathbf{P}_{\mathcal{H}}
========================

## \mathbf{I}

\mathbf{L}_{\mathcal{H}}.
$$

For an embedding matrix

$$
\mathbf{Z}
==========

\begin{bmatrix}
\mathbf{z}_1^{\mathsf{T}} \
\mathbf{z}*2^{\mathsf{T}} \
\vdots \
\mathbf{z}*{|\mathcal{V}|}^{\mathsf{T}}
\end{bmatrix},
$$

the hypergraph structural energy is

$$
\mathcal{E}_{\mathcal{H}}(\mathbf{Z})
=====================================

\mathrm{Tr}
\left(
\mathbf{Z}^{\mathsf{T}}
\mathbf{L}_{\mathcal{H}}
\mathbf{Z}
\right).
$$

A low value of this energy encourages entities participating in the same hyperedge to obtain compatible representations.

Thus,

$$
v_i,v_j\in e
\quad \Longrightarrow \quad
|\mathbf{z}_i-\mathbf{z}_j|_2
\text{ is structurally constrained}.
$$

The constraint is determined by shared higher-order membership rather than only pairwise adjacency.

---

## 5. Transformer Baseline

Let an interaction sequence be represented by

$$
\mathbf{S}
==========

\begin{bmatrix}
\mathbf{s}_1^{\mathsf{T}} \
\mathbf{s}_2^{\mathsf{T}} \
\vdots \
\mathbf{s}_m^{\mathsf{T}}
\end{bmatrix}
\in
\mathbb{R}^{m\times d}.
$$

The query, key, and value matrices are

$$
\mathbf{Q}
==========

\mathbf{S}\mathbf{W}_Q,
$$

$$
\mathbf{K}
==========

\mathbf{S}\mathbf{W}_K,
$$

and

$$
\mathbf{V}
==========

\mathbf{S}\mathbf{W}_V.
$$

Scaled dot-product attention is

$$
\mathrm{Attention}
\left(
\mathbf{Q},
\mathbf{K},
\mathbf{V}
\right)
=======

\mathrm{softmax}
\left(
\frac{
\mathbf{Q}\mathbf{K}^{\mathsf{T}}
}{
\sqrt{d_k}
}
\right)
\mathbf{V}.
$$

The attention coefficient between positions $i$ and $j$ is

$$
\alpha_{ij}
===========

\frac{
\exp
\left(
\mathbf{q}_i^{\mathsf{T}}\mathbf{k}*j/\sqrt{d_k}
\right)
}{
\sum*{r=1}^{m}
\exp
\left(
\mathbf{q}_i^{\mathsf{T}}\mathbf{k}_r/\sqrt{d_k}
\right)
}.
$$

Each primitive attention coefficient therefore represents a pairwise relation:

$$
(i,j).
$$

Multiple attention layers may learn complex contextual dependencies, but their elementary interaction remains pairwise.

The mathematical comparison is therefore

$$
\mathbf{D}_v^{-1/2}
\mathbf{H}
\mathbf{W}
\mathbf{D}_e^{-1}
\mathbf{H}^{\mathsf{T}}
\mathbf{D}_v^{-1/2}
$$

versus

$$
\mathrm{softmax}
\left(
\frac{
\mathbf{Q}\mathbf{K}^{\mathsf{T}}
}{
\sqrt{d_k}
}
\right).
$$

The first operator begins from explicit higher-order incidence structure.

The second learns pairwise contextual weights from the input sequence.

---

## 6. Why Pairwise Reduction Can Lose Information

Consider a hyperedge containing $r$ entities:

$$
e = {v_1,v_2,\ldots,v_r}.
$$

A pairwise expansion replaces this hyperedge with up to

$$
\binom{r}{2}
============

\frac{r(r-1)}{2}
$$

ordinary edges.

However,

$$
e
\neq
\bigcup_{1\leq i<j\leq r}
(v_i,v_j)
$$

as relational objects.

The pairwise expansion records that individual pairs are connected, but it does not necessarily preserve the fact that all $r$ entities participated in one common event.

For example, the bundle

$$
{
\text{user},
\text{article},
\text{topic},
\text{session},
\text{time}
}
$$

contains a joint relation that is not identical to the set

$$
{
(\text{user},\text{article}),
(\text{user},\text{topic}),
(\text{article},\text{topic}),
\ldots
}.
$$

The HGNN preserves the factorisation

$$
\mathcal{V}
\xrightarrow{\mathbf{H}^{\mathsf{T}}}
\mathcal{E}
\xrightarrow{\mathbf{H}}
\mathcal{V}.
$$

This gives the model an explicit inductive bias for recovering multi-entity structure.

---

## 7. Prediction Function

Let

$$
\mathbf{z}_{u_i}
\in
\mathbb{R}^{d}
$$

be the learned representation of user $u_i$, and let

$$
\mathbf{z}_{a_j}
\in
\mathbb{R}^{d}
$$

be the learned representation of candidate article $a_j$.

A basic compatibility score is

$$
s_{ij}
======

\mathbf{z}*{u_i}^{\mathsf{T}}
\mathbf{z}*{a_j}.
$$

The corresponding relevance probability is

$$
\widehat{p}_{ij}
================

\frac{1}{1+\exp(-s_{ij})}.
$$

For binary labels

$$
y_{ij}\in{0,1},
$$

the binary cross-entropy loss is

$$
\mathcal{L}_{\mathrm{BCE}}
==========================

*

\frac{1}{N}
\sum_{(i,j)}
\left[
y_{ij}\log(\widehat{p}*{ij})
+
(1-y*{ij})\log(1-\widehat{p}_{ij})
\right].
$$

With quadratic regularisation,

$$
\mathcal{L}
===========

\mathcal{L}_{\mathrm{BCE}}
+
\lambda
|\theta|_2^2.
$$

A pairwise ranking objective may alternatively be written as

$$
\mathcal{L}_{\mathrm{rank}}
===========================

*

\sum_{(i,j,k)}
\log
\left[
\frac{
1
}{
1+\exp\left(-(s_{ij}^{+}-s_{ik}^{-})\right)
}
\right].
$$

---

## 8. Evaluation Metrics

### Hit Rate at 10

Let $\mathcal{R}_i$ be the relevant-item set for user $u_i$, and let $\widehat{\mathcal{R}}_i^{(10)}$ be the ten highest-ranked predictions.

Then

$$
\mathrm{HR@10}
==============

\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}
\left(
\mathcal{R}_i
\cap
\widehat{\mathcal{R}}_i^{(10)}
\neq
\varnothing
\right).
$$

### Discounted Cumulative Gain

For user $u_i$,

$$
\mathrm{DCG@10}_i
=================

\sum_{r=1}^{10}
\frac{
2^{\mathrm{rel}_{i,r}}-1
}{
\log_2(r+1)
}.
$$

### Normalised Discounted Cumulative Gain

$$
\mathrm{NDCG@10}_i
==================

\frac{
\mathrm{DCG@10}_i
}{
\mathrm{IDCG@10}_i
}.
$$

The mean score is

$$
\mathrm{NDCG@10}
================

\frac{1}{N}
\sum_{i=1}^{N}
\mathrm{NDCG@10}_i.
$$

### Precision

$$
\mathrm{Precision}
==================

\frac{
\mathrm{TP}
}{
\mathrm{TP}+\mathrm{FP}
}.
$$

### Recall

$$
\mathrm{Recall}
===============

\frac{
\mathrm{TP}
}{
\mathrm{TP}+\mathrm{FN}
}.
$$

### Accuracy

$$
\mathrm{Accuracy}
=================

\frac{
\mathrm{TP}+\mathrm{TN}
}{
\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}
}.
$$

---

## 9. Results

| Metric     |      HGNN |         Transformer |
| :--------- | --------: | ------------------: |
| HR@10      | **0.842** |               0.611 |
| NDCG@10    | **0.791** |               0.574 |
| Precision  | **0.900** |               0.650 |
| Recall     | **0.870** |               0.600 |
| Accuracy   | **0.900** |               0.650 |
| Final loss | **0.214** | approximately 0.750 |

For a performance metric $M$ where larger values are preferable, define the relative improvement as

$$
\Delta_M
========

\frac{
M_{\mathrm{HGNN}}
-----------------

M_{\mathrm{Transformer}}
}{
M_{\mathrm{Transformer}}
}
\times 100.
$$

For HR@10,

$$
\Delta_{\mathrm{HR@10}}
=======================

\frac{
0.842-0.611
}{
0.611
}
\times 100
\approx
37.8%.
$$

For NDCG@10,

$$
\Delta_{\mathrm{NDCG@10}}
=========================

\frac{
0.791-0.574
}{
0.574
}
\times 100
\approx
37.8%.
$$

For precision,

$$
\Delta_{\mathrm{Precision}}
===========================

\frac{
0.900-0.650
}{
0.650
}
\times 100
\approx
38.5%.
$$

For recall,

$$
\Delta_{\mathrm{Recall}}
========================

\frac{
0.870-0.600
}{
0.600
}
\times 100
==========

45.0%.
$$

Since lower loss is preferable, the relative loss reduction is

$$
\Delta_{\mathrm{Loss}}
======================

\frac{
0.750-0.214
}{
0.750
}
\times 100
\approx
71.5%.
$$

The primary empirical inequalities are

$$
0.842 > 0.611
$$

and

$$
0.791 > 0.574.
$$

Therefore,

$$
\mathrm{HR@10}_{\mathrm{HGNN}}

>

\mathrm{HR@10}_{\mathrm{Transformer}}
$$

and

$$
\mathrm{NDCG@10}_{\mathrm{HGNN}}

>

\mathrm{NDCG@10}_{\mathrm{Transformer}}.
$$

Full visual comparisons and experimental summaries are available in [`results/summary.md`](results/summary.md).

---

## 10. Experimental Interpretation

The benchmark supports the hypothesis that the dataset contains predictive relational structure of order greater than two:

$$
\exists e\in\mathcal{E}
\text{ such that }
|e|>2.
$$

The observed performance relation is

$$
\text{higher-order incidence modelling}
\rightarrow
\text{improved ranking quality}.
$$

This result does not imply that an HGNN is universally superior to every Transformer architecture.

The benchmark-specific conclusion is:

$$
\text{HGNN performance}

>

\text{selected Transformer baseline performance}
$$

under the implemented data construction, training procedure, and evaluation protocol.

The evidence indicates that the HGNN inductive bias was more closely aligned with the bundle-discovery structure of this task.

---

## 11. Popularity-Segment Robustness

Let the popularity of article $a_j$ be

$$
\pi(a_j)
========

\sum_{i=1}^{N}
\mathbf{1}
\left(
u_i
\text{ interacted with }
a_j
\right).
$$

The article set may be partitioned into

$$
\mathcal{A}
===========

\mathcal{A}*{\mathrm{head}}
\cup
\mathcal{A}*{\mathrm{middle}}
\cup
\mathcal{A}_{\mathrm{tail}}.
$$

A popularity-only scoring mechanism would behave approximately as

$$
s_{ij}
\propto
\pi(a_j).
$$

A relational scoring mechanism instead has the form

$$
s_{ij}
======

f_\theta
\left(
u_i,
a_j,
\mathcal{H}
\right).
$$

The popularity-segment analysis tests whether the model learns only global item frequency or preserves useful structural information for less frequent items.

---

## 12. Architectural Comparison

### HGNN

$$
\mathbf{X}^{(\ell+1)}
=====================

\sigma
\left(
\mathbf{D}_v^{-1/2}
\mathbf{H}
\mathbf{W}
\mathbf{D}_e^{-1}
\mathbf{H}^{\mathsf{T}}
\mathbf{D}_v^{-1/2}
\mathbf{X}^{(\ell)}
\mathbf{\Theta}^{(\ell)}
\right).
$$

Primary inductive bias:

$$
\text{explicit multi-entity incidence}.
$$

### Transformer

$$
\mathbf{Z}
==========

\mathrm{softmax}
\left(
\frac{
\mathbf{Q}\mathbf{K}^{\mathsf{T}}
}{
\sqrt{d_k}
}
\right)
\mathbf{V}.
$$

Primary inductive bias:

$$
\text{learned pairwise contextual dependence}.
$$

The comparison can therefore be summarised as

$$
\mathcal{H}=(\mathcal{V},\mathcal{E})
$$

versus

$$
\mathbf{Q}\mathbf{K}^{\mathsf{T}}.
$$

Equivalently,

$$
\text{explicit relational order}
$$

versus

$$
\text{implicit relational approximation}.
$$

---

## 13. Relevance to Structured AI and Cybersecurity

The mathematical architecture is not specific to recommendation data.

A cybersecurity hyperedge could be defined as

$$
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
$$

Each entity may appear individually benign, while the complete joint configuration is anomalous.

This can be expressed as

$$
\mathrm{Risk}
\left(
e_{\mathrm{attack}}
\right)

>

\sum_{v\in e_{\mathrm{attack}}}
\mathrm{Risk}(v).
$$

The higher-order relation therefore contains information not visible from isolated entity scores.

The same incidence-based learning architecture can support:

* coordinated-account detection;
* shared-infrastructure analysis;
* campaign-level behaviour discovery;
* multi-stage attack modelling;
* identity-device-resource correlation;
* distributed fraud detection;
* structured state representations for autonomous agents.

The domain transfer is

$$
\text{news bundle}
\rightarrow
\text{multi-entity behavioural bundle}
\rightarrow
\text{coordinated security event}.
$$

The general principle is:

$$
\text{Do not reduce a higher-order event to isolated pairwise observations}
$$

when the complete interaction carries the predictive information.

---

## 14. Reproducibility

A controlled benchmark requires both models to use the same dataset partitions:

$$
\mathcal{D}_{\mathrm{train}}^{\mathrm{HGNN}}
============================================

\mathcal{D}_{\mathrm{train}}^{\mathrm{Transformer}},
$$

$$
\mathcal{D}_{\mathrm{validation}}^{\mathrm{HGNN}}
=================================================

\mathcal{D}_{\mathrm{validation}}^{\mathrm{Transformer}},
$$

and

$$
\mathcal{D}_{\mathrm{test}}^{\mathrm{HGNN}}
===========================================

\mathcal{D}_{\mathrm{test}}^{\mathrm{Transformer}}.
$$

The models must also use the same evaluation function:

$$
\mathcal{M}_{\mathrm{eval}}^{\mathrm{HGNN}}
===========================================

\mathcal{M}_{\mathrm{eval}}^{\mathrm{Transformer}}.
$$

For a fixed random seed $s$,

$$
s_{\mathrm{Python}}
===================

# s_{\mathrm{NumPy}}

# s_{\mathrm{PyTorch}}

s.
$$

The implementation is designed around:

* deterministic data preparation;
* controlled model comparison;
* fixed evaluation definitions;
* explicit artefact generation;
* minimal external dependencies.

---

## 15. Environment

* Python 3.10.11
* PyTorch 2.1.0
* CUDA 12.4, optional
* Windows-native execution
* no Docker requirement
* no WSL requirement
* CPU-compatible execution path

---

## 16. Installation

Clone the repository:

```bash
git clone https://github.com/Hars15kth/-HGNN-News.git
cd -HGNN-News
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify PyTorch and CUDA availability:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 17. Dataset

The project uses the [MIND-Small dataset](https://msnews.github.io/).

Let the raw interaction dataset be

$$
\mathcal{D}
===========

{
(u_i,\mathcal{I}_i,\mathcal{C}*i,y_i)
}*{i=1}^{N},
$$

where:

* $u_i$ is a user;
* $\mathcal{I}_i$ is the interaction history;
* $\mathcal{C}_i$ is the candidate-item set;
* $y_i$ is the relevance label.

The preprocessing pipeline constructs a hypergraph:

$$
\mathcal{D}
\rightarrow
\mathcal{H}
===========

(\mathcal{V},\mathcal{E},\mathbf{X}).
$$

The dataset is not redistributed in this repository. Download it from the official MIND website and follow its original licence and usage conditions.

---

## 18. Research Contribution

The project tests whether directly represented higher-order relations provide a stronger inductive bias than pairwise attention for bundle-sensitive ranking.

The primary hypothesis is

$$
M(f_{\mathrm{HGNN}})

>

M(f_{\mathrm{Transformer}})
$$

for

$$
M
\in
{
\mathrm{HR@10},
\mathrm{NDCG@10}
}.
$$

The observed results satisfy both inequalities:

$$
\mathrm{HR@10}(f_{\mathrm{HGNN}})

>

\mathrm{HR@10}(f_{\mathrm{Transformer}})
$$

and

$$
\mathrm{NDCG@10}(f_{\mathrm{HGNN}})

>

\mathrm{NDCG@10}(f_{\mathrm{Transformer}}).
$$

The project therefore provides evidence for the principle

$$
\text{representational structure matters independently of model generality}.
$$

A structurally specialised model can outperform a more general sequence architecture when its mathematical representation more closely matches the data-generating process.

---

## 19. Limitations

The results do not prove that

$$
f_{\mathrm{HGNN}}

>

f_{\mathrm{Transformer}}
$$

for every dataset, architecture, or hyperparameter configuration.

Observed performance depends on

$$
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
$$

where:

* $\mathcal{D}$ is the dataset;
* $\mathcal{H}$ is the hypergraph construction;
* $\theta$ is the model parameterisation;
* $\Omega$ is the optimisation procedure;
* $\mathcal{S}$ is the sampling strategy;
* $\mathcal{B}$ is the baseline configuration.

The correct conclusion is limited to the implemented experiment:

$$
\text{HGNN}

>

\text{selected Transformer baseline}
$$

for the reported bundle-discovery metrics under the controlled benchmark configuration.

---

## 20. Author

**Harshwardhan Singh**

Mathematical AI researcher working across:

* graph and hypergraph learning;
* topological feature extraction;
* spectral optimisation;
* quantum-encoded machine learning;
* structured representations;
* autonomous decision systems;
* mathematical intelligence layers for cybersecurity.

GitHub: [github.com/Hars15kth](https://github.com/Hars15kth)

LinkedIn: [linkedin.com/in/harshwardhan-singh-2b1453318](https://linkedin.com/in/harshwardhan-singh-2b1453318)

---

## Core Result

$$
\mathrm{HR@10}_{\mathrm{HGNN}}
==============================

0.842

>

# 0.611

\mathrm{HR@10}_{\mathrm{Transformer}}.
$$

$$
\mathrm{NDCG@10}_{\mathrm{HGNN}}
================================

0.791

>

# 0.574

\mathrm{NDCG@10}_{\mathrm{Transformer}}.
$$

Therefore, for this benchmark,

$$
\text{explicit higher-order structure}

>

\text{pairwise-attention baseline}.
$$


