# Hypergraph Discovery: HGNN vs Transformer on MIND-Small

A mathematical benchmark comparing a **Hypergraph Neural Network (HGNN)** with a **Transformer** for higher-order bundle discovery on the [MIND-Small](https://msnews.github.io/) dataset.

The project investigates whether explicitly representing multi-entity relations through hypergraph incidence geometry produces stronger ranking performance than learning relational structure through pairwise self-attention.

The central hypothesis is:

$$\text{Explicit higher-order structure}\Longrightarrow\text{improved bundle discovery}.$$

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

$$\mathcal{V}={v_1,v_2,\ldots,v_n}.$$

The entities may represent users, articles, topics, categories, or interaction states.

A conventional graph encodes pairwise relations through

$$\mathcal{E}_2\subseteq\mathcal{V}\times\mathcal{V}.$$

Every primitive relation therefore has the form

$$(v_i,v_j).$$

A hypergraph is defined as

$$\mathcal{H}=(\mathcal{V},\mathcal{E}),$$

where every hyperedge satisfies

$$e\subseteq\mathcal{V}.$$

Unlike an ordinary edge, a hyperedge may connect more than two entities:

$$|e|\geq2.$$

A behavioural bundle may therefore be represented as

$$e_k={u_i,a_{j_1},a_{j_2},\ldots,a_{j_m},c_r,t_s},$$

where:

* $u_i$ is a user;
* $a_{j_\ell}$ is an interacted article;
* $c_r$ is a category or topic;
* $t_s$ is a temporal or contextual state.

The learning objective is to construct a scoring function

$$f_\theta:\mathcal{U}\times\mathcal{A}\longrightarrow\mathbb{R}.$$

For user $u_i$ and candidate article $a_j$, the predicted compatibility score is

$$s_{ij}=f_\theta(u_i,a_j).$$

For a relevant candidate $a_j^{+}$ and a non-relevant candidate $a_k^{-}$, the desired ordering is

$$s_{ij}^{+}>s_{ik}^{-}.$$

---

## 2. Hypergraph Incidence Representation

The hypergraph is encoded by an incidence matrix

$$\mathbf{H}\in{0,1}^{|\mathcal{V}|\times|\mathcal{E}|}.$$

Its entries are defined by

$$H_{ve}=\begin{cases}1,&v\in e,\0,&v\notin e.\end{cases}$$

The incidence matrix preserves the identity of each higher-order relation.

An adjacency matrix records whether two vertices are pairwise connected. The incidence matrix instead records which vertices participate in the same multi-entity event.

Let the hyperedge-weight matrix be

$$\mathbf{W}=\mathrm{diag}(w_1,w_2,\ldots,w_{|\mathcal{E}|}).$$

The degree of a vertex $v$ is

$$d(v)=\sum_{e\in\mathcal{E}}w_eH_{ve}.$$

The degree of a hyperedge $e$ is

$$\delta(e)=\sum_{v\in\mathcal{V}}H_{ve}.$$

The vertex-degree matrix is

$$\mathbf{D}*v=\mathrm{diag}(d(v_1),d(v_2),\ldots,d(v*{|\mathcal{V}|})).$$

The hyperedge-degree matrix is

$$\mathbf{D}*e=\mathrm{diag}(\delta(e_1),\delta(e_2),\ldots,\delta(e*{|\mathcal{E}|})).$$

---

## 3. Hypergraph Propagation Operator

The normalised hypergraph propagation operator is

$$\mathbf{P}_{\mathcal{H}}=\mathbf{D}_v^{-1/2}\mathbf{H}\mathbf{W}\mathbf{D}_e^{-1}\mathbf{H}^{\mathsf{T}}\mathbf{D}_v^{-1/2}.$$

Let the initial entity-feature matrix be

$$\mathbf{X}^{(0)}\in\mathbb{R}^{|\mathcal{V}|\times d_0}.$$

A general hypergraph neural layer is

$$\mathbf{X}^{(\ell+1)}=\sigma\bigl(\mathbf{P}_{\mathcal{H}}\mathbf{X}^{(\ell)}\mathbf{\Theta}^{(\ell)}+\mathbf{b}^{(\ell)}\bigr).$$

Here:

* $\mathbf{X}^{(\ell)}$ is the representation at layer $\ell$;
* $\mathbf{\Theta}^{(\ell)}$ is a trainable weight matrix;
* $\mathbf{b}^{(\ell)}$ is a bias vector;
* $\sigma$ is a nonlinear activation.

For the first layer,

$$\mathbf{X}^{(1)}=\mathrm{ReLU}\bigl(\mathbf{P}_{\mathcal{H}}\mathbf{X}^{(0)}\mathbf{\Theta}^{(0)}+\mathbf{b}^{(0)}\bigr).$$

For the second layer,

$$\mathbf{Z}*{\mathcal{H}}=\mathbf{P}*{\mathcal{H}}\mathbf{X}^{(1)}\mathbf{\Theta}^{(1)}+\mathbf{b}^{(1)}.$$

The propagation process is

$$\text{vertex}\longrightarrow\text{hyperedge}\longrightarrow\text{vertex}.$$

The vertex-to-hyperedge aggregation term is

$$\mathbf{X}_{\mathcal{E}}^{(\ell)}=\mathbf{D}_e^{-1}\mathbf{H}^{\mathsf{T}}\mathbf{D}_v^{-1/2}\mathbf{X}^{(\ell)}.$$

The hyperedge-to-vertex propagation term is

$$\mathbf{X}_{\mathcal{V}}^{(\ell+1)}=\mathbf{D}*v^{-1/2}\mathbf{H}\mathbf{W}\mathbf{X}*{\mathcal{E}}^{(\ell)}.$$

Combining the two stages gives

$$\mathbf{X}_{\mathcal{V}}^{(\ell+1)}=\mathbf{D}_v^{-1/2}\mathbf{H}\mathbf{W}\mathbf{D}_e^{-1}\mathbf{H}^{\mathsf{T}}\mathbf{D}_v^{-1/2}\mathbf{X}^{(\ell)}.$$

This allows information to propagate through a complete multi-entity relation without decomposing that relation into independent pairs.

---

## 4. Hypergraph Laplacian

The normalised hypergraph Laplacian is

$$\mathbf{L}*{\mathcal{H}}=\mathbf{I}-\mathbf{P}*{\mathcal{H}}.$$

Equivalently,

$$\mathbf{P}*{\mathcal{H}}=\mathbf{I}-\mathbf{L}*{\mathcal{H}}.$$

For an embedding matrix

$$\mathbf{Z}=\begin{bmatrix}\mathbf{z}_1^{\mathsf{T}}\\mathbf{z}*2^{\mathsf{T}}\\vdots\\mathbf{z}*{|\mathcal{V}|}^{\mathsf{T}}\end{bmatrix},$$

the hypergraph structural energy is

$$\mathcal{E}*{\mathcal{H}}(\mathbf{Z})=\mathrm{Tr}\bigl(\mathbf{Z}^{\mathsf{T}}\mathbf{L}*{\mathcal{H}}\mathbf{Z}\bigr).$$

A low structural energy encourages entities participating in the same hyperedge to acquire compatible representations.

Thus,

$$v_i,v_j\in e\Longrightarrow|\mathbf{z}_i-\mathbf{z}_j|_2\text{ is structurally constrained}.$$

The smoothness objective is

$$\min_{\mathbf{Z}}\mathcal{E}*{\mathcal{H}}(\mathbf{Z})=\min*{\mathbf{Z}}\mathrm{Tr}\bigl(\mathbf{Z}^{\mathsf{T}}\mathbf{L}_{\mathcal{H}}\mathbf{Z}\bigr).$$

---

## 5. Transformer Baseline

Let an interaction sequence be represented by

$$\mathbf{S}=\begin{bmatrix}\mathbf{s}_1^{\mathsf{T}}\\mathbf{s}_2^{\mathsf{T}}\\vdots\\mathbf{s}_m^{\mathsf{T}}\end{bmatrix}\in\mathbb{R}^{m\times d}.$$

The query matrix is

$$\mathbf{Q}=\mathbf{S}\mathbf{W}_Q.$$

The key matrix is

$$\mathbf{K}=\mathbf{S}\mathbf{W}_K.$$

The value matrix is

$$\mathbf{V}=\mathbf{S}\mathbf{W}_V.$$

Scaled dot-product attention is

$$\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\mathrm{softmax}\biggl(\frac{\mathbf{Q}\mathbf{K}^{\mathsf{T}}}{\sqrt{d_k}}\biggr)\mathbf{V}.$$

The attention coefficient between positions $i$ and $j$ is

$$\alpha_{ij}=\frac{\exp(\mathbf{q}_i^{\mathsf{T}}\mathbf{k}*j/\sqrt{d_k})}{\sum*{r=1}^{m}\exp(\mathbf{q}_i^{\mathsf{T}}\mathbf{k}_r/\sqrt{d_k})}.$$

Each primitive attention coefficient represents a pairwise relation:

$$(i,j).$$

Multiple attention layers can approximate complex dependencies, but the elementary interaction remains pairwise.

The hypergraph operator is

$$\mathbf{P}_{\mathcal{H}}=\mathbf{D}_v^{-1/2}\mathbf{H}\mathbf{W}\mathbf{D}_e^{-1}\mathbf{H}^{\mathsf{T}}\mathbf{D}_v^{-1/2}.$$

The attention operator is

$$\mathbf{A}_{\mathrm{attn}}=\mathrm{softmax}\biggl(\frac{\mathbf{Q}\mathbf{K}^{\mathsf{T}}}{\sqrt{d_k}}\biggr).$$

The first begins with explicit higher-order incidence structure. The second learns pairwise contextual weights from the input sequence.

---

## 6. Why Pairwise Reduction Can Lose Information

Consider a hyperedge containing $r$ entities:

$$e={v_1,v_2,\ldots,v_r}.$$

A pairwise expansion replaces this hyperedge with up to

$$\binom{r}{2}=\frac{r(r-1)}{2}$$

ordinary edges.

However, as relational objects,

$$e\neq\Pi_2(e),\qquad \Pi_2(e)=\bigcup_{1\leq i<j\leq r}(v_i,v_j).$$

The pairwise expansion records that individual pairs are connected, but it does not necessarily preserve the fact that all $r$ entities participated in one common event.

For example, consider the bundle

$$e_{\mathrm{bundle}}={\text{user},\text{article},\text{topic},\text{session},\text{time}}.$$

Its pairwise projection is

$$\Pi_2(e_{\mathrm{bundle}})={(\text{user},\text{article}),(\text{user},\text{topic}),(\text{article},\text{topic}),\ldots}.$$

In general,

$$e_{\mathrm{bundle}}\not\equiv\Pi_2(e_{\mathrm{bundle}}).$$

The HGNN preserves the factorisation

$$\mathcal{V}\xrightarrow{\mathbf{H}^{\mathsf{T}}}\mathcal{E}\xrightarrow{\mathbf{H}}\mathcal{V}.$$

This gives the model an explicit inductive bias for recovering multi-entity structure.

---

## 7. Prediction Function

Let the learned user representation be

$$\mathbf{z}_{u_i}\in\mathbb{R}^{d}.$$

Let the learned candidate representation be

$$\mathbf{z}_{a_j}\in\mathbb{R}^{d}.$$

A basic compatibility score is

$$s_{ij}=\mathbf{z}*{u_i}^{\mathsf{T}}\mathbf{z}*{a_j}.$$

The corresponding relevance probability is

$$\widehat{p}*{ij}=\frac{1}{1+\exp(-s*{ij})}.$$

For binary relevance labels,

$$y_{ij}\in{0,1},$$

the binary cross-entropy loss is

$$\mathcal{L}*{\mathrm{BCE}}=-\frac{1}{N}\sum*{(i,j)}\bigl[y_{ij}\log(\widehat{p}*{ij})+(1-y*{ij})\log(1-\widehat{p}_{ij})\bigr].$$

With quadratic regularisation, the total objective is

$$\mathcal{L}=\mathcal{L}_{\mathrm{BCE}}+\lambda|\theta|_2^2.$$

A pairwise ranking objective may alternatively be written as

$$\mathcal{L}*{\mathrm{rank}}=-\sum*{(i,j,k)}\log\biggl(\frac{1}{1+\exp(-(s_{ij}^{+}-s_{ik}^{-}))}\biggr).$$

---

## 8. Evaluation Metrics

### Hit Rate at 10

Let $\mathcal{R}_i$ be the relevant-item set for user $u_i$, and let $\widehat{\mathcal{R}}_i^{(10)}$ be the ten highest-ranked predictions.

The Hit Rate at 10 is

$$\mathrm{HR@10}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\bigl[\mathcal{R}_i\cap\widehat{\mathcal{R}}_i^{(10)}\neq\varnothing\bigr].$$

### Discounted Cumulative Gain

For user $u_i$,

$$\mathrm{DCG@10}*i=\sum*{r=1}^{10}\frac{2^{\mathrm{rel}_{i,r}}-1}{\log_2(r+1)}.$$

### Normalised Discounted Cumulative Gain

For user $u_i$,

$$\mathrm{NDCG@10}_i=\frac{\mathrm{DCG@10}_i}{\mathrm{IDCG@10}_i}.$$

The mean NDCG is

$$\mathrm{NDCG@10}=\frac{1}{N}\sum_{i=1}^{N}\mathrm{NDCG@10}_i.$$

### Precision

$$\mathrm{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}.$$

### Recall

$$\mathrm{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}.$$

### Accuracy

$$\mathrm{Accuracy}=\frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}.$$

---

## 9. Results

| Metric     |      HGNN |         Transformer |
| :--------- | --------: | ------------------: |
| HR@10      | **0.842** |               0.611 |
| NDCG@10    | **0.791** |               0.574 |
| Precision  | **0.900** |               0.650 |
| Recall     | **0.870** |               0.600 |
| Accuracy   | **0.900** |               0.650 |
| Final Loss | **0.214** | approximately 0.750 |

For a metric $M$ where larger values are preferable, define the relative improvement as

$$\Delta_M=\frac{M_{\mathrm{HGNN}}-M_{\mathrm{Transformer}}}{M_{\mathrm{Transformer}}}\times100%.$$

For HR@10,

$$\Delta_{\mathrm{HR@10}}=\frac{0.842-0.611}{0.611}\times100%\approx37.8%.$$

For NDCG@10,

$$\Delta_{\mathrm{NDCG@10}}=\frac{0.791-0.574}{0.574}\times100%\approx37.8%.$$

For precision,

$$\Delta_{\mathrm{Precision}}=\frac{0.900-0.650}{0.650}\times100%\approx38.5%.$$

For recall,

$$\Delta_{\mathrm{Recall}}=\frac{0.870-0.600}{0.600}\times100%=45.0%.$$

For accuracy,

$$\Delta_{\mathrm{Accuracy}}=\frac{0.900-0.650}{0.650}\times100%\approx38.5%.$$

Since lower loss is preferable, the relative loss reduction is

$$\Delta_{\mathrm{Loss}}=\frac{0.750-0.214}{0.750}\times100%\approx71.5%.$$

The primary empirical inequalities are

$$0.842>0.611$$

and

$$0.791>0.574.$$

Therefore,

$$\mathrm{HR@10}*{\mathrm{HGNN}}>\mathrm{HR@10}*{\mathrm{Transformer}}$$

and

$$\mathrm{NDCG@10}*{\mathrm{HGNN}}>\mathrm{NDCG@10}*{\mathrm{Transformer}}.$$

Full visual comparisons and experimental summaries are available in [`results/summary.md`](results/summary.md).

---

## 10. Experimental Interpretation

The benchmark supports the hypothesis that the dataset contains predictive relational structure of order greater than two:

$$\exists e\in\mathcal{E}\text{ such that }|e|>2.$$

The observed performance relation is

$$\text{higher-order incidence modelling}\longrightarrow\text{improved ranking quality}.$$

This result does not imply that an HGNN is universally superior to every Transformer architecture.

The benchmark-specific conclusion is

$$\text{HGNN performance}>\text{selected Transformer baseline performance}$$

under the implemented data construction, training procedure, and evaluation protocol.

The evidence indicates that the HGNN inductive bias is more closely aligned with the bundle-discovery structure of this task.

---

## 11. Popularity-Segment Robustness

Let the popularity of article $a_j$ be

$$\pi(a_j)=\sum_{i=1}^{N}\mathbf{1}[u_i\text{ interacted with }a_j].$$

The article set may be partitioned into

$$\mathcal{A}=\mathcal{A}*{\mathrm{head}}\cup\mathcal{A}*{\mathrm{middle}}\cup\mathcal{A}_{\mathrm{tail}}.$$

A popularity-only scoring mechanism behaves approximately as

$$s_{ij}\propto\pi(a_j).$$

A relational scoring mechanism instead has the form

$$s_{ij}=f_\theta(u_i,a_j,\mathcal{H}).$$

The popularity-segment analysis tests whether the model learns only global item frequency or preserves useful structural information for less frequent items.

### MAP by Item Popularity Segment

| Segment | HGNN MAP | Transformer MAP |
| :------ | -------: | --------------: |
| Head    | **0.91** |            0.68 |
| Mid     | **0.88** |            0.63 |
| Tail    | **0.84** |            0.59 |

The tail-segment relative gain is

$$\Delta_{\mathrm{Tail}}=\frac{0.84-0.59}{0.59}\times100%\approx42.4%.$$

The increasing advantage in lower-popularity segments suggests that explicit higher-order structure is particularly valuable where pairwise interaction evidence is sparse.

---

## 12. Architectural Comparison

### HGNN

The HGNN propagation rule is

$$\mathbf{X}^{(\ell+1)}=\sigma\bigl(\mathbf{D}_v^{-1/2}\mathbf{H}\mathbf{W}\mathbf{D}_e^{-1}\mathbf{H}^{\mathsf{T}}\mathbf{D}_v^{-1/2}\mathbf{X}^{(\ell)}\mathbf{\Theta}^{(\ell)}\bigr).$$

Its primary inductive bias is

$$\text{explicit multi-entity incidence}.$$

### Transformer

The Transformer attention rule is

$$\mathbf{Z}=\mathrm{softmax}\biggl(\frac{\mathbf{Q}\mathbf{K}^{\mathsf{T}}}{\sqrt{d_k}}\biggr)\mathbf{V}.$$

Its primary inductive bias is

$$\text{learned pairwise contextual dependence}.$$

The architectural comparison is therefore

$$\mathcal{H}=(\mathcal{V},\mathcal{E})$$

versus

$$\mathbf{Q}\mathbf{K}^{\mathsf{T}}.$$

Equivalently,

$$\text{explicit relational order}$$

versus

$$\text{implicit relational approximation}.$$

---

## 13. Structured-System Relevance

The mathematical architecture is not specific to recommendation data.

A higher-order event can be represented as

$$e_{\mathrm{event}}={\text{identity},\text{device},\text{process},\text{network origin},\text{resource},\text{time window}}.$$

Each entity may appear individually ordinary while the complete joint configuration carries the relevant signal.

This can be expressed as

$$\mathrm{Risk}(e_{\mathrm{event}})>\sum_{v\in e_{\mathrm{event}}}\mathrm{Risk}(v).$$

The higher-order relation therefore contains information that is not visible from isolated entity scores.

The same incidence-based architecture can support:

* coordinated-entity detection;
* shared-infrastructure analysis;
* campaign-level behaviour discovery;
* multi-stage event modelling;
* identity-device-resource correlation;
* distributed anomaly detection;
* structured state representations for autonomous agents.

The domain transfer is

$$\text{news bundle}\longrightarrow\text{multi-entity behavioural bundle}\longrightarrow\text{coordinated event}.$$

The general principle is:

$$\text{Do not reduce a higher-order event to isolated pairwise observations}$$

when the complete interaction carries the predictive information.

---

## 14. Reproducibility

A controlled benchmark requires both models to use the same training partition:

$$\mathcal{D}*{\mathrm{train}}^{\mathrm{HGNN}}=\mathcal{D}*{\mathrm{train}}^{\mathrm{Transformer}}.$$

The validation partitions must satisfy

$$\mathcal{D}*{\mathrm{validation}}^{\mathrm{HGNN}}=\mathcal{D}*{\mathrm{validation}}^{\mathrm{Transformer}}.$$

The test partitions must satisfy

$$\mathcal{D}*{\mathrm{test}}^{\mathrm{HGNN}}=\mathcal{D}*{\mathrm{test}}^{\mathrm{Transformer}}.$$

The evaluation functions must satisfy

$$\mathcal{M}*{\mathrm{eval}}^{\mathrm{HGNN}}=\mathcal{M}*{\mathrm{eval}}^{\mathrm{Transformer}}.$$

For a fixed random seed $s$,

$$s_{\mathrm{Python}}=s_{\mathrm{NumPy}}=s_{\mathrm{PyTorch}}=s.$$

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

$$\mathcal{D}={(u_i,\mathcal{I}_i,\mathcal{C}*i,y_i)}*{i=1}^{N},$$

where:

* $u_i$ is a user;
* $\mathcal{I}_i$ is the interaction history;
* $\mathcal{C}_i$ is the candidate-item set;
* $y_i$ is the relevance label.

The preprocessing pipeline constructs a hypergraph:

$$\mathcal{D}\longrightarrow\mathcal{H}=(\mathcal{V},\mathcal{E},\mathbf{X}).$$

The dataset is not redistributed in this repository. Download it from the official MIND website and follow its original licence and usage conditions.

---

## 18. Research Contribution

The project tests whether directly represented higher-order relations provide a stronger inductive bias than pairwise attention for bundle-sensitive ranking.

The primary hypothesis is

$$M(f_{\mathrm{HGNN}})>M(f_{\mathrm{Transformer}})$$

for

$$M\in{\mathrm{HR@10},\mathrm{NDCG@10}}.$$

The observed results satisfy both inequalities:

$$\mathrm{HR@10}(f_{\mathrm{HGNN}})>\mathrm{HR@10}(f_{\mathrm{Transformer}})$$

and

$$\mathrm{NDCG@10}(f_{\mathrm{HGNN}})>\mathrm{NDCG@10}(f_{\mathrm{Transformer}}).$$

The project therefore provides evidence for the principle

$$\text{representational structure matters independently of model generality}.$$

A structurally specialised model can outperform a more general sequence architecture when its mathematical representation more closely matches the data-generating process.

---

## 19. Limitations

The results do not prove that

$$f_{\mathrm{HGNN}}>f_{\mathrm{Transformer}}$$

for every dataset, architecture, or hyperparameter configuration.

Observed performance depends on

$$M=F(\mathcal{D},\mathcal{H},\theta,\Omega,\mathcal{S},\mathcal{B}),$$

where:

* $\mathcal{D}$ is the dataset;
* $\mathcal{H}$ is the hypergraph construction;
* $\theta$ is the model parameterisation;
* $\Omega$ is the optimisation procedure;
* $\mathcal{S}$ is the sampling strategy;
* $\mathcal{B}$ is the baseline configuration.

The correct conclusion is limited to the implemented experiment:

$$\text{HGNN}>\text{selected Transformer baseline}$$

for the reported bundle-discovery metrics under the controlled benchmark configuration.

---


## Core Result

$$\mathrm{HR@10}*{\mathrm{HGNN}}=0.842>0.611=\mathrm{HR@10}*{\mathrm{Transformer}}.$$

$$\mathrm{NDCG@10}*{\mathrm{HGNN}}=0.791>0.574=\mathrm{NDCG@10}*{\mathrm{Transformer}}.$$

$$\mathrm{MAP}*{\mathrm{Tail,HGNN}}=0.84>0.59=\mathrm{MAP}*{\mathrm{Tail,Transformer}}.$$

Therefore, for this benchmark,

$$\text{explicit higher-order structure}>\text{pairwise-attention baseline}.$$





