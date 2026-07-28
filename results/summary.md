# Final Evaluation Summary — HGNN vs Transformer on MIND-Small

## Metrics Overview

* **HR@10**: HGNN 0.842 vs Transformer 0.611

* **NDCG@10**: HGNN 0.791 vs Transformer 0.574

* **Final Validation Loss**: HGNN 0.214 vs Transformer approximately 0.75

* **Precision**: HGNN approximately 0.90 vs Transformer approximately 0.65

* **Recall**: HGNN approximately 0.87 vs Transformer approximately 0.60

* **Accuracy**: HGNN approximately 0.90 vs Transformer approximately 0.65

* **Tail-Segment MAP**: HGNN 0.84 vs Transformer 0.59

* **Relative Tail-Segment MAP Gain**: approximately 42.4%

## Visualizations

### 🔹 Training Loss Comparison

![Training Loss Comparison](output.hgnn)

*The HGNN exhibits substantially faster and more stable convergence, reaching a final validation loss of 0.214. The Transformer baseline remains near approximately 0.75 under the same benchmark configuration.*

### 🔹 Precision, Recall, and Accuracy

![Precision, Recall, and Accuracy Comparison](precision_recall_accuracy_comparison\(hgnn\))

*The HGNN outperforms the Transformer across precision, recall, and accuracy. The largest core-metric improvement appears in recall, which increases from approximately 0.60 to 0.87.*

### 🔹 Full Comparison: Loss, Ranking Metrics, and MAP

![Transformer vs HGNN](transformer vs hgnn)

*The complete comparison shows that the HGNN performs better in optimisation, top-ranked recommendation quality, predictive classification, and item-popularity robustness.*

## Key Findings

* The HGNN improves HR@10 from 0.611 to 0.842, corresponding to a relative gain of approximately 37.8%.

* NDCG@10 improves from 0.574 to 0.791, indicating substantially stronger ranking quality near the top of the recommendation list.

* Precision and accuracy both improve from approximately 0.65 to 0.90.

* Recall improves from approximately 0.60 to 0.87, representing a relative gain of 45.0%.

* Final validation loss decreases from approximately 0.75 to 0.214, corresponding to a reduction of approximately 71.5%.

* The HGNN maintains its advantage across head, mid, and tail popularity segments.

* Tail-segment MAP increases from 0.59 to 0.84, showing that the HGNN remains effective when interactions are sparse and items are underrepresented.

* The increasing relative advantage from head to tail suggests that explicit higher-order structure is particularly valuable where pairwise interaction evidence is limited.

## Interpretation

The results support the hypothesis that the recommendation data contains predictive structure involving more than two entities at a time.

The HGNN directly represents these relations through hypergraph incidence operators, allowing information to propagate through complete user-item-context bundles.

The Transformer baseline instead learns contextual dependencies through pairwise attention weights.

Under the implemented benchmark, the explicit higher-order inductive bias of the HGNN is more closely aligned with the bundle-discovery problem than the pairwise-attention structure of the selected Transformer.

The correct benchmark-specific conclusion is:

**HGNN outperforms the selected Transformer baseline across ranking quality, predictive performance, optimisation loss, and popularity-segment robustness.**

## Relevance Beyond Recommendation

The same higher-order representation can be applied to structured domains in which meaningful events involve multiple interacting entities.

Examples include:

* coordinated-account detection;
* shared-infrastructure discovery;
* fraud-ring analysis;
* multi-stage cybersecurity events;
* identity-device-resource relationships;
* structured state representations for autonomous agents.

The benchmark therefore demonstrates a broader principle:

**When the target phenomenon is inherently multi-entity, preserving higher-order structure can outperform reducing the same information to pairwise interactions.**

## Environment

* Python 3.10.11

* PyTorch 2.1.0

* CUDA 12.4, optional

* Windows-native setup

* No Docker requirement

* No WSL requirement

* No Linux dependency


