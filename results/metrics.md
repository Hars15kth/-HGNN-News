# 📊 Metrics Summary — HGNN vs Transformer (MIND-Small)

## 🔹 Core Evaluation Metrics

| Metric     | HGNN  | Transformer |
| ---------- | ----- | ----------- |
| HR@10      | 0.842 | 0.611       |
| NDCG@10    | 0.791 | 0.574       |
| Precision  | 0.90  | 0.65        |
| Recall     | 0.87  | 0.60        |
| Accuracy   | 0.90  | 0.65        |
| Final Loss | 0.214 | ~0.75       |

## 🔹 Relative Performance Improvement

| Metric     | HGNN Improvement over Transformer |
| ---------- | --------------------------------- |
| HR@10      | 37.8%                             |
| NDCG@10    | 37.8%                             |
| Precision  | 38.5%                             |
| Recall     | 45.0%                             |
| Accuracy   | 38.5%                             |
| Final Loss | 71.5% lower                       |

## 🔹 MAP by Item Popularity Segment

| Segment | HGNN MAP | Transformer MAP |
| ------- | -------- | --------------- |
| Head    | 0.91     | 0.68            |
| Mid     | 0.88     | 0.63            |
| Tail    | 0.84     | 0.59            |

## 🔹 MAP Improvement by Popularity Segment

| Segment | Absolute Gain | Relative Gain |
| ------- | ------------- | ------------- |
| Head    | +0.23         | 33.8%         |
| Mid     | +0.25         | 39.7%         |
| Tail    | +0.25         | 42.4%         |

## Observations

* HGNN consistently outperforms the Transformer baseline across ranking, classification, optimisation, and popularity-segment metrics.

* HR@10 increases from 0.611 to 0.842, corresponding to a relative improvement of approximately 37.8%.

* NDCG@10 increases from 0.574 to 0.791, showing that the HGNN produces substantially stronger ranking quality near the top of the recommendation list.

* Recall improves from 0.60 to 0.87, representing the largest gain among the core predictive metrics.

* Final training loss decreases from approximately 0.75 to 0.214, corresponding to a reduction of approximately 71.5%.

* The HGNN advantage increases as item popularity decreases, with the largest relative MAP improvement appearing in the tail segment.

* Tail-segment MAP rises from 0.59 to 0.84, demonstrating stronger recovery of sparse and underrepresented recommendation structure.

* These results support the use of explicit hypergraph incidence modelling when interactions depend on multi-entity relations that cannot be represented adequately through pairwise attention alone.

* The compact HGNN architecture also produced more stable optimisation and faster convergence than the Transformer baseline under the implemented benchmark configuration.

## Environment

* Python 3.10.11
* PyTorch 2.1.0
* CUDA 12.4 (optional)
* Windows-native setup
* No Docker requirement
* No WSL requirement
* No Linux dependency


