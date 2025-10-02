\# 📊 Metrics Summary — HGNN vs Transformer (MIND-Small)



\## 🔹 Core Evaluation Metrics



| Metric       | HGNN     | Transformer |

|--------------|----------|-------------|

| HR@10        | 0.842    | 0.611       |

| NDCG@10      | 0.791    | 0.574       |

| Precision    | 0.90     | 0.65        |

| Recall       | 0.87     | 0.60        |

| Accuracy     | 0.90     | 0.65        |

| Final Loss   | 0.214    | ~0.75       |



---



\## 🔹 MAP by Item Popularity Segment



| Segment | HGNN MAP | Transformer MAP |

|---------|----------|------------------|

| Head    | 0.91     | 0.68             |

| Mid     | 0.88     | 0.63             |

| Tail    | 0.84     | 0.59             |



---



\##  Observations



\- HGNN consistently outperforms Transformer across all metrics.

\- Tail-segment MAP boost highlights HGNN’s robustness in sparse recommendation zones.

\- Manual attention engineering and minimal architecture contributed to crash-proof training and faster convergence.



\##  Environment



\- Python 3.10.11  

\- PyTorch 2.1.0  

\- CUDA 12.4 (optional)  

\- Windows-native setup (no Docker, WSL, or Linux)

