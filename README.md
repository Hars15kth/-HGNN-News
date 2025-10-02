\# HGNN vs Transformer — Bundle Discovery Benchmarking on MIND-Small



\## 🔍 Project Overview



This project benchmarks a Hybrid Neural Network (HGNN) against a Transformer model for bundle discovery on the \[MIND-Small](https://msnews.github.io/) dataset. It evaluates training efficiency, predictive performance, and robustness across item popularity segments.



Built for reproducibility, minimal dependencies, and recruiter clarity.



---



\## 📊 Key Results



| Metric       | HGNN     | Transformer |

|--------------|----------|-------------|

| HR@10        | 0.842    | 0.611       |

| NDCG@10      | 0.791    | 0.574       |

| Precision    | 0.90     | 0.65        |

| Recall       | 0.87     | 0.60        |

| Accuracy     | 0.90     | 0.65        |

| Final Loss   | 0.214    | ~0.75       |



📈 Visual comparisons and full metrics available in \[`results/summary.md`](results/summary.md)



---



\##  Architecture



\- \*\*HGNN\*\*: Two-layer feedforward network with ReLU activation

\- \*\*Transformer\*\*: Standard encoder-based architecture

\- Manual attention engineering for crash-proof training



\##  Environment



\- Python 3.10.11  

\- PyTorch 2.1.0  

\- CUDA 12.4 (optional)  

\- Windows-native setup (no Docker, WSL, or Linux)



---



\##  Run Instructions



\### 1. Clone the repository

```powershell

git clone https://github.com/Hars15kth/-HGNN-News.git

cd -HGNN-News



