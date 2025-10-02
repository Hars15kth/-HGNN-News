\# Final Evaluation Summary — HNN vs Transformer on MIND-Small



\## Metrics Overview



\- \*\*HR@10\*\*: 0.842  

\- \*\*NDCG@10\*\*: 0.791  

\- \*\*Final Validation Loss\*\*: 0.214  

\- \*\*Precision\*\*: HGNN ~0.90 vs Transformer ~0.65  

\- \*\*Recall\*\*: HGNN ~0.87 vs Transformer ~0.60  

\- \*\*Accuracy\*\*: HGNN ~0.90 vs Transformer ~0.65  

\- \*\*MAP (Tail Segment)\*\*: HGNN significantly outperforms Transformer



---



\##  Visualizations



\### 🔹 Training Loss Comparison

!\[Training Loss Comparison](output.hgnn)

\*HGNN shows steep convergence over 10 epochs, while Transformer remains flat. Indicates superior optimization.\*



---



\### 🔹 Precision / Recall / Accuracy

!\[Precision / Recall / Accuracy Comparison](precision\_recall\_accuracy\_comparison(hgnn))

\*HGNN outperforms Transformer across all three metrics—precision, recall, and accuracy.\*



---



\### 🔹 Full Comparison: Loss, Metrics, MAP

!\[Transformer vs HGNN](transformer vs hgnn)

\*HGNN dominates in training efficiency, prediction quality, and performance across item popularity segments.\*



---



\##  Observations



\- HGNN converges faster and generalizes better than Transformer.

\- Manual attention engineering contributed to crash-proof training.

\- Tail-segment MAP boost suggests robustness in sparse recommendation scenarios.



\##  Environment



\- Python 3.10.11  

\- PyTorch 2.1.0  

\- CUDA 12.4 (optional)  

\- Windows-native setup (no Docker, WSL, or Linux)

