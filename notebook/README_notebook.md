\# HNN Benchmarking Notebook Overview



This notebook benchmarks a Hybrid Neural Network (HNN) on the \*\*MindSmall\*\* dataset over 12 epochs. It includes model definition, training loop, evaluation metrics, and visualizations.



\## 📚 Notebook Sections



1\. \*\*Setup and Imports\*\*  

&nbsp;  Loads required libraries and sets device configuration.



2\. \*\*Data Loading\*\*  

&nbsp;  Loads and preprocesses the MindSmall dataset.



3\. \*\*Model Architecture\*\*  

&nbsp;  Defines the HNN using PyTorch with ReLU activation and two linear layers.



4\. \*\*Training Loop\*\*  

&nbsp;  Trains the model over 12 epochs, tracking loss and performance.



5\. \*\*Evaluation\*\*  

&nbsp;  Computes HR@10 and NDCG@10 to assess top-k recommendation quality.



6\. \*\*Visualizations\*\*  

&nbsp;  Plots training/validation loss curves and performance metrics.



\## 📊 Key Results



\- HR@10: \*\*0.842\*\*

\- NDCG@10: \*\*0.791\*\*

\- Final Validation Loss: \*\*0.214\*\*



\## 🧠 Observations



\- The model converged steadily over 12 epochs.

\- Manual attention engineering improved stability.

\- Evaluation metrics indicate strong top-k performance.



\## ⚙️ Environment



\- Python 3.10.11

\- PyTorch 2.1.0

\- CUDA 12.4 (optional)

\- Windows-native setup (no Docker, WSL, or Linux)



\## 📎 Source



This notebook is derived from `HNN-12EPCHSs.pdf` and preserved in `notebook/` for reference.

