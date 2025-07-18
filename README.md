# **CaRTed — Causal Representation Learning with Irregular Tensors**

This repository provides the official implementation for our paper
**“Toward Temporal Causal Representation Learning with Tensor Decomposition.”**
The code base—**CaRTed**—unifies PARAFAC2 tensor decomposition with causal‐structure learning to recover both contemporaneous and temporal graphs from irregular, multi-slice data.

---

## Directory overview

| Path              | Purpose                                                                            |
| ----------------- | ---------------------------------------------------------------------------------- |
| `CDPAR_demo.py`   | End-to-end demo script that ties all blocks together.                              |
| `Causal_block.py` | Functions for updating the causal (graph-learning) block.                          |
| `Tensor_block.py` | Functions for updating the PARAFAC2 tensor block.                                  |
| `Mat_sim.py`      | Utilities for simulating ground-truth matrices $W$ and lag matrices $\{A^{(p)}\}$. |
| `Data Generation.ipynb`     | Example generator for synthetic patient–visit–diagnosis tensors (with time lags).  |
| `Structure Simulation.ipynb`| Example generator for causal graph.|
| `Metric.py`       | Evaluation metrics: SHD, TPR, FDR, factor-similarity, etc.                         |
| `utils.py`        | Miscellaneous helpers (normalisation, logging, plotting).                          |
| `data_eg/`        | Sample synthetic tensor used in the demo.                                          |
| `eg_matrix/`      | Sample $W$ and $\{A^{(p)}\}$ for the demo.                                         |

---
## Demos 

The **CaRTed** demos directory includes example implementations of our methods, demonstrating how to use the packages and apply the methodology.  

---

## Citation

```bibtex
@misc{carted2024,
  title   = {Toward Temporal Causal Representation Learning with Tensor Decomposition},
  author  = {Chen, Jianhong and Ma, Ying and Yue, Xubo},
  year    = {2024},
  eprint  = {arXiv:2412.09814},
  archivePrefix = {arXiv}
}
```

---

Feel free to open an issue or pull request if you encounter problems or have suggestions.
