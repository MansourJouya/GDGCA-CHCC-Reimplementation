> **⚠️ Archived.** This repository has been merged into
> [MansourJouya/AGAMC](https://github.com/MansourJouya/AGAMC) (see the `baselines/` folder),
> which is the maintained version. Please use that repository.
> 
<h1 align="center">GDGCA & CHCC — Re-implementations</h1>
<h3 align="center">Reproducible re-implementations of two graph-clustering baselines<br>(GDGCA and CHCC) for ACM and DBLP</h3>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Not official" src="https://img.shields.io/badge/code-unofficial%20re--implementation-orange">
</p>

This repository contains simplified, from-scratch **re-implementations** of two deep graph-clustering
methods used as baselines in our multi-view clustering paper
*"Adaptive Graph Augmentation for Multi-View Clustering: A Contrastive Learning Approach with
View-Specific Augmentations" (AGAMC)*:

* **GDGCA** – Gated Deep Graph Clustering with Contrastive Alignment (Wang et al., *Applied Sciences*, 2026)
* **CHCC** – Cross-scale Hypergraph Contrastive Clustering (Liu et al., *J. King Saud Univ. – CIS*, 2026)

They were created because no comparable results of these methods were available on the **ACM** and
**DBLP** benchmarks. Everything needed to reproduce the reported numbers is included.

> ⚠️ **These are NOT the official implementations.** They re-implement the published *ideas* in a
> simplified form. Numbers may differ from those of the original authors' code.
> Authors: Mansour Jouya, Alireza Abdollahpouri — Department of Computer Engineering, University of Kurdistan.

## Contents
[Methods](#methods) · [Datasets](#datasets) · [Protocol](#evaluation-protocol) · [Results](#results) ·
[Environment](#experimental-environment) · [Reproduction](#installation--reproduction) ·
[Structure](#repository-structure) · [Limitations](#limitations--honest-notes) · [Citation](#citation)

## Methods

### GDGCA (3-stage pipeline)
1. **AE pre-training** – autoencoder (D → 256 → 128) trained with MSE reconstruction (300 steps).
2. **Dual-stream pre-training** – an AE stream and a GCN stream are fused layer-wise by a learnable gate
   `h = λ·h_AE + (1−λ)·h_GCN`. Trained with feature reconstruction **plus GAE-style adjacency reconstruction**
   (200 steps). K-means on the **fused** embedding initialises the cluster centres.
3. **Joint clustering** – DEC-style KL loss on Student-t soft assignments + feature reconstruction +
   AE/GCN contrastive alignment (InfoNCE, batch 256). Training stops when fewer than 0.1 % of
   assignments change (label-free).

*Why the pre-training stage matters.* A first attempt that initialised centres from the AE branch only
collapsed on DBLP (ACC ≈ 0.31). Initialising on the fused embedding after adjacency-aware
pre-training fixed it (see `results/logs/`).

### CHCC (simplified)
* micro level: node embeddings from a 2-layer GCN; meso level: hyperedges (node + 1-hop neighbours);
  macro level: cluster prototypes.
* node-to-hyperedge contrastive loss where neighbouring hyperedges are positives (avoids false negatives),
* clustering-guided prototype alignment (KL) after a contrastive warm-up, plus an entropy balance term.

### Hyper-parameters
| | GDGCA | CHCC |
|---|---|---|
| Encoder | 256 → 128 | 256 → 128 |
| Epochs | 300 (AE) + 200 (dual-stream) + ≤ 200 (clustering) | 200 (warm-up 50) |
| Learning rate | 1e-3 (AE) / 5e-4 | 1e-3 (wd 5e-4) |
| Temperature | τ = 0.5 | τ = 0.5, τ_c = 0.5 |
| Loss weights | ξ₁ = 1.0 (rec), ξ₂ = 0.1 (align) | λ_c = 0.5, λ_b = 1.0 |
| Augmentation | – | edge drop 0.2, feature mask 0.3 |
| KMeans | n_init = 20 | n_init = 20 |

## Datasets

| Dataset | Nodes | Feature dim | Clusters | Graphs |
|---|---|---|---|---|
| **ACM**  | 3,025 | 1,902 | 3 | co-author (13,374 edges), subject (1,107,032 edges), union |
| **DBLP** | 4,057 | 334   | 4 | co-author (3,528 edges), cosine k-NN k=10 (31,137 edges), union |

ACM: papers from three research areas (Database, Data Mining, Multimedia).
DBLP: authors from four research areas, with term-usage features.
Format and preparation: [`data/README.md`](data/README.md). Datasets are not redistributed.

## Evaluation protocol

* Metrics: **ACC** (Hungarian matching), **NMI** (geometric), **ARI**, **macro-F1**.
* **5 runs**, seeds `0, 22, 62, 77, 99`; mean ± std reported.
* Graph used: the **`union`** of all available graphs (element-wise max) — a *label-free* protocol.
* Ground-truth labels are used only for evaluation and for `[sanity]` / `[diag]` log lines; they never
  influence training, initialisation or stopping.
* `--view_select best` picks the best single graph with ground-truth ACC (oracle). It is provided only as
  an upper bound and should not be presented as a fair result.

## Results

| Dataset | Method | ACC | NMI | ARI | F1 |
|---|---|---|---|---|---|
| ACM  | GDGCA | 0.656 ± 0.016 | 0.422 ± 0.025 | 0.361 ± 0.034 | 0.668 ± 0.013 |
| ACM  | CHCC  | 0.655 ± 0.010 | 0.418 ± 0.014 | 0.359 ± 0.021 | 0.666 ± 0.008 |
| DBLP | GDGCA | 0.800 ± 0.001 | 0.488 ± 0.002 | 0.546 ± 0.002 | 0.795 ± 0.002 |
| DBLP | CHCC  | 0.795 ± 0.003 | 0.483 ± 0.003 | 0.539 ± 0.006 | 0.790 ± 0.003 |

Raw console output: [`results/logs/`](results/logs).

**Reference points (same data, K-means / smoothing, seed 0)**

| Dataset | K-means on raw X | K-means on Â²X |
|---|---|---|
| ACM  | 0.693 | 0.638 |
| DBLP | 0.565 | 0.685 |

**Notes on interpretation**
* On ACM the `union` graph is dominated by the dense *subject* graph (density ≈ 0.24), a poor input for
  single-graph methods. Both methods therefore do not exceed K-means on raw features here; the numbers are
  conservative for these baselines. Run `--view_select best` for the oracle upper bound.
* On DBLP both methods are within 0.5 ACC points of each other.

## Experimental environment

| Component | Specification |
|---|---|
| Machine | Dell Precision 5550 (mobile workstation, x64) |
| CPU | Intel Core i7-10850H @ 2.70 GHz, 6 cores / 12 threads |
| GPU | NVIDIA Quadro T2000 with Max-Q Design, 4 GB GDDR6 |
| RAM | 32 GB (31.8 GB usable) |
| OS | Windows 11 Pro, 10.0.26200 (Build 26200) |
| BIOS | Dell Inc. 1.41.0 (03/26/2026), UEFI |
| Python / PyTorch | 3.12 / 2.5.1 + CUDA 12.1 |

K-means (scikit-learn) always runs on the CPU. The dense N×N adjacency fits in 4 GB for N ≤ 4,057.

## Installation & reproduction

```bash
git clone https://github.com/<your-username>/GDGCA-CHCC-Reimplementation.git
cd GDGCA-CHCC-Reimplementation
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect True
```

```bash
# Results reported above (label-free protocol, 5 seeds)
python baselines/run_baselines.py --dataset dblp --methods gdgca chcc --view_select union
python baselines/run_baselines.py --dataset acm  --methods gdgca chcc --view_select union

# Oracle upper bound (labels used to choose the graph)
python baselines/run_baselines.py --dataset acm  --methods gdgca chcc --view_select best
```

Options: `--seeds`, `--knn_k`, `--warm`, `--warm2`, `--epochs`, `--lr`, `--tau`, `--xi1`, `--xi2`,
`--graph_rec 0|1`, `--gdgca_std 0|1`, `--chcc_*`.

## Repository structure

```
GDGCA-CHCC-Reimplementation/
├── baselines/run_baselines.py   # GDGCA & CHCC + data loader + metrics
├── data/                        # datasets (not tracked) + format description
├── results/logs/                # raw console logs of the reported runs
├── requirements.txt
├── LICENSE
└── README.md
```

## Limitations & honest notes

* Unofficial, simplified re-implementations; no tuning against ground-truth labels was used to pick the
  reported configuration, but the design of GDGCA stage 2 was revised after observing a failure on DBLP
  (diagnosed with label-based K-means checks that are printed only for logging).
* Only ACM and DBLP are covered; no results for other datasets are provided here.
* Ground-truth-free early stopping is used for GDGCA; CHCC runs a fixed number of epochs.
* Graphs up to ~4 k nodes only (dense adjacency); not intended for large graphs.

## Citation

Please cite the original methods, and our paper if you use these baselines:

```bibtex
@article{wang2026adaptive,
  title   = {Adaptive Deep Graph Clustering via Layer-Wise Gated Fusion and Cross-View Contrastive Alignment},
  author  = {Wang, C. and others},
  journal = {Applied Sciences}, volume = {16}, number = {4}, pages = {2131}, year = {2026}
}
@article{liu2026cross,
  title   = {Cross-scale hypergraph contrastive clustering},
  author  = {Liu, Y. and An, J. and Ding, Y.},
  journal = {Journal of King Saud University -- Computer and Information Sciences}, volume = {38}, year = {2026}
}
@article{jouya2026agamc,
  title   = {Adaptive Graph Augmentation for Multi-View Clustering: A Contrastive Learning Approach with View-Specific Augmentations},
  author  = {Jouya, Mansour and Abdollahpouri, Alireza},
  journal = {Under review}, year = {2026}
}
```

## License
MIT — see [LICENSE](LICENSE).
