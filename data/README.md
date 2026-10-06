# Data

Datasets are **not** redistributed in this repository.

Expected layout for `ACM` and `DBLP` (used by `baselines/run_baselines.py`):

```
data/
├── acm/
│   ├── features.npy          # (N, D) node features
│   ├── labels.npy            # (N,) integer labels  (or one-hot (N, C))
│   ├── graph_coauthor.npz    # scipy sparse adjacency, one file per view: graph_<name>.npz
│   └── graph_subject.npz
└── dblp/
    ├── features.npy
    ├── labels.npy
    └── graph_coauthor.npz    # a cosine k-NN graph (k=10) is built automatically from features
```

Rules implemented by the loader:
* every `graph_<name>.npz` becomes a view; adjacency is symmetrised, binarised, self-loops removed;
* if fewer than two graphs are found, a cosine k-NN graph (`--knn_k`, default 10) is added;
* a `union` view (element-wise max of all views) is added and is the **default evaluation graph**.

| Dataset | Nodes | Feature dim | Clusters | Views used by the baselines |
|---|---|---|---|---|
| ACM  | 3,025 | 1,902 | 3 | co-author, subject, union |
| DBLP | 4,057 | 334   | 4 | co-author, k-NN (k=10), union |

> **TODO (authors):** add the original source / download link and preprocessing
> script of each dataset here.
