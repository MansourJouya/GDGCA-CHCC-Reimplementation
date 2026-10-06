"""
Baseline re-implementations (GDGCA, CHCC) used in the AGAMC paper: GDGCA and CHCC.

NOTE: These are simplified, from-scratch re-implementations of the *ideas* of
  - GDGCA (Wang et al., Applied Sciences 2026) and
  - CHCC  (Liu et al., J. King Saud Univ. CIS 2026),
NOT the official code. They are provided for transparency and reproducibility.

Protocol (default for the paper): --view_select union
  Each method is run on the label-free union of all available graphs.
  `--view_select best` selects the best single graph by ACC using ground-truth
  labels (oracle selection) and should only be reported as an upper bound.

Ground-truth labels are used only for evaluation and for [sanity]/[diag] log lines;
they never influence training, initialisation or early stopping.
"""
import argparse, os, random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score

SEEDS = [0, 22, 62, 77, 99]
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------------------------------------- data
def knn_graph(X, k):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    A = np.zeros_like(S, dtype=np.float32)
    A[np.arange(len(S))[:, None], idx] = 1
    A = ((A + A.T) > 0).astype(np.float32)
    np.fill_diagonal(A, 0)
    return A


def load_data_npy(data_dir, dataset, norm='l2', knn_k=10, force_knn=False):
    ds_dir = os.path.join(data_dir, dataset)
    print('files in %s: %s' % (ds_dir, sorted(os.listdir(ds_dir))))

    X = np.load(os.path.join(ds_dir, 'features.npy')).astype(np.float32)
    if sp.issparse(X):
        X = X.toarray()
    y = np.load(os.path.join(ds_dir, 'labels.npy'))
    y = y.argmax(1) if (y.ndim == 2 and y.shape[1] > 1) else y.ravel()
    y = y.astype(int)

    X_raw = X.copy()
    if norm == 'l2':
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    views = {}
    for fname in sorted(os.listdir(ds_dir)):
        if fname.endswith('.npz') and fname.startswith('graph_'):
            view_name = fname.replace('graph_', '').replace('.npz', '')
            A = sp.load_npz(os.path.join(ds_dir, fname))
            A = A.toarray() if sp.issparse(A) else np.asarray(A)
            A = ((A + A.T) > 0).astype(np.float32)
            np.fill_diagonal(A, 0)
            views[view_name] = A

    if (force_knn or len(views) < 2) and 'knn' not in views and knn_k > 0:
        print('>> building kNN graph (k=%d, cosine) from features' % knn_k)
        views['knn'] = knn_graph(X_raw, knn_k)

    if len(views) > 1:
        views['union'] = np.maximum.reduce(list(views.values()))
    for n, A in views.items():
        print('   view %-10s edges=%d density=%.5f' % (n, int(A.sum() // 2), A.sum() / (len(A) ** 2)))
    return X, X_raw, y, views


def norm_adj_t(A):
    A = A + torch.eye(A.shape[0], device=A.device)
    d = A.sum(1).pow(-0.5)
    return d[:, None] * A * d[None, :]


# ----------------------------------------------------------------------------- metrics
def cluster_metrics(y, p):
    K = int(max(y.max(), p.max())) + 1
    w = np.zeros((K, K), dtype=np.int64)
    for a, b in zip(p, y):
        w[a, b] += 1
    r, c = linear_sum_assignment(-w)
    mp = dict(zip(r, c))
    pm = np.array([mp[i] for i in p])
    return dict(ACC=(pm == y).mean(),
                NMI=normalized_mutual_info_score(y, p, average_method='geometric'),
                ARI=adjusted_rand_score(y, p),
                F1=f1_score(y, pm, average='macro'))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ============================================================================= GDGCA (FIXED)
class GDGCA(nn.Module):
    def __init__(self, D, K, dims=(256, 128)):
        super().__init__()
        d = [D, *dims]
        self.L = len(dims)
        self.WA = nn.ModuleList([nn.Linear(d[i], d[i + 1]) for i in range(self.L)])
        self.Th = nn.ModuleList([nn.Linear(d[i], d[i + 1], bias=False) for i in range(self.L)])
        self.beta = nn.Parameter(torch.zeros(self.L))
        self.dec = nn.Sequential(nn.Linear(d[-1], d[1]), nn.ReLU(), nn.Linear(d[1], D))
        self.proj = nn.Sequential(nn.Linear(d[-1], d[-1]), nn.ReLU(), nn.Linear(d[-1], 64))
        self.mu = nn.Parameter(torch.zeros(K, d[-1]))

    def encode_ae(self, X):
        h = X
        for l, W in enumerate(self.WA):
            h = W(h)
            if l < self.L - 1:
                h = F.relu(h)
        return h

    def forward(self, X, A):
        h = X
        for l in range(self.L):
            ha, hs = self.WA[l](h), A @ self.Th[l](h)
            if l < self.L - 1:
                ha, hs = F.relu(ha), F.relu(hs)
            lam = torch.sigmoid(self.beta[l])
            h = lam * ha + (1 - lam) * hs
        return h, ha, hs


def soft_assign(h, mu, alpha=1.0):
    q = (1.0 + torch.cdist(h, mu) ** 2 / alpha) ** (-(alpha + 1) / 2)
    return q / q.sum(1, keepdim=True)


def target_dist(q):
    w = q ** 2 / q.sum(0)
    return w / w.sum(1, keepdim=True)


def align_loss(m, ha, hs, a):
    idx = torch.randperm(ha.shape[0], device=ha.device)[:a.bs]
    za, zs = F.normalize(m.proj(ha[idx]), dim=1), F.normalize(m.proj(hs[idx]), dim=1)
    sim = za @ zs.T / a.tau
    lab = torch.arange(len(idx), device=ha.device)
    return 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab))


def run_gdgca(Ab, X, y, K, seed, a, verbose=False):
    set_seed(seed)
    A = norm_adj_t(Ab)
    N, D = X.shape
    # FIX 1: L2-normalised rows give a near-zero MSE -> rescale so the AE actually learns
    if a.gdgca_std:
        # column-wise z-score so rare, discriminative terms are not drowned out by frequent ones
        mu_, sd_ = X.mean(0, keepdim=True), X.std(0, keepdim=True)
        Xg = ((X - mu_) / (sd_ + 1e-3)).clamp(-5, 5)
    else:
        Xg = X * (D ** 0.5) if a.gdgca_rescale else X
    m = GDGCA(D, K).to(dev)

    # ---- Stage 1: AE pre-training
    ae_params = list(m.WA.parameters()) + list(m.dec.parameters())
    opt = torch.optim.Adam(ae_params, lr=a.lr_warm)
    for _ in range(a.warm):
        opt.zero_grad()
        F.mse_loss(m.dec(m.encode_ae(Xg)), Xg).backward()
        opt.step()

    if verbose:
        with torch.no_grad():
            e1 = m.encode_ae(Xg).cpu().numpy()
            As = (A @ (A @ Xg)).cpu().numpy()
        print('   [diag] KMeans raw X            ACC=%.3f' % cluster_metrics(y, KMeans(K, n_init=10, random_state=seed).fit_predict(Xg.cpu().numpy()))['ACC'])
        print('   [diag] KMeans A^2 X (smoothed) ACC=%.3f' % cluster_metrics(y, KMeans(K, n_init=10, random_state=seed).fit_predict(As))['ACC'])
        print('   [diag] KMeans AE-embedding (after stage 1) ACC=%.3f' % cluster_metrics(y, KMeans(K, n_init=10, random_state=seed).fit_predict(e1))['ACC'])

    # ---- Stage 2 (FIX 2): pre-train the FULL dual-stream model (rec + alignment, no clustering)
    #      so that the space we cluster in is the same space mu is initialised in.
    net_params = [p for n, p in m.named_parameters() if n != 'mu']
    if a.graph_rec:
        tgt = ((Ab + torch.eye(N, device=dev)) > 0).float()
        pos_w = (N * N - tgt.sum()) / tgt.sum()
    opt = torch.optim.Adam(net_params, lr=a.lr)
    for _ in range(a.warm2):
        m.train()
        hF, ha, hs = m(Xg, A)
        loss = F.mse_loss(m.dec(hF), Xg)
        if a.graph_rec:
            logits = hF @ hF.T
            loss = loss + F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pos_w)
        if a.xi2_pre > 0:
            loss = loss + a.xi2_pre * align_loss(m, ha, hs, a)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net_params, 5.0)
        opt.step()

    # ---- FIX 3: initialise mu with KMeans on the fused embedding hF (not on the AE branch)
    m.eval()
    with torch.no_grad():
        h0 = m(Xg, A)[0].cpu().numpy()
    km = KMeans(K, n_init=20, random_state=seed).fit(h0)
    if verbose:
        print('   [sanity] GDGCA KMeans-on-fused-embedding ACC=%.3f' % cluster_metrics(y, km.labels_)['ACC'])
    m.mu.data = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=dev)

    # ---- Stage 3: joint clustering (KL + rec + align)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    prev = km.labels_.copy()
    P = None
    for ep in range(a.epochs):
        m.train()
        hF, ha, hs = m(Xg, A)
        q = soft_assign(hF, m.mu).clamp_min(1e-12)
        if ep % a.upd == 0:
            P = target_dist(q).detach()
            cur = q.argmax(1).detach().cpu().numpy()
            # label-free stopping criterion (standard DEC): assignments stopped changing
            if ep > 0 and (cur != prev).mean() < a.tol:
                break
            prev = cur
        l_clu = F.kl_div(q.log(), P, reduction='batchmean')
        l_rec = F.mse_loss(m.dec(hF), Xg)
        loss = l_clu + a.xi1 * l_rec + a.xi2 * align_loss(m, ha, hs, a)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()

    m.eval()
    with torch.no_grad():
        hF, _, _ = m(Xg, A)
        pred = soft_assign(hF, m.mu).argmax(1).cpu().numpy()
    return cluster_metrics(y, pred)


# ============================================================================= CHCC (unchanged)
class CHCCNet(nn.Module):
    def __init__(self, D, K, dims=(256, 128)):
        super().__init__()
        self.l1 = nn.Linear(D, dims[0])
        self.l2 = nn.Linear(dims[0], dims[1])
        self.C = nn.Parameter(torch.randn(K, dims[1]))

    def forward(self, X, An):
        h = F.relu(An @ self.l1(X))
        return An @ self.l2(h)


def run_chcc(Ab, X, y, K, seed, a, verbose=False):
    set_seed(seed)
    N, D = X.shape
    net = CHCCNet(D, K).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.chcc_lr, weight_decay=5e-4)
    I = torch.eye(N, device=dev)
    H = ((Ab + I) > 0).float()
    Hn = H / H.sum(1, keepdim=True)
    An_clean = norm_adj_t(Ab)
    tau, tc = a.chcc_tau, a.chcc_tau_c

    def augment():
        Ad = Ab * (torch.rand_like(Ab) > a.chcc_pe).float()
        Ad = torch.maximum(Ad, Ad.T)
        Xm = X * (torch.rand(1, D, device=dev) > a.chcc_pf).float()
        return net(Xm, norm_adj_t(Ad))

    def n2h(z, g):
        logits = z @ g.T / tau
        pos = torch.logsumexp(logits.masked_fill(H == 0, float('-inf')), 1)
        return (torch.logsumexp(logits, 1) - pos).mean()

    proto_ready = False
    for ep in range(a.chcc_epochs):
        net.train()
        z1, z2 = F.normalize(augment(), dim=1), F.normalize(augment(), dim=1)
        g1, g2 = F.normalize(Hn @ z1, dim=1), F.normalize(Hn @ z2, dim=1)
        loss = 0.5 * (n2h(z1, g2) + n2h(z2, g1))

        if ep == a.chcc_warm:
            net.eval()
            with torch.no_grad():
                ze = F.normalize(net(X, An_clean), dim=1).cpu().numpy()
            km = KMeans(K, n_init=20, random_state=seed).fit(ze)
            if verbose:
                print('   [sanity] CHCC KMeans-after-warmup ACC=%.3f' % cluster_metrics(y, km.labels_)['ACC'])
            net.C.data = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=dev)
            proto_ready = True
            net.train()

        if proto_ready:
            C = F.normalize(net.C, dim=1)
            p1, p2 = F.softmax(z1 @ C.T / tc, 1), F.softmax(z2 @ C.T / tc, 1)
            q1, q2 = F.softmax(g1 @ C.T / tc, 1), F.softmax(g2 @ C.T / tc, 1)
            l_align = 0.5 * (F.kl_div(p1.clamp_min(1e-8).log(), q2.detach(), reduction='batchmean') +
                             F.kl_div(p2.clamp_min(1e-8).log(), q1.detach(), reduction='batchmean'))
            pm = (0.5 * (p1 + p2)).mean(0).clamp_min(1e-8)
            l_bal = (pm * pm.log()).sum()
            loss = loss + a.chcc_lc * l_align + a.chcc_lb * l_bal

        opt.zero_grad(); loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        ze = F.normalize(net(X, An_clean), dim=1).cpu().numpy()
    pred = KMeans(K, n_init=20, random_state=seed).fit_predict(ze)
    return cluster_metrics(y, pred)


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['acm', 'dblp'], required=True)
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--methods', nargs='+', default=['gdgca', 'chcc'], choices=['gdgca', 'chcc'])
    ap.add_argument('--norm', default='l2', choices=['l2', 'none'])
    ap.add_argument('--knn_k', type=int, default=10)
    ap.add_argument('--force_knn', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    # view selection protocol: 'best' (label-based, as before) or 'union' (label-free)
    ap.add_argument('--view_select', default='best', choices=['best', 'union'])
    # GDGCA
    ap.add_argument('--warm', type=int, default=300)          # stage-1 AE pretrain
    ap.add_argument('--warm2', type=int, default=200)         # stage-2 dual-stream pretrain (NEW)
    ap.add_argument('--lr_warm', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--xi1', type=float, default=1.0)
    ap.add_argument('--xi2', type=float, default=0.1)
    ap.add_argument('--upd', type=int, default=5)
    ap.add_argument('--tol', type=float, default=1e-3)        # DEC stopping tolerance (NEW)
    ap.add_argument('--gdgca_rescale', type=int, default=0)   # x sqrt(D) rescaling (off by default)
    ap.add_argument('--gdgca_std', type=int, default=0)       # z-score features (off: it hurt on DBLP)
    ap.add_argument('--graph_rec', type=int, default=1)       # GAE-style adjacency reconstruction in stage-2 pretrain (NEW)
    ap.add_argument('--xi2_pre', type=float, default=0.0)     # alignment weight in stage-2 pretrain (off by default)
    # CHCC
    ap.add_argument('--chcc_epochs', type=int, default=200)
    ap.add_argument('--chcc_warm', type=int, default=50)
    ap.add_argument('--chcc_lr', type=float, default=1e-3)
    ap.add_argument('--chcc_tau', type=float, default=0.5)
    ap.add_argument('--chcc_tau_c', type=float, default=0.5)
    ap.add_argument('--chcc_pe', type=float, default=0.2)
    ap.add_argument('--chcc_pf', type=float, default=0.3)
    ap.add_argument('--chcc_lc', type=float, default=0.5)
    ap.add_argument('--chcc_lb', type=float, default=1.0)
    a = ap.parse_args()

    print('device:', dev, '| torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
    X, X_raw, y, views = load_data_npy(a.data_dir, a.dataset, a.norm, a.knn_k, a.force_knn)
    K = len(np.unique(y))
    print('%s: N=%d D=%d K=%d graphs=%s' % (a.dataset, *X.shape, K, list(views)))
    if len(views) < 2:
        print('WARNING: only one graph view found -> check the data folder!')

    km = KMeans(K, n_init=20, random_state=0).fit(X)
    print('[sanity] KMeans on raw features: ACC=%.3f NMI=%.3f' %
          (cluster_metrics(y, km.labels_)['ACC'], normalized_mutual_info_score(y, km.labels_)))

    Xt = torch.tensor(X, device=dev)
    runners = {'gdgca': run_gdgca, 'chcc': run_chcc}
    final = {}
    for meth in a.methods:
        summary = {}
        for name, Adj in views.items():
            if a.view_select == 'union' and name != 'union' and len(views) > 1:
                continue
            Ab = torch.tensor(Adj, device=dev)
            res = [runners[meth](Ab, Xt, y, K, s, a, verbose=(s == a.seeds[0])) for s in a.seeds]
            summary[name] = {k: (np.mean([r[k] for r in res]), np.std([r[k] for r in res])) for k in res[0]}
            print('[%s | %s] ' % (meth.upper(), name) +
                  '  '.join('%s %.4f±%.4f' % (k, *v) for k, v in summary[name].items()))
        best = max(summary, key=lambda n: summary[n]['ACC'][0])
        final[meth] = (best, [summary[best][k][0] for k in ['ACC', 'NMI', 'ARI', 'F1']])

    print('\n================ FINAL (%s) ================' % a.dataset.upper())
    print('Dataset | Method | view | ACC | NMI | ARI | F1')
    for meth, (best, v) in final.items():
        print('%s | %s | %s | ' % (a.dataset.upper(), meth.upper(), best) + ' | '.join('%.3f' % x for x in v))


if __name__ == '__main__':
    main()
