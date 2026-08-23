"""Generate all figures used in the slide deck from a synthetic dataset.
Pure numpy + matplotlib so it runs anywhere. Deck color palette baked in."""
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
os.makedirs("figures", exist_ok=True)
import matplotlib.pyplot as plt

INK   = "#1B2540"
CYAN  = "#0E7490"
TEAL  = "#0F766E"
AMBER = "#B45309"
VIOLET= "#6D28D9"
MUTED = "#5B6B8C"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0, "font.family": "sans-serif",
})

# ----------------------------------------------------------------------
# 1. Synthetic "electricity-like" dataset: trend + daily + weekly + noise
# ----------------------------------------------------------------------
def make_series(n=24 * 30, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend   = 50 + 0.005 * t
    daily   = 12 * np.sin(2 * np.pi * t / 24)
    weekly  = 2 * np.sin(2 * np.pi * t / (24 * 7))
    noise   = rng.normal(0, 2.5, n)
    return t, trend + daily + weekly + noise

t, y = make_series()
H = 48                      # forecast horizon (2 days, hourly)
y_train, y_test = y[:-H], y[-H:]

# ---- Fig 1: the dataset ----
fig, ax = plt.subplots(figsize=(9, 3.0))
ax.plot(t[:-H], y_train, color=CYAN, lw=1.1, label="history")
ax.plot(t[-H:], y_test, color=AMBER, lw=1.6, label="held-out future (H=48)")
ax.axvline(t[-H], color=MUTED, ls=":", lw=1.2)
ax.set_xlabel("hour"); ax.set_ylabel("load")
ax.legend(loc="upper left", frameon=False, fontsize=11)
fig.tight_layout(); fig.savefig("figures/fig_data.png", dpi=150); plt.close(fig)

# ----------------------------------------------------------------------
# 2. Baselines: seasonal-naive (period 24) with an empirical interval
# ----------------------------------------------------------------------
m = 24
seasonal_naive = y_train[-m:][np.arange(H) % m]

# proper predictive spread: rolling-origin backtest of seasonal-naive on train
def sn_backtest_resid(y, m, H, origins=120):
    res = []
    for o in range(len(y) - H - origins, len(y) - H):
        f = y[o - m:o][np.arange(H) % m]
        res.append(y[o:o + H] - f)
    return np.array(res)                      # (origins, H) residuals per horizon

resid_h = sn_backtest_resid(y_train, m, H)
sigma_h = resid_h.std(axis=0)                 # horizon-dependent std
sigma = sigma_h.mean()
lo, hi = seasonal_naive - 1.28 * sigma_h, seasonal_naive + 1.28 * sigma_h  # 80%

fig, ax = plt.subplots(figsize=(9, 3.0))
ax.plot(np.arange(-72, 0), y_train[-72:], color=CYAN, lw=1.1, label="history")
ax.plot(np.arange(H), y_test, color=AMBER, lw=1.6, label="actual")
ax.plot(np.arange(H), seasonal_naive, color=INK, lw=1.6, ls="--",
        label="seasonal-naive")
ax.fill_between(np.arange(H), lo, hi, color=INK, alpha=0.12, label="80% interval")
ax.axvline(0, color=MUTED, ls=":", lw=1.2)
ax.set_xlabel("hour (0 = forecast origin)"); ax.set_ylabel("load")
ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=2)
fig.tight_layout(); fig.savefig("figures/fig_baseline.png", dpi=150); plt.close(fig)

mase_naive = np.mean(np.abs(y_test - seasonal_naive)) / \
             np.mean(np.abs(y_train[m:] - y_train[:-m]))
print(f"Seasonal-naive MASE = {mase_naive:.3f}")

# ----------------------------------------------------------------------
# 3. Self-attention from scratch -> attention weight heatmap
# ----------------------------------------------------------------------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

rng = np.random.default_rng(1)
L, d, dk = 10, 16, 16
# patch-embed 10 windows of the series as tokens
patches = y_train[-L * m::m][:L]                     # one value per day
X = np.stack([patches, np.gradient(patches)], axis=1)  # level + slope feature
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
X = X @ rng.normal(0, 1, (2, d)) / np.sqrt(2)         # lift to width d
Wq, Wk, Wv = (rng.normal(0, 1, (d, dk)) / np.sqrt(d) for _ in range(3))
Q, K, V = X @ Wq, X @ Wk, X @ Wv
A = softmax(Q @ K.T / np.sqrt(dk))                    # (L, L) attention

fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(A, cmap="cividis", aspect="equal")
ax.set_xlabel("key (attended-to step)"); ax.set_ylabel("query (output step)")
ax.set_xticks(range(L)); ax.set_yticks(range(L))
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")
fig.tight_layout(); fig.savefig("figures/fig_attention.png", dpi=150); plt.close(fig)

# ----------------------------------------------------------------------
# 4. Diffusion forward process on a single window
# ----------------------------------------------------------------------
N = 100
betas = np.linspace(1e-4, 0.08, N)
alphas = 1 - betas
abar = np.cumprod(alphas)
x0 = (y_test - y_test.mean()) / y_test.std()
levels = [0, 20, 50, 99]
fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.4), sharey=True)
for ax, n in zip(axes, levels):
    eps = np.random.default_rng(n).standard_normal(len(x0))
    xn = np.sqrt(abar[n]) * x0 + np.sqrt(1 - abar[n]) * eps
    ax.plot(xn, color=AMBER, lw=1.2)
    ax.set_title(f"n = {n}", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].set_ylabel("value")
fig.suptitle(r"forward noising:  $x^{(n)}=\sqrt{\bar\alpha_n}\,x^{(0)}+\sqrt{1-\bar\alpha_n}\,\epsilon$",
             fontsize=13, color=INK)
fig.tight_layout(); fig.savefig("figures/fig_diffusion.png", dpi=150); plt.close(fig)

# ----------------------------------------------------------------------
# 5. Probabilistic forecast: sample paths -> quantile fan + coverage
# ----------------------------------------------------------------------
# toy generative forecaster: seasonal-naive mean + correlated noise samples
S, phi = 200, 0.6
rng7 = np.random.default_rng(7)
ar = np.zeros((S, H))
ar[:, 0] = rng7.normal(0, sigma_h[0], S)             # start at horizon-1 spread
for h in range(1, H):
    innov = sigma_h[h] * np.sqrt(1 - phi**2)
    ar[:, h] = phi * ar[:, h - 1] + rng7.normal(0, innov, S)
samples = seasonal_naive[None, :] + ar
q10, q50, q90 = np.percentile(samples, [10, 50, 90], axis=0)

fig, ax = plt.subplots(figsize=(9, 3.0))
for s in samples[:40]:
    ax.plot(np.arange(H), s, color=VIOLET, lw=0.4, alpha=0.12)
ax.fill_between(np.arange(H), q10, q90, color=VIOLET, alpha=0.18, label="80% band")
ax.plot(np.arange(H), q50, color=VIOLET, lw=1.6, label="median")
ax.plot(np.arange(H), y_test, color=AMBER, lw=1.6, label="actual")
ax.set_xlabel("hour"); ax.set_ylabel("load")
ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=3)
fig.tight_layout(); fig.savefig("figures/fig_samples.png", dpi=150); plt.close(fig)

# CRPS (sample-based) and coverage
def crps_sample(samps, obs):
    s = np.sort(samps, axis=0); n = s.shape[0]
    t1 = np.mean(np.abs(samps - obs[None, :]), axis=0)
    diff = np.abs(s[1:] - s[:-1])
    w = (np.arange(1, n) * (n - np.arange(1, n)))[:, None]
    t2 = (w * diff).sum(0) / n**2
    return np.mean(t1 - t2)

cov = np.mean((y_test >= q10) & (y_test <= q90))
print(f"CRPS = {crps_sample(samples, y_test):.3f}   80%-coverage = {cov:.2f}")

# ----------------------------------------------------------------------
# 6. Classical VAR(1): simulate -> OLS fit -> impulse response
# ----------------------------------------------------------------------
A_true = np.array([[0.5, 0.3],
                   [0.0, 0.4]])           # x2 does NOT depend on x1's past
def simulate_var1(A, T=400, seed=3):
    rng = np.random.default_rng(seed)
    k = A.shape[0]; Y = np.zeros((T, k))
    for s in range(1, T):
        Y[s] = A @ Y[s - 1] + rng.normal(0, 1, k)
    return Y

Y = simulate_var1(A_true)

def fit_var1(Y):                          # OLS: Y_t = A Y_{t-1} + e
    X, Z = Y[1:], Y[:-1]
    A_hat = np.linalg.lstsq(Z, X, rcond=None)[0].T
    return A_hat

A_hat = fit_var1(Y)

# impulse response: B_h = A^h  (response to unit shocks)
hmax = 10
B = [np.linalg.matrix_power(A_hat, h) for h in range(hmax + 1)]
B = np.array(B)                           # (h, response_i, shock_j)

labels = [r"$y^{(1)}$", r"$y^{(2)}$"]
fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.4), sharex=True)
for i in range(2):
    for j in range(2):
        ax = axes[i, j]
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        ax.bar(range(hmax + 1), B[:, i, j], color=TEAL, width=0.6)
        ax.set_title(f"shock to {labels[j]} $\\rightarrow$ {labels[i]}",
                     fontsize=11)
        if i == 1:
            ax.set_xlabel("horizon $h$")
fig.suptitle("Estimated impulse-response functions  $B^{(h)}=A^{h}$",
             fontsize=13, color=INK)
fig.tight_layout(); fig.savefig("figures/fig_irf.png", dpi=150); plt.close(fig)

print("A_true =\n", A_true)
print("A_hat  =\n", A_hat.round(2))
print("Figures written.")
