"""
Time Series Forecasting: From Transformers to Diffusion Models
Hands-on exercises (Module 4).

Pure numpy + matplotlib -- runs on any machine, no GPU needed.
Each exercise has a WORKED part (runs as-is) and a `# TODO` part for you.

Run:   python exercises.py
"""
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# Exercise 1 -- Build (or load) the dataset
# ======================================================================
def make_series(n=24 * 30, seed=2):
    """Synthetic hourly 'load': trend + daily + weekly + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend  = 50 + 0.005 * t
    daily  = 12 * np.sin(2 * np.pi * t / 24)
    weekly = 2 * np.sin(2 * np.pi * t / (24 * 7))
    noise  = rng.normal(0, 2.5, n)
    return t, trend + daily + weekly + noise

t, y = make_series()
H = 48                          # forecast horizon (2 days)
m = 24                          # daily seasonal period
y_train, y_test = y[:-H], y[-H:]

# TODO 1: replace make_series() with pd.read_csv("your.csv")["value"].values
# TODO 1b: plot the autocorrelation (np.correlate) and find the dominant period.


# ======================================================================
# Exercise 2 -- Classical VAR(1): OLS fit, Granger test, impulse response
# ======================================================================
A_true = np.array([[0.5, 0.3],
                   [0.0, 0.4]])          # note: y2 does NOT depend on y1's past

def simulate_var1(A, T=400, seed=3):
    rng = np.random.default_rng(seed)
    k = A.shape[0]
    Y = np.zeros((T, k))
    for s in range(1, T):
        Y[s] = A @ Y[s - 1] + rng.normal(0, 1, k)
    return Y

def fit_var1(Y):
    """OLS for Y_t = A Y_{t-1} + e  (no intercept here)."""
    X, Z = Y[1:], Y[:-1]
    return np.linalg.lstsq(Z, X, rcond=None)[0].T

def granger_F(Y, caused=1, cause=0):
    """Does `cause` Granger-cause `caused`? F-test of dropping the cause lag."""
    X = Y[1:, caused]
    Zu = Y[:-1]                                   # unrestricted: both lags
    Zr = Y[:-1, [caused]]                          # restricted: own lag only
    def rss(Z):
        b = np.linalg.lstsq(Z, X, rcond=None)[0]
        return ((X - Z @ b) ** 2).sum()
    rss_u, rss_r = rss(Zu), rss(Zr)
    q = Zu.shape[1] - Zr.shape[1]                  # restrictions
    n, k = len(X), Zu.shape[1]
    return ((rss_r - rss_u) / q) / (rss_u / (n - k))

Yv = simulate_var1(A_true)
A_hat = fit_var1(Yv)
F_1to2 = granger_F(Yv, caused=1, cause=0)          # expect small (no causality)
F_2to1 = granger_F(Yv, caused=0, cause=1)          # expect large
print(f"[E2] A_hat=\n{A_hat.round(2)}")
print(f"[E2] Granger F  y1->y2 = {F_1to2:.2f} (small),  "
      f"y2->y1 = {F_2to1:.2f} (large)")

# impulse response: B_h = A^h
B = np.array([np.linalg.matrix_power(A_hat, h) for h in range(11)])

# TODO 2: add an intercept and choose p by AIC; compare A_hat to A_true.
# TODO 2b: turn F into a p-value with scipy.stats.f, and confirm y1 does
#          not Granger-cause y2 at the 5% level.


# ======================================================================
# Exercise 3 -- Baselines and MASE, with a backtest-based interval
# ======================================================================
seasonal_naive = y_train[-m:][np.arange(H) % m]

def backtest_resid(y, m, H, origins=120):
    """Rolling-origin residuals of seasonal-naive -> honest predictive spread."""
    R = []
    for o in range(len(y) - H - origins, len(y) - H):
        f = y[o - m:o][np.arange(H) % m]
        R.append(y[o:o + H] - f)
    return np.array(R)

sigma_h = backtest_resid(y_train, m, H).std(axis=0)        # per-horizon std
lo, hi = seasonal_naive - 1.28 * sigma_h, seasonal_naive + 1.28 * sigma_h

def mase(y_true, y_pred, y_insample, m):
    naive = np.abs(y_insample[m:] - y_insample[:-m]).mean()
    return np.abs(y_true - y_pred).mean() / naive

print(f"[E3] seasonal-naive MASE = {mase(y_test, seasonal_naive, y_train, m):.3f}")

# TODO 3: fit AutoETS (from statsforecast) and compare its MASE to seasonal-naive.
# TODO 3b: explain why a one-step-error interval would be too narrow here.


# ======================================================================
# Exercise 4 -- Self-attention from scratch
# ======================================================================
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(X, Wq, Wk, Wv, causal=False):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    dk = Q.shape[-1]
    S = Q @ K.T / np.sqrt(dk)
    if causal:                                  # solution to TODO 3
        mask = np.triu(np.ones_like(S), k=1).astype(bool)
        S = np.where(mask, -np.inf, S)
    A = softmax(S)
    return A @ V, A

rng = np.random.default_rng(1)
L, d, dk = 10, 16, 16
patches = y_train[-L * m::m][:L]
X = np.stack([patches, np.gradient(patches)], axis=1)
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
X = X @ rng.normal(0, 1, (2, d)) / np.sqrt(2)
Wq, Wk, Wv = (rng.normal(0, 1, (d, dk)) / np.sqrt(d) for _ in range(3))
Z, A = attention(X, Wq, Wk, Wv)
assert np.allclose(A.sum(1), 1), "rows of A must sum to 1"
print(f"[E4] attention output shape {Z.shape}, rows sum to 1: OK")

# TODO 4: call attention(..., causal=True) and confirm A is lower-triangular.
# TODO 4b: implement multi-head: split d into h heads, run each, concatenate.


# ======================================================================
# Exercise 5 -- Diffusion forward process (and a reverse-step stub)
# ======================================================================
N = 100
betas = np.linspace(1e-4, 0.08, N)
alphas = 1 - betas
abar = np.cumprod(alphas)

def q_sample(x0, n, rng):
    """Forward: jump straight to noise level n."""
    eps = rng.standard_normal(len(x0))
    return np.sqrt(abar[n]) * x0 + np.sqrt(1 - abar[n]) * eps

x0 = (y_test - y_test.mean()) / y_test.std()
noised = {n: q_sample(x0, n, np.random.default_rng(n)) for n in (0, 20, 50, 99)}
print(f"[E5] noised window at n=99 has std ~ {noised[99].std():.2f} (expect ~1)")

def p_step(x_n, n, eps_theta, rng):
    """One reverse step (DDPM). eps_theta(x, n) -> predicted noise."""
    eps = eps_theta(x_n, n)
    mean = (x_n - betas[n] / np.sqrt(1 - abar[n]) * eps) / np.sqrt(alphas[n])
    if n == 0:
        return mean
    return mean + np.sqrt(betas[n]) * rng.standard_normal(len(x_n))

# TODO 5: train (or mock) eps_theta, then sample x_N ~ N(0,I) and run p_step
#         from n=N-1 down to 0 to generate a synthetic window.


# ======================================================================
# Exercise 6 -- Probabilistic forecast, CRPS, and coverage
# ======================================================================
S, phi = 200, 0.6
g = np.random.default_rng(7)
ar = np.zeros((S, H))
ar[:, 0] = g.normal(0, sigma_h[0], S)
for h in range(1, H):
    ar[:, h] = phi * ar[:, h - 1] + g.normal(0, sigma_h[h] * np.sqrt(1 - phi**2), S)
samples = seasonal_naive[None, :] + ar
q10, q50, q90 = np.percentile(samples, [10, 50, 90], axis=0)

def crps_sample(samps, obs):
    """Energy-form sample CRPS (lower is better)."""
    a = np.abs(samps - obs[None, :]).mean(0)
    b = np.abs(samps[:, None, :] - samps[None, :, :]).mean((0, 1))
    return (a - 0.5 * b).mean()

coverage = np.mean((y_test >= q10) & (y_test <= q90))
print(f"[E6] CRPS = {crps_sample(samples, y_test):.3f}   "
      f"80%-coverage = {coverage:.2f}")

# TODO 6: the band covers ~70% instead of 80% -- widen sigma_h or recalibrate,
#         then re-measure. Stretch: replace `samples` with Chronos-2 draws.

if __name__ == "__main__":
    print("\nAll worked exercises ran. Open the file and tackle the # TODO lines.")
