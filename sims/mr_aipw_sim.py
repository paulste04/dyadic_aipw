"""
Monte Carlo check: naive dyadic AIPW vs. multiply robust (MR) dyadic AIPW.

Target: m = E[Y_12(w=1, x=1)] (Graham's dyadic ASF).
DGP:
  R_i, S_i ~ N(0,1) iid (sender / receiver proxies)
  eta_i, zeta_i ~ N(0, 0.5^2)  latent parts of A_i, B_i (independent of policy -> A4 holds)
  W_i ~ Bern(Lambda(-0.2 + 0.8 R + 0.5 R^2))
  X_j ~ Bern(Lambda( 0.2 - 0.8 S + 0.5 S^2))
  Y_ij = 1 + .5 W_i + .5 X_j + R_i + S_j + c(R_i^2 + S_j^2) + d R_i S_j + 0.4 R_i^2 S_j^2 + eta_i + zeta_j + V_ij
  => q(1,1,r,s) = 2 + r + s + c(r^2+s^2) + d r s + 0.4 r^2 s^2,   m = 2 + 2c + 0.4.
Misspecification: q linear in (R,S) only; scores linear in R (resp. S) only.
"""
import numpy as np

rng = np.random.default_rng(20261018)
c, d = 0.7, 0.5
M_TRUE = 2 + 2 * c + 0.4
N, REPS = 300, 500


def logit_fit(X, y, iters=50):
    """Unpenalised logit by Newton-Raphson (true MLE)."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return 1 / (1 + np.exp(-X @ b))


def ols_pred(Xfit, yfit, Xpred):
    beta, *_ = np.linalg.lstsq(Xfit, yfit, rcond=None)
    return Xpred @ beta


def one_rep(q_ok, pw_ok, px_ok):
    R, S = rng.normal(size=N), rng.normal(size=N)
    eta, zeta = 0.5 * rng.normal(size=N), 0.5 * rng.normal(size=N)
    W = rng.binomial(1, 1 / (1 + np.exp(-(-0.2 + 0.8 * R + 0.5 * R**2))))
    X = rng.binomial(1, 1 / (1 + np.exp(-(0.2 - 0.8 * S + 0.5 * S**2))))
    Ri, Sj = R[:, None], S[None, :]
    Y = (1 + .5 * W[:, None] + .5 * X[None, :] + Ri + Sj + c * (Ri**2 + Sj**2)
         + d * Ri * Sj + 0.4 * Ri**2 * Sj**2 + eta[:, None] + zeta[None, :] + rng.normal(size=(N, N)))
    off = ~np.eye(N, dtype=bool)  # i != j

    # ---- outcome model q(1,1,r,s), fitted on treated dyads ----
    RR, SS = np.broadcast_to(Ri, (N, N)), np.broadcast_to(Sj, (N, N))
    if q_ok:
        basis = lambda r, s: np.stack([np.ones_like(r), r, s, r**2, s**2, r * s, r**2 * s**2], -1)
    else:
        basis = lambda r, s: np.stack([np.ones_like(r), r, s], -1)
    B = basis(RR, SS)
    treated = off & (W[:, None] == 1) & (X[None, :] == 1)
    qhat = ols_pred(B[treated], Y[treated], B.reshape(-1, B.shape[-1])).reshape(N, N)

    # ---- scores (unpenalised logits) ----
    Zw = np.column_stack([np.ones(N), R] + ([R**2] if pw_ok else []))
    Zx = np.column_stack([np.ones(N), S] + ([S**2] if px_ok else []))
    pw, px = logit_fit(Zw, W), logit_fit(Zx, X)
    a, b = W / pw, X / px                     # ego and alter weights
    ab = a[:, None] * b[None, :]

    def avg(M):                               # average over ordered pairs i != j
        return M[off].mean()

    # ---- naive dyadic AIPW ----
    naive = avg(qhat + ab * (Y - qhat))

    # ---- MR dyadic AIPW ----
    # q^a(s): regress pseudo-outcome qhat + a_i (Y - qhat) on poly(S_j) among X_j = 1
    #         (DR for q^a: right if q or p_w is right)
    Pa = np.where(off, qhat + a[:, None] * (Y - qhat), np.nan)
    pa_mean_j = np.nanmean(Pa, axis=0)        # equal group sizes -> same OLS as dyad level
    polyS = np.column_stack([np.ones(N), S, S**2, S**3])
    qa = ols_pred(polyS[X == 1], pa_mean_j[X == 1], polyS)
    # q^e(r): regress pseudo-outcome qhat + b_j (Y - qhat) on poly(R_i) among W_i = 1
    Pe = np.where(off, qhat + b[None, :] * (Y - qhat), np.nan)
    pe_mean_i = np.nanmean(Pe, axis=1)
    polyR = np.column_stack([np.ones(N), R, R**2, R**3])
    qe = ols_pred(polyR[W == 1], pe_mean_i[W == 1], polyR)

    qe_i, qa_j = qe[:, None], qa[None, :]
    phi = (qe_i + qa_j - qhat
           + a[:, None] * (qhat - qe_i)
           + b[None, :] * (qhat - qa_j)
           + ab * (Y - qhat))
    mr = avg(phi)
    return naive, mr


scenarios = [
    ("q ok,   pw ok,  px ok ", True, True, True),
    ("q WRONG, pw ok,  px ok ", False, True, True),
    ("q WRONG, pw ok,  px WRONG", False, True, False),
    ("q WRONG, pw WRONG, px ok ", False, False, True),
    ("q ok,   pw WRONG, px WRONG", True, False, False),
    ("all WRONG               ", False, False, False),
]

print(f"true m = {M_TRUE:.3f},  N = {N},  reps = {REPS}\n")
print(f"{'scenario':28s} | {'naive bias':>10s} {'naive sd':>9s} | {'MR bias':>8s} {'MR sd':>7s}")
for name, qo, pwo, pxo in scenarios:
    res = np.array([one_rep(qo, pwo, pxo) for _ in range(REPS)])
    bias = res.mean(0) - M_TRUE
    sd = res.std(0)
    print(f"{name:28s} | {bias[0]:10.3f} {sd[0]:9.3f} | {bias[1]:8.3f} {sd[1]:7.3f}")
