"""
Numerical check of the Step 6b variance results (true nuisances, no estimation).

Claims checked:
  (1) N * Var(m_AIPW)  ->  Var(psi_AIPW) = sum over roles of
        Var(q^e(R)) + E[(g^e(A) - q^e(R))^2 / p_w(R)]
  (2) N * [Var(m_Hajek) - Var(m_AIPW)]  ->  E[(q^e - m)^2 (1-p_w)/p_w] + E[(q^a - m)^2 (1-p_x)/p_x]

DGP as in mr_aipw_sim.py, evaluated at (w,x) = (1,1):
  hbar(A_i, B_j) = 2 + R + S + c(R^2+S^2) + d R S + e R^2 S^2 + eta_i + zeta_j,
  sender type A_i = (R_i, eta_i), receiver type B_j = (S_j, zeta_j); roles independent within agent.
"""
import numpy as np

rng = np.random.default_rng(6)
c, d, e = 0.7, 0.5, 0.4
m = 2 + 2 * c + e
sd_lat = 0.5
lam = lambda z: 1 / (1 + np.exp(-z))
pw = lambda r: lam(-0.2 + 0.8 * r + 0.5 * r**2)
px = lambda s: lam(0.2 - 0.8 * s + 0.5 * s**2)
qe = lambda r: 2 + r + c * (r**2 + 1) + e * r**2          # E_S[q(r,S)]
qa = lambda s: 2 + s + c * (1 + s**2) + e * s**2          # E_R[q(R,s)]

# ---- theoretical quantities by Monte Carlo integration ----
Z = rng.normal(size=4_000_000)
var_aipw = (np.var(qe(Z)) + sd_lat**2 * np.mean(1 / pw(Z))
            + np.var(qa(Z)) + sd_lat**2 * np.mean(1 / px(Z)))
gap = (np.mean((qe(Z) - m) ** 2 * (1 - pw(Z)) / pw(Z))
       + np.mean((qa(Z) - m) ** 2 * (1 - px(Z)) / px(Z)))

# ---- finite-sample Monte Carlo ----
N, REPS = 400, 1000
est = np.empty((REPS, 2))
off = ~np.eye(N, dtype=bool)
for rep in range(REPS):
    R, S = rng.normal(size=N), rng.normal(size=N)
    eta, zeta = sd_lat * rng.normal(size=N), sd_lat * rng.normal(size=N)
    W = rng.binomial(1, pw(R))
    X = rng.binomial(1, px(S))
    Ri, Sj = R[:, None], S[None, :]
    q = 2 + Ri + Sj + c * (Ri**2 + Sj**2) + d * Ri * Sj + e * Ri**2 * Sj**2
    Y = q + eta[:, None] + zeta[None, :] + rng.normal(size=(N, N))   # outcome at (1,1)
    a, b = W / pw(R), X / px(S)
    ab = a[:, None] * b[None, :]
    hajek = (ab * Y)[off].sum() / ab[off].sum()
    aipw = (q + ab * (Y - q))[off].mean()
    est[rep] = hajek, aipw

v_h, v_a = N * est.var(axis=0)
print(f"true m = {m:.3f}; mean Hajek = {est[:,0].mean():.3f}, mean AIPW = {est[:,1].mean():.3f}")
print(f"(1) N*Var(AIPW):            simulated {v_a:7.3f}   theory {var_aipw:7.3f}")
print(f"(2) N*[Var(Hajek)-Var(AIPW)]: simulated {v_h - v_a:7.3f}   theory {gap:7.3f}")
