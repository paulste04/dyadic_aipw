"""
Data-generating processes for the dyadic AIPW / MR Monte Carlo.

Agents i = 1..N carry proxies (R_i, S_i), latent parts (eta_i, zeta_i), policies (W_i, X_i).
Sender type A_i = (R_i, eta_i), receiver type B_i = (S_i, zeta_i).

Proxies are Uniform(-sqrt3, sqrt3) marginally (variance 1, bounded => controllable overlap);
within-agent dependence between R_i and S_i via a Gaussian copula with correlation `rho`.

Policies (selection on proxies, assumption (U)):
    P(W_i=1 | R_i, eta_i) = Lambda( gamma*(-0.2 + 0.8 R + 0.5 R^2) + lam * eta_i/sd_lat )
    P(X_i=1 | S_i, zeta_i) = Lambda( gamma*( 0.2 - 0.8 S + 0.5 S^2) + lam * zeta_i/sd_lat )
lam != 0 violates (U) (experiment E7).  wto=True: S_i = R_i and X_i = W_i (one policy, both roles).

Outcomes, potential outcome at (w, x):
  gauss  : Y = 1 + w(0.5+0.5R_i) + x(0.5-0.3S_j) + R_i + S_j + c(R_i^2+S_j^2) + d R_i S_j
               + e R_i^2 S_j^2 + eta_i + zeta_j + V_ij,         V_ij ~ N(0,1)
  poisson: Y ~ Poisson( exp( b0 + w(0.4+0.2R_i) + x(0.3-0.2S_j) + 0.5R_i + 0.5S_j
               + cp(R_i^2+S_j^2) + dp R_i S_j + eta_i + zeta_j ) )
The truth m(w,x) integrates over the PRODUCT of the marginal type distributions.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import norm

SQ3 = np.sqrt(3.0)


def lam(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class DGP:
    family: str = "gauss"          # "gauss" or "poisson"
    rho: float = 0.0               # within-agent corr of (R_i, S_i)
    gamma: float = 1.0             # overlap knob (score index scale)
    lam_u: float = 0.0             # violation of (U)
    wto: bool = False              # one policy plays both roles
    sd_lat: float = 0.5
    # gauss outcome
    c: float = 0.7
    d: float = 0.5
    e: float = 0.4
    # poisson outcome
    b0: float = -0.5
    cp: float = 0.15
    dp: float = 0.3

    # ---------- score indices ----------
    def idx_w(self, r):
        return self.gamma * (-0.2 + 0.8 * r + 0.5 * r**2)

    def idx_x(self, s):
        if self.wto:
            return self.idx_w(s)
        return self.gamma * (0.2 - 0.8 * s + 0.5 * s**2)

    # ---------- potential-outcome means given types ----------
    def mean_outcome(self, w, x, R, S, eta, zeta):
        """E[Y_ij(w,x) | A_i, B_j]; arrays broadcast (R, eta: senders; S, zeta: receivers)."""
        if self.family == "gauss":
            return (1 + w * (0.5 + 0.5 * R) + x * (0.5 - 0.3 * S) + R + S
                    + self.c * (R**2 + S**2) + self.d * R * S + self.e * R**2 * S**2 + eta + zeta)
        lin = (self.b0 + w * (0.4 + 0.2 * R) + x * (0.3 - 0.2 * S) + 0.5 * R + 0.5 * S
               + self.cp * (R**2 + S**2) + self.dp * R * S + eta + zeta)
        return np.exp(lin)

    # ---------- truth ----------
    def truth(self, w, x, n_nodes=200):
        if self.family == "gauss":
            # product measure: E[R^2]=E[S^2]=1, E[R]=E[S]=E[RS]=0, E[R^2 S^2]=1
            return 1 + 0.5 * w + 0.5 * x + 2 * self.c + self.e
        # Gauss-Legendre on [-sqrt3, sqrt3]^2 (uniform marginals), latent parts analytically
        t, wt = np.polynomial.legendre.leggauss(n_nodes)
        r = SQ3 * t
        wt = wt / 2.0                                   # density 1/(2 sqrt3) * Jacobian sqrt3
        Rg, Sg = np.meshgrid(r, r, indexing="ij")
        val = self.mean_outcome(w, x, Rg, Sg, 0.0, 0.0) * np.exp(self.sd_lat**2)   # E e^eta e^zeta
        return float(wt @ val @ wt)

    # ---------- draw a sample ----------
    def draw(self, N, rng):
        z1, z2 = rng.normal(size=N), rng.normal(size=N)
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
        R = SQ3 * (2 * norm.cdf(z1) - 1)
        S = R.copy() if self.wto else SQ3 * (2 * norm.cdf(z2) - 1)
        eta = self.sd_lat * rng.normal(size=N)
        zeta = self.sd_lat * rng.normal(size=N)
        W = rng.binomial(1, lam(self.idx_w(R) + self.lam_u * eta / self.sd_lat))
        if self.wto:
            X = W.copy()
        else:
            X = rng.binomial(1, lam(self.idx_x(S) + self.lam_u * zeta / self.sd_lat))
        Ri, Sj = R[:, None], S[None, :]
        mu = self.mean_outcome(W[:, None], X[None, :], Ri, Sj, eta[:, None], zeta[None, :])
        if self.family == "gauss":
            Y = mu + rng.normal(size=(N, N))
        else:
            Y = rng.poisson(mu).astype(float)
        np.fill_diagonal(Y, np.nan)
        return dict(R=R, S=S, W=W, X=X, Y=Y, eta=eta, zeta=zeta)

    def true_scores(self, R, S):
        """Population scores P(W=1|R), P(X=1|S) when (U) holds (lam_u = 0)."""
        return lam(self.idx_w(R)), lam(self.idx_x(S))
