"""
Checks of the simulation package against results proved in the theory note.
Run:  python -m pytest -q tests   (from sims/)
"""
import numpy as np
from dyadsim import DGP, Spec, analyse
from dyadsim.core import kernels_from


def test_truth_gauss_vs_quadrature():
    """Closed-form Gaussian truth equals direct numerical integration."""
    g = DGP("gauss")
    t, wt = np.polynomial.legendre.leggauss(120)
    r = np.sqrt(3) * t; wt = wt / 2
    Rg, Sg = np.meshgrid(r, r, indexing="ij")
    num = wt @ g.mean_outcome(1, 1, Rg, Sg, 0.0, 0.0) @ wt
    assert abs(num - g.truth(1, 1)) < 1e-10


def test_truth_poisson_vs_montecarlo():
    g = DGP("poisson")
    rng = np.random.default_rng(0)
    n = 2_000_000
    R, S = rng.uniform(-np.sqrt(3), np.sqrt(3), (2, n))
    eta, zeta = 0.5 * rng.normal(size=(2, n))
    mc = g.mean_outcome(1, 1, R, S, eta, zeta)
    assert abs(mc.mean() - g.truth(1, 1)) < 4 * mc.std() / np.sqrt(n)


def test_estimators_agree_on_truth_at_large_N():
    """All five estimators are consistent when every model is right (one big sample)."""
    for fam in ("gauss", "poisson"):
        g = DGP(fam)
        d = g.draw(700, np.random.default_rng(1))
        res = analyse(d, Spec(fam), targets=((1, 1),))
        for r in res:
            assert abs(r["est"] - g.truth(1, 1)) < 5 * max(r["se_dyadic"], 0.02), (fam, r)


def test_mr_reduces_to_aipw_with_implied_partial_averages():
    """MR kernel with q^e = E_S q, q^a = E_R q has the same average as naive AIPW up to O(1/N)."""
    g = DGP("gauss")
    d = g.draw(300, np.random.default_rng(2))
    R, S, W, X, Y = d["R"], d["S"], d["W"], d["X"], d["Y"]
    pw, px = g.true_scores(R, S)
    q = 2 + R[:, None] + S[None, :]              # any (wrong) q
    qe, qa = np.nanmean(q, 1), np.nanmean(q, 0)
    K = kernels_from(Y, W, X, (1, 1), pw, px, q, qe, qa)
    assert abs(np.nanmean(K["mr"]) - np.nanmean(K["aipw"])) < 0.05


def test_score_correction_shrinks_hajek_se():
    """With estimated scores the corrected Hajek SE is smaller than the uncorrected one (Thm 3)."""
    g = DGP("gauss")
    d = g.draw(300, np.random.default_rng(3))
    res = analyse(d, Spec("gauss"), targets=((1, 1),))
    h = [r for r in res if r["estimator"] == "hajek"][0]
    assert h["se_dyadic_corr"] < h["se_dyadic"]
