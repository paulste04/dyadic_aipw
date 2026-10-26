"""
Estimators, cross-fitting and inference for the dyadic ASF  m(w,x).

Every estimator is represented by its KERNEL MATRIX phi[i, j] (i = sender, j = receiver,
diagonal NaN); the estimate is the average over i != j, and one inference routine
(dyadic psi-hat) serves all estimators.

Estimators (Parts III, VII, VIII of the theory note):
  reg   : Graham's proxy regression               phi = q(R_i,S_j)
  ht    : Horvitz-Thompson IPW                   phi = a_i b_j Y_ij
  hajek : normalised IPW                         phi = a_i b_j (Y_ij - m_H)/wbar + m_H
  aipw  : naive dyadic AIPW                      phi = q + a b (Y - q)
  mr    : multiply robust                        phi = q^e + q^a - q + a(q - q^e) + b(q - q^a) + a b (Y - q)
"""
import numpy as np
from .nuisances import Logit, LGBScore, ParamQ, LGBQ, OneDim, score_basis

ESTIMATORS = ("reg", "ht", "hajek", "aipw", "mr")


# ====================================================================== specification
class Spec:
    """Which nuisance models are correctly specified / which learners are used."""

    def __init__(self, family="gauss", q_ok=True, pw_ok=True, px_ok=True, pa_ok=True,
                 q_learner="param", crossfit=None, K=5, trim=None, lgb_kw=None):
        self.family, self.q_ok, self.pw_ok, self.px_ok, self.pa_ok = family, q_ok, pw_ok, px_ok, pa_ok
        self.q_learner, self.crossfit, self.K, self.trim = q_learner, crossfit, K, trim
        self.lgb_kw = lgb_kw or {}

    def new_q(self):
        return ParamQ(self.family, self.q_ok) if self.q_learner == "param" else LGBQ(self.family, **self.lgb_kw)


# ====================================================================== helpers
def weights(W, X, pw1, px1, target, trim=None):
    """a_i, b_j and the product a_i b_j for target (w, x). pw1, px1 = P(W=1|R), P(X=1|S)."""
    w, x = target
    pw = pw1 if w == 1 else 1 - pw1
    px = px1 if x == 1 else 1 - px1
    if trim is not None and trim[0] == "agent":           # truncate each score at t
        pw, px = np.maximum(pw, trim[1]), np.maximum(px, trim[1])
    a, b = (W == w) / pw, (X == x) / px
    ab = a[:, None] * b[None, :]
    if trim is not None and trim[0] == "dyad":            # cap the product weight
        ab = np.minimum(ab, 1.0 / trim[1])
    return a, b, ab


def fit_scores(data, idx, spec):
    R, S, W, X = data["R"], data["S"], data["W"], data["X"]
    mw = Logit(spec.pw_ok).fit(R[idx], W[idx])
    mx = mw if data.get("wto") else Logit(spec.px_ok).fit(S[idx], X[idx])
    return mw, mx


def fit_q(data, senders, receivers, target, spec, dyad_mask=None):
    """Fit q(r, s) at target on dyads (i in senders, j in receivers, i != j) in the target cell."""
    R, S, W, X, Y = data["R"], data["S"], data["W"], data["X"], data["Y"]
    w, x = target
    si = senders[W[senders] == w]
    rj = receivers[X[receivers] == x]
    Ii, Jj = np.meshgrid(si, rj, indexing="ij")
    keep = Ii != Jj
    if dyad_mask is not None:
        keep &= dyad_mask[Ii, Jj]
    ii, jj = Ii[keep], Jj[keep]
    return spec.new_q().fit(R[ii], S[jj], Y[ii, jj])


def fit_partial(data, agents, qmod, mw, mx, target, spec):
    """MR partial averages from doubly robust pseudo-outcomes (Lemma 5), on dyads within `agents`."""
    R, S, W, X, Y = data["R"], data["S"], data["W"], data["X"], data["Y"]
    w, x = target
    A = np.asarray(agents)
    Rs, Ss = R[A], S[A]
    qg = qmod.predict(Rs[:, None], Ss[None, :])
    # same trimming as in the final kernel (before 26 Oct the pseudo-outcomes were left untrimmed)
    a, b, _ = weights(W[A], X[A], mw.predict(Rs), mx.predict(Ss), target, spec.trim)
    Yg = Y[np.ix_(A, A)]
    off = ~np.eye(len(A), dtype=bool)
    # receiver side: rows i (any W_i), receivers j with X_j = x; per-j mean over senders
    Pa = np.where(off, qg + a[:, None] * (Yg - qg), np.nan)
    pa_j = np.nanmean(Pa, axis=0)
    rec = X[A] == x
    qa = OneDim(spec.family, spec.pa_ok).fit(Ss[rec], pa_j[rec])
    Pe = np.where(off, qg + b[None, :] * (Yg - qg), np.nan)
    pe_i = np.nanmean(Pe, axis=1)
    sen = W[A] == w
    qe = OneDim(spec.family, spec.pa_ok).fit(Rs[sen], pe_i[sen])
    return qe, qa


def kernels_from(Y, W, X, target, pw1, px1, qg, qe_v, qa_v, trim=None, which=ESTIMATORS):
    """Kernel matrices given nuisance VALUES on the grid (vectors for agents, matrix for q)."""
    a, b, ab = weights(W, X, pw1, px1, target, trim)
    N = len(W)
    off = ~np.eye(N, dtype=bool)
    out = {}
    if "reg" in which:
        out["reg"] = qg
    if "ht" in which:
        out["ht"] = ab * Y
    if "hajek" in which:
        mh = np.nansum(np.where(off, ab * Y, 0)) / ab[off].sum()
        wbar = ab[off].mean()
        out["hajek"] = ab * (Y - mh) / wbar + mh
    if "aipw" in which:
        out["aipw"] = qg + ab * (Y - qg)
    if "mr" in which and qe_v is not None:
        qe_i, qa_j = qe_v[:, None], qa_v[None, :]
        out["mr"] = qe_i + qa_j - qg + a[:, None] * (qg - qe_i) + b[None, :] * (qg - qa_j) + ab * (Y - qg)
    for k in out:
        out[k] = np.where(off, out[k], np.nan)
    return out


# ====================================================================== estimation with / without cross-fitting
def estimate(data, target, spec, rng=None):
    """Return dict estimator -> kernel matrix, plus the fitted score models (no-CF case) for corrections."""
    R, S, W, X, Y = data["R"], data["S"], data["W"], data["X"], data["Y"]
    N = len(R)
    allA = np.arange(N)

    if spec.crossfit is None:
        mw, mx = fit_scores(data, allA, spec)
        qmod = fit_q(data, allA, allA, target, spec)
        qe, qa = fit_partial(data, allA, qmod, mw, mx, target, spec)
        K = kernels_from(Y, W, X, target, mw.predict(R), mx.predict(S),
                         qmod.predict(R[:, None], S[None, :]), qe.predict(R), qa.predict(S), spec.trim)
        return K, (mw, mx)

    if spec.crossfit == "dyad":
        # WRONG way: random folds of dyads; q-hat for each dyad from a model not trained on that dyad
        mw, mx = fit_scores(data, allA, spec)
        fold = rng.integers(spec.K, size=(N, N))
        qg = np.empty((N, N))
        for f in range(spec.K):
            qmod = fit_q(data, allA, allA, target, spec, dyad_mask=(fold != f))
            pred = qmod.predict(R[:, None], S[None, :])
            qg[fold == f] = pred[fold == f]
        # partial averages from the out-of-fold q-hat (pseudo-outcomes in-sample)
        a, b, _ = weights(W, X, mw.predict(R), mx.predict(S), target)
        off = ~np.eye(N, dtype=bool)
        pa_j = np.nanmean(np.where(off, qg + a[:, None] * (Y - qg), np.nan), axis=0)
        pe_i = np.nanmean(np.where(off, qg + b[None, :] * (Y - qg), np.nan), axis=1)
        w, x = target
        qa = OneDim(spec.family, spec.pa_ok).fit(S[X == x], pa_j[X == x])
        qe = OneDim(spec.family, spec.pa_ok).fit(R[W == w], pe_i[W == w])
        K = kernels_from(Y, W, X, target, mw.predict(R), mx.predict(S), qg,
                         qe.predict(R), qa.predict(S), spec.trim)
        return K, None

    # ---------------- agent-level cross-fitting (Section 21.1), nested split for MR
    folds = rng.permutation(N) % spec.K
    kern = {k: np.full((N, N), np.nan) for k in ESTIMATORS}
    for k in range(spec.K):
        for l in range(k, spec.K):
            test = (folds == k) | (folds == l)
            T = allA[~test]
            Tp = rng.permutation(T)
            T1, T2 = np.sort(Tp[: len(T) // 2]), np.sort(Tp[len(T) // 2:])
            # AIPW / IPW / reg: nuisances on all of T
            mw, mx = fit_scores(data, T, spec)
            qmod = fit_q(data, T, T, target, spec)
            # MR: first stage on T1, partial averages on T2
            mw1, mx1 = fit_scores(data, T1, spec)
            qmod1 = fit_q(data, T1, T1, target, spec)
            qe, qa = fit_partial(data, T2, qmod1, mw1, mx1, target, spec)
            for (fs, fr) in {(k, l), (l, k)}:
                si, rj = allA[folds == fs], allA[folds == fr]
                Yb = Y[np.ix_(si, rj)]
                a, b, ab = weights(W[si], X[rj], mw.predict(R[si]), mx.predict(S[rj]), target, spec.trim)
                qg = qmod.predict(R[si][:, None], S[rj][None, :])
                kern["reg"][np.ix_(si, rj)] = qg
                kern["ht"][np.ix_(si, rj)] = ab * Yb
                kern["aipw"][np.ix_(si, rj)] = qg + ab * (Yb - qg)
                kern["hajek"][np.ix_(si, rj)] = ab        # placeholder: weights, normalised below
                a1, b1, ab1 = weights(W[si], X[rj], mw1.predict(R[si]), mx1.predict(S[rj]), target, spec.trim)
                qg1 = qmod1.predict(R[si][:, None], S[rj][None, :])
                qe_i, qa_j = qe.predict(R[si])[:, None], qa.predict(S[rj])[None, :]
                kern["mr"][np.ix_(si, rj)] = (qe_i + qa_j - qg1 + a1[:, None] * (qg1 - qe_i)
                                              + b1[None, :] * (qg1 - qa_j) + ab1 * (Yb - qg1))
    off = ~np.eye(N, dtype=bool)
    wmat = kern["hajek"]
    mh = np.nansum(np.where(off, wmat * Y, 0)) / wmat[off].sum()
    kern["hajek"] = wmat * (Y - mh) / wmat[off].mean() + mh
    for k in kern:
        kern[k] = np.where(off, kern[k], np.nan)
    return kern, None


# ====================================================================== inference
def dyadic_psi(phi):
    """psi-hat_i = (row_i + col_i)/(N-1) of the centred kernel (Section 22)."""
    N = phi.shape[0]
    m = np.nanmean(phi)
    eps = np.nan_to_num(phi - m)
    return m, (eps.sum(1) + eps.sum(0)) / (N - 1), eps


def se_recipes(phi):
    N = phi.shape[0]
    nd = N * (N - 1)
    m, psi, eps = dyadic_psi(phi)
    row, col = eps.sum(1), eps.sum(0)
    return m, psi, {
        "iid": np.sqrt((eps**2).sum()) / nd,
        "sender": np.sqrt((row**2).sum()) / nd,
        "twoway": np.sqrt(max((row**2).sum() + (col**2).sum() - (eps**2).sum(), 0)) / nd,
        "dyadic": np.sqrt(np.mean(psi**2) / N),
    }


def score_correction(phi_centred_raw, data, mw, mx, target):
    """Plug-in correction M I^{-1} s for HT / Hajek with parametric logit scores (Thms 2-3).
    phi_centred_raw: matrix whose derivative w.r.t. the scores is -phi * d(R_i) (sender) and
    -phi * d(S_j) (receiver): a b Y for HT, a b (Y - m_H)/wbar for Hajek."""
    R, S, W, X = data["R"], data["S"], data["W"], data["X"]
    w, x = target
    N = len(R)
    nd = N * (N - 1)
    E = np.nan_to_num(phi_centred_raw)

    def block(model, prox, D, side_target, sum_axis):
        Z = model.design(prox)
        p1 = model.predict(prox)
        dlog = ((1 - p1) if side_target == 1 else -p1)[:, None] * Z    # d log p_w / d alpha
        tot = E.sum(axis=sum_axis)                                     # sum over the partner
        M = -(tot[:, None] * dlog).sum(0) / nd
        return M

    Zw = mw.design(R); p1w = mw.predict(R)
    Iw = (Zw * (p1w * (1 - p1w))[:, None]).T @ Zw / N
    sw = Zw * (W - p1w)[:, None]
    Mw = block(mw, R, W, w, 1)
    if data.get("wto"):
        Mx = block(mx, S, X, x, 0)
        return sw @ np.linalg.solve(Iw, Mw + Mx)
    Zx = mx.design(S); p1x = mx.predict(S)
    Ix = (Zx * (p1x * (1 - p1x))[:, None]).T @ Zx / N
    sx = Zx * (X - p1x)[:, None]
    Mx = block(mx, S, X, x, 0)
    return sw @ np.linalg.solve(Iw, Mw) + sx @ np.linalg.solve(Ix, Mx)


def analyse(data, spec, targets=((1, 1), (0, 1)), rng=None, estimators=ESTIMATORS):
    """Estimate m at each target and the effect tau = m(t0) - m(t1); return list of result dicts."""
    res, psis = [], {}
    for t in targets:
        K, scores = estimate(data, t, spec, rng)
        for est in estimators:
            phi = K[est]
            m, psi, ses = se_recipes(phi)
            corr = None
            if scores is not None and est in ("ht", "hajek"):
                off = ~np.eye(phi.shape[0], dtype=bool)
                mw, mx = scores
                a, b, ab = weights(data["W"], data["X"], mw.predict(data["R"]), mx.predict(data["S"]), t, spec.trim)
                if est == "ht":
                    raw = ab * data["Y"]
                else:
                    raw = (phi - m)               # = a b (Y - m_H)/wbar
                corr = psi + score_correction(np.where(off, raw, 0), data, mw, mx, t)
                ses["dyadic_corr"] = np.sqrt(np.mean(corr**2) / len(psi))
            psis[(est, t)] = corr if corr is not None else psi
            res.append(dict(estimator=est, quantity=f"m{t[0]}{t[1]}", est=m, **{f"se_{k}": v for k, v in ses.items()}))
    if len(targets) == 2:
        t0, t1 = targets
        for est in estimators:
            m0 = [r["est"] for r in res if r["estimator"] == est and r["quantity"] == f"m{t0[0]}{t0[1]}"][0]
            m1 = [r["est"] for r in res if r["estimator"] == est and r["quantity"] == f"m{t1[0]}{t1[1]}"][0]
            dpsi = psis[(est, t0)] - psis[(est, t1)]
            res.append(dict(estimator=est, quantity="tau", est=m0 - m1,
                            se_dyadic=np.sqrt(np.mean(dpsi**2) / len(dpsi))))
    return res
