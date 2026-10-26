"""
Nuisance estimators: propensity scores, outcome regression (Graham's PVR), and the
one-dimensional partial-average regressions used by the multiply robust (MR) estimator.

All maximum-likelihood fits are UNPENALISED (a penalised logit biases IPW).
"""
import numpy as np


# ---------------------------------------------------------------- bases
def score_basis(r, ok=True):
    return np.column_stack([np.ones_like(r), r, r**2] if ok else [np.ones_like(r), r])


def q_basis(r, s, family, ok=True):
    r, s = np.broadcast_arrays(np.asarray(r, float), np.asarray(s, float))
    cols = [np.ones_like(r), r, s]
    if ok:
        cols += [r**2, s**2, r * s]
        if family == "gauss":
            cols += [r**2 * s**2]
    return np.stack(cols, axis=-1)


def onedim_basis(x, ok=True):
    return np.column_stack([np.ones_like(x), x, x**2, x**3] if ok else [np.ones_like(x), x])


# ---------------------------------------------------------------- generic fitters
def _newton(loglik, grad_hess, b, iters=60, tol=1e-10):
    ll = loglik(b)
    for _ in range(iters):
        g, H = grad_hess(b)
        step = np.linalg.solve(H + 1e-10 * np.eye(len(b)), g)
        t = 1.0
        while True:                                   # backtracking line search
            nb = b + t * step
            nll = loglik(nb)
            if nll >= ll - 1e-12 or t < 1e-8:
                break
            t /= 2
        b, ll = nb, nll
        if np.max(np.abs(t * step)) < tol:
            break
    return b


def logit_mle(Z, y):
    def ll(b):
        eta = Z @ b
        return np.sum(y * eta - np.logaddexp(0, eta))

    def gh(b):
        p = 1 / (1 + np.exp(-(Z @ b)))
        return Z.T @ (y - p), (Z * (p * (1 - p))[:, None]).T @ Z

    return _newton(ll, gh, np.zeros(Z.shape[1]))


def poisson_quasi_mle(X, y, w=None):
    """Poisson (quasi-)MLE with log link; y may be negative (pseudo-outcomes): the
    pseudo-log-likelihood sum w (y*eta - exp(eta)) is concave for any y."""
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(np.average(y, weights=w), 1e-3))

    def ll(b):
        eta = np.clip(X @ b, -30, 30)
        return np.sum(w * (y * eta - np.exp(eta)))

    def gh(b):
        mu = np.exp(np.clip(X @ b, -30, 30))
        return X.T @ (w * (y - mu)), (X * (w * mu)[:, None]).T @ X

    return _newton(ll, gh, b)


def wls(X, y, w=None):
    if w is None:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


# ---------------------------------------------------------------- score models
class Logit:
    """P(policy = 1 | proxy) by unpenalised logit on score_basis."""

    def __init__(self, ok=True, clip=1e-4):
        self.ok, self.clip = ok, clip

    def fit(self, x, d):
        self.coef = logit_mle(score_basis(x, self.ok), d.astype(float))
        return self

    def predict(self, x):
        p = 1 / (1 + np.exp(-(score_basis(x, self.ok) @ self.coef)))
        return np.clip(p, self.clip, 1 - self.clip)

    def design(self, x):
        return score_basis(x, self.ok)


class LGBScore:
    """P(policy = 1 | proxy) by gradient boosting (for ML experiments)."""

    def __init__(self, clip=1e-3, **kw):
        import lightgbm as lgb
        self.clip = clip
        self.m = lgb.LGBMClassifier(verbose=-1, **({"n_estimators": 200, "learning_rate": 0.05,
                                                    "num_leaves": 8, "min_child_samples": 20} | kw))

    def fit(self, x, d):
        self.m.fit(x[:, None], d)
        return self

    def predict(self, x):
        return np.clip(self.m.predict_proba(x[:, None])[:, 1], self.clip, 1 - self.clip)


# ---------------------------------------------------------------- outcome models q(r, s) at a fixed (w,x)
class ParamQ:
    """Graham's proxy-variable regression, fully interacted with the policy cell:
    fitted on dyads with (W_i, X_j) = (w, x). OLS (gauss) or Poisson MLE (poisson)."""

    def __init__(self, family, ok=True):
        self.family, self.ok = family, ok

    def fit(self, r, s, y):
        B = q_basis(r, s, self.family, self.ok)
        self.coef = wls(B, y) if self.family == "gauss" else poisson_quasi_mle(B, y)
        return self

    def predict(self, r, s):
        eta = q_basis(r, s, self.family, self.ok) @ self.coef
        return eta if self.family == "gauss" else np.exp(np.clip(eta, -30, 30))


class LGBQ:
    """Flexible q(r, s) by gradient boosting on (r, s)."""

    def __init__(self, family, **kw):
        import lightgbm as lgb
        base = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15,
                "min_child_samples": 10, "subsample": 0.8, "subsample_freq": 1}
        obj = "regression" if family == "gauss" else "poisson"
        self.m = lgb.LGBMRegressor(objective=obj, verbose=-1, **(base | kw))

    def fit(self, r, s, y):
        self.m.fit(np.column_stack([r, s]), y)
        return self

    def predict(self, r, s):
        shp = np.broadcast(r, s).shape
        rr, ss = np.broadcast_to(r, shp).ravel(), np.broadcast_to(s, shp).ravel()
        return self.m.predict(np.column_stack([rr, ss])).reshape(shp)


# ---------------------------------------------------------------- one-dimensional regressions (MR)
class OneDim:
    """Regression of a pseudo-outcome on ONE agent's proxy (cubic; exp-link for poisson)."""

    def __init__(self, family, ok=True):
        self.family, self.ok = family, ok

    def fit(self, x, y, w=None):
        B = onedim_basis(x, self.ok)
        self.coef = wls(B, y, w) if self.family == "gauss" else poisson_quasi_mle(B, y, w)
        return self

    def predict(self, x):
        eta = onedim_basis(x, self.ok) @ self.coef
        return eta if self.family == "gauss" else np.exp(np.clip(eta, -30, 30))
