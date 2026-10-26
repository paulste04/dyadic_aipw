"""
Monte Carlo driver.  Usage:  python run.py E1 [E2 ...] [--reps-scale 0.1] [--jobs 2]

Each experiment is a list of cells; a cell = (DGP kwargs, Spec kwargs, N, reps).
Every replication draws ONE sample that all estimators share (paired comparison), with a
seed derived deterministically from (experiment, cell, N, replication).
Results: results/<EXP>.csv, one row per (replication, estimator, quantity).
"""
import os, sys, time, zlib, json, argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dyadsim import DGP, Spec, analyse

TRUTH_CACHE = {}


def truths(dgp_kw):
    key = json.dumps(dgp_kw, sort_keys=True)
    if key not in TRUTH_CACHE:
        g = DGP(**dgp_kw)
        m11, m01 = g.truth(1, 1), g.truth(0, 1)
        TRUTH_CACHE[key] = {"m11": m11, "m01": m01, "tau": m11 - m01}
    return TRUTH_CACHE[key]


# --------------------------------------------------------------------------- experiment definitions
CELLS_E1 = {  # name: (q_ok, pw_ok, px_ok, pa_ok)
    "all right": (1, 1, 1, 1),
    "q wrong": (0, 1, 1, 1),
    "q, p_x wrong": (0, 1, 0, 1),
    "q, p_w wrong": (0, 0, 1, 1),
    "p_w wrong": (1, 0, 1, 1),
    "p_x wrong": (1, 1, 0, 1),
    "both p wrong": (1, 0, 0, 1),
    "all wrong": (0, 0, 0, 1),
    "pa wrong": (1, 1, 1, 0),
    "q, p_x, pa wrong": (0, 1, 0, 0),
}


def spec_from(fam, c, **kw):
    return dict(family=fam, q_ok=bool(c[0]), pw_ok=bool(c[1]), px_ok=bool(c[2]), pa_ok=bool(c[3]), **kw)


def experiments():
    E = {}
    fams = ("gauss", "poisson")
    E["E1"] = [(dict(family=f), spec_from(f, c), 400, 1000, {"cell": n}) for f in fams for n, c in CELLS_E1.items()]
    E["E2"] = [(dict(family=f), spec_from(f, (1, 1, 1, 1)), N, 1000, {"cell": "all right"})
               for f in fams for N in (100, 200, 400, 800)]
    E["E3"] = [(dict(family=f, **dk), spec_from(f, (1, 1, 1, 1)), 300, 2000, {"cell": lab})
               for f in fams for lab, dk in (("rho=0", {"rho": 0.0}), ("rho=0.6", {"rho": 0.6}),
                                             ("W=X (WTO)", {"wto": True}))]
    E["E4"] = [(dict(family=f), spec_from(f, c), N, 2000 if N < 800 else 1000, {"cell": lab})
               for f in fams for lab, c in (("q wrong, scores right", (0, 1, 1, 1)),
                                            ("q right, p_w wrong", (1, 0, 1, 1)))
               for N in (200, 400, 800)]
    E["E5"] = [(dict(family=f, gamma=g), spec_from(f, (1, 1, 1, 1), trim=tr), 400, 1000,
                {"cell": f"gamma={g}", "trim": "none" if tr is None else f"{tr[0]}:{tr[1]}"})
               for f in fams for g in (1.0, 1.5, 2.5) for tr in (None, ("agent", 0.05), ("dyad", 0.01))]
    E["E7"] = [(dict(family=f, lam_u=l), spec_from(f, (1, 1, 1, 1)), 400, 500, {"cell": f"lambda={l}"})
               for f in fams for l in (0.0, 0.5, 1.0)]
    lgb = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 5}
    E["E6"] = [(dict(family=f), spec_from(f, (1, 1, 1, 1), q_learner="lgb", crossfit=cf, lgb_kw=lgb), N, reps,
                {"cell": {None: "no cross-fitting", "dyad": "split by dyads", "agent": "split by agents"}[cf]})
               for f in fams for N, reps in ((300, 400), (600, 200)) for cf in (None, "dyad", "agent")]
    return E


# --------------------------------------------------------------------------- one replication
def one_rep(exp, ci, dgp_kw, spec_kw, N, rep, meta):
    seed = [zlib.crc32(exp.encode()), ci, N, rep]
    rng = np.random.default_rng(seed)
    g = DGP(**dgp_kw)
    d = g.draw(N, rng)
    d["wto"] = g.wto
    out = analyse(d, Spec(**spec_kw), rng=rng)
    tr = truths(dgp_kw)
    for r in out:
        r.update(exp=exp, family=g.family, N=N, rep=rep, truth=tr[r["quantity"]], **meta)
    return out


def run(exp, scale=1.0, jobs=2, family=None, tag="", N=None, batch=50):
    """Resumable: results are written every `batch` replications, and replications already in the
    output file (same cell label, family, N, rep) are skipped on a restart.  Seeds depend only on
    (exp, cell index, N, rep), so a resumed run gives exactly the same numbers as an uninterrupted one."""
    cells = [c for c in experiments()[exp] if (family is None or c[0].get("family") == family) and (N is None or c[2] == N)]
    path = f"results/{exp}{tag}.csv"
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    rows, t0 = old.to_dict("records"), time.time()
    for ci, (dgp_kw, spec_kw, N, reps, meta) in enumerate(cells):
        R = max(int(reps * scale), 5)
        done = set()
        if len(old):
            mask = (old["cell"] == meta["cell"]) & (old["family"] == dgp_kw["family"]) & (old["N"] == N)
            done = set(old.loc[mask, "rep"].unique())
        todo = [r for r in range(R) if r not in done]
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            res = Parallel(n_jobs=jobs)(delayed(one_rep)(exp, ci, dgp_kw, spec_kw, N, r, meta) for r in chunk)
            for rr in res:
                rows.extend(rr)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"[{exp}] cell {ci+1}/{len(cells)} {meta} {dgp_kw.get('family')} N={N} "
                  f"reps {len(done) + b + len(chunk)}/{R} elapsed {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--reps-scale", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--family", default=None)
    ap.add_argument("--N", type=int, default=None)
    a = ap.parse_args()
    for e in a.exps:
        tag = ("" if a.family is None else "_" + a.family) + ("" if a.N is None else f"_N{a.N}")
        run(e, a.reps_scale, a.jobs, a.family, tag, a.N)
