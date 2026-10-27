"""
Summaries, LaTeX tables and figures from results/*.csv.   python analyze.py
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

OUT = "results"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

EST = ["reg", "ht", "hajek", "aipw", "mr"]
LABEL = {"reg": "Proxy regression", "ht": "HT", "hajek": "Hájek", "aipw": "Naive AIPW", "mr": "MR AIPW"}
COL = {"reg": "#2a78d6", "ht": "#eb6834", "hajek": "#1baf7a", "aipw": "#eda100", "mr": "#e87ba4"}
MRK = {"reg": "o", "ht": "s", "hajek": "^", "aipw": "D", "mr": "v"}
FAM = {"gauss": "Gaussian outcome", "poisson": "Poisson gravity outcome"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2, "savefig.dpi": 200})


def load(e):
    p = os.path.join(OUT, f"{e}.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    parts = sorted(glob.glob(os.path.join(OUT, f"{e}_*.csv")))   # runs split by --family / --N
    return pd.concat([pd.read_csv(q) for q in parts], ignore_index=True) if parts else None


def se_col(df):
    """Dyadic SE; score-corrected for HT/Hajek levels (Thms 2-3)."""
    se = df["se_dyadic"].copy()
    if "se_dyadic_corr" in df:
        se = df["se_dyadic_corr"].where(df["se_dyadic_corr"].notna(), se)
    return se


def summarise(df, keys, se_name=None):
    df = df.copy()
    df["err"] = df["est"] - df["truth"]
    df["se"] = se_col(df) if se_name is None else df[se_name]
    df["cover"] = (df["err"].abs() <= 1.96 * df["se"]).astype(float)
    g = df.groupby(keys)
    s = g.agg(bias=("err", "mean"), sd=("est", "std"), rmse=("err", lambda x: np.sqrt(np.mean(x**2))),
              mean_se=("se", "mean"), coverage=("cover", "mean"), reps=("rep", "count")).reset_index()
    s["bias_mcse"] = s["sd"] / np.sqrt(s["reps"])
    s["se_ratio"] = s["mean_se"] / s["sd"]
    return s


def fmt_cov(c, est):
    return "--" if est == "reg" else f"{c:.3f}"


# =========================================================== E1: misspecification grid
def e1():
    df = load("E1")
    if df is None:
        return
    order = list(dict.fromkeys(df["cell"]))
    s = summarise(df, ["family", "cell", "estimator", "quantity"])
    s.to_csv(os.path.join(OUT, "summary_E1.csv"), index=False)
    for q in ("m11", "tau"):
        lines = [r"\begin{tabular}{@{}ll" + "r" * len(EST) + "@{}}", r"\toprule",
                 "& & " + " & ".join(LABEL[e] for e in EST) + r"\\", r"\midrule"]
        for fam in ("gauss", "poisson"):
            lines.append(r"\multicolumn{%d}{@{}l}{\emph{%s}}\\" % (2 + len(EST), FAM[fam]))
            for c in order:
                row = s[(s.family == fam) & (s.cell == c) & (s.quantity == q)].set_index("estimator")
                b = " & ".join(f"{row.loc[e,'bias']:+.3f}" for e in EST)
                cv = " & ".join(fmt_cov(row.loc[e, 'coverage'], e) for e in EST)
                lines.append(f"{c} & bias & {b}\\\\")
                lines.append(f" & cover & {cv}\\\\")
            lines.append(r"\midrule")
        lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}")
        open(os.path.join(OUT, f"table_E1_{q}.tex"), "w").write("\n".join(lines))

    # figure: |bias| / MC sd by cell (dot plot), tau
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True)
    for ax, fam in zip(axes, ("gauss", "poisson")):
        for k, e in enumerate(EST):
            sub = s[(s.family == fam) & (s.quantity == "tau") & (s.estimator == e)].set_index("cell").loc[order]
            y = np.arange(len(order)) + (k - 2) * 0.12
            ax.scatter(sub["bias"] / sub["sd"], y, s=26, marker=MRK[e], color=COL[e], label=LABEL[e],
                       edgecolor="white", linewidth=0.6, zorder=3)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(np.arange(len(order)), order)
        ax.invert_yaxis()
        ax.set_title(FAM[fam], color=INK, loc="left", fontsize=9)
        ax.set_xlabel("bias / sampling sd  (effect $\\tau$)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_E1_bias.pdf")); fig.savefig(os.path.join(FIG, "fig_E1_bias.png"))
    plt.close(fig)


# =========================================================== E2: rates and efficiency
def e2():
    df = load("E2")
    if df is None:
        return
    s = summarise(df, ["family", "N", "estimator", "quantity"])
    s["rootN_sd"] = np.sqrt(s["N"]) * s["sd"]
    # robust spread (IQR/1.349): one Poisson MR draw at N=100 has error -185 and swamps the sd
    iqr = (df.assign(err=df["est"] - df["truth"]).groupby(["family", "N", "estimator", "quantity"])["err"]
           .agg(lambda x: (x.quantile(.75) - x.quantile(.25)) / 1.349).rename("iqr_sd").reset_index())
    s = s.merge(iqr, on=["family", "N", "estimator", "quantity"])
    s["rootN_iqr_sd"] = np.sqrt(s["N"]) * s["iqr_sd"]
    s.to_csv(os.path.join(OUT, "summary_E2.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, fam in zip(axes, ("gauss", "poisson")):
        for e in EST:
            sub = s[(s.family == fam) & (s.quantity == "tau") & (s.estimator == e)].sort_values("N")
            ax.plot(sub["N"], sub["rootN_iqr_sd"], marker=MRK[e], ms=6, color=COL[e], label=LABEL[e],
                    mec="white", mew=0.6)
        ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks([100, 200, 400, 800], ["100", "200", "400", "800"])
        ax.set_xlabel("number of agents N"); ax.set_ylabel(r"$\sqrt{N}\times$ robust sd  (effect $\tau$)")
        ax.set_title(FAM[fam], color=INK, loc="left", fontsize=9)
        ax.set_ylim(bottom=0)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_E2_rates.pdf")); fig.savefig(os.path.join(FIG, "fig_E2_rates.png"))
    plt.close(fig)


# =========================================================== E3: SE recipes
def e3():
    df = load("E3")
    if df is None:
        return
    recipes = [("se_iid", "i.i.d. dyads"), ("se_sender", "cluster by sender"),
               ("se_twoway", "two-way clustering"), ("se_dyadic", "dyadic")]
    rows = []
    for col, lab in recipes:
        for est in ("hajek", "aipw", "mr"):
            sub = df[(df.estimator == est) & (df.quantity == "m11")]
            if col == "se_dyadic" and est == "hajek":
                s = summarise(sub, ["family", "cell"], "se_dyadic_corr")
            else:
                s = summarise(sub, ["family", "cell"], col)
            s["recipe"], s["estimator"] = lab, est
            rows.append(s)
    s = pd.concat(rows)
    s.to_csv(os.path.join(OUT, "summary_E3.csv"), index=False)
    cells = list(dict.fromkeys(df["cell"]))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    rec_col = {"i.i.d. dyads": "#2a78d6", "cluster by sender": "#eb6834",
               "two-way clustering": "#1baf7a", "dyadic": "#eda100"}
    rec_mrk = {"i.i.d. dyads": "o", "cluster by sender": "s", "two-way clustering": "^", "dyadic": "D"}
    for ax, fam in zip(axes, ("gauss", "poisson")):
        for k, (_, lab) in enumerate(recipes):
            sub = s[(s.family == fam) & (s.estimator == "aipw") & (s.recipe == lab)].set_index("cell").loc[cells]
            x = np.arange(len(cells)) + (k - 1.5) * 0.15
            ax.scatter(x, sub["coverage"], s=30, marker=rec_mrk[lab], color=rec_col[lab], label=lab,
                       edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(0.95, color=MUTED, lw=1, ls="--")
        ax.set_xticks(np.arange(len(cells)), cells)
        ax.set_ylim(0, 1.02); ax.set_ylabel("coverage of 95% CI (AIPW, $m(1,1)$)")
        ax.set_title(FAM[fam], color=INK, loc="left", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_E3_se.pdf")); fig.savefig(os.path.join(FIG, "fig_E3_se.png"))
    plt.close(fig)


# =========================================================== E4: robust inference
def e4():
    df = load("E4")
    if df is None:
        return
    s = summarise(df, ["family", "cell", "N", "estimator", "quantity"])
    s.to_csv(os.path.join(OUT, "summary_E4.csv"), index=False)
    cells = list(dict.fromkeys(df["cell"]))
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6), sharex=True, sharey=True)
    for i, fam in enumerate(("gauss", "poisson")):
        for j, c in enumerate(cells):
            ax = axes[i, j]
            for e in ("aipw", "mr"):
                sub = s[(s.family == fam) & (s.cell == c) & (s.estimator == e) & (s.quantity == "tau")].sort_values("N")
                ax.plot(sub["N"], sub["coverage"], marker=MRK[e], ms=6, color=COL[e], label=LABEL[e],
                        mec="white", mew=0.6)
            ax.axhline(0.95, color=MUTED, lw=1, ls="--")
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks([200, 400, 800], ["200", "400", "800"])
            ax.set_ylim(0.8, 1.0)
            ax.set_title(f"{FAM[fam]}: {c}", color=INK, loc="left", fontsize=8.5)
            if i == 1: ax.set_xlabel("number of agents N")
            if j == 0: ax.set_ylabel("coverage (effect $\\tau$)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_E4_robust_inference.pdf")); fig.savefig(os.path.join(FIG, "fig_E4_robust_inference.png"))
    plt.close(fig)


# =========================================================== E5, E7: tables
def e5():
    df = load("E5")
    if df is None:
        return
    s = summarise(df, ["family", "cell", "trim", "estimator", "quantity"])
    s.to_csv(os.path.join(OUT, "summary_E5.csv"), index=False)


def e7():
    df = load("E7")
    if df is None:
        return
    s = summarise(df, ["family", "cell", "estimator", "quantity"])
    s.to_csv(os.path.join(OUT, "summary_E7.csv"), index=False)


# =========================================================== E6: ML and cross-fitting
def e6():
    df = load("E6")
    if df is None:
        return
    s = summarise(df, ["family", "N", "cell", "estimator", "quantity"])
    s.to_csv(os.path.join(OUT, "summary_E6.csv"), index=False)
    cells = ["no cross-fitting", "split by dyads", "split by agents"]
    Ns = sorted(df["N"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, fam in zip(axes, ("gauss", "poisson")):
        for k, (e, N) in enumerate([(e, N) for e in ("aipw", "mr") for N in Ns]):
            sub = s[(s.family == fam) & (s.estimator == e) & (s.N == N) & (s.quantity == "tau")].set_index("cell")
            sub = sub.reindex(cells)
            x = np.arange(len(cells)) + (k - 1.5) * 0.14
            ax.scatter(x, sub["coverage"], s=30, marker=MRK[e] if N == Ns[0] else ("P" if e == "aipw" else "X"),
                       color=COL[e], label=f"{LABEL[e]}, N={N}", edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(0.95, color=MUTED, lw=1, ls="--")
        ax.set_xticks(np.arange(len(cells)), cells)
        ax.set_ylim(0.6, 1.0); ax.set_ylabel("coverage (effect $\\tau$)")
        ax.set_title(FAM[fam], color=INK, loc="left", fontsize=9)
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_E6_crossfit.pdf")); fig.savefig(os.path.join(FIG, "fig_E6_crossfit.png"))
    plt.close(fig)


if __name__ == "__main__":
    for f in (e1, e2, e3, e4, e5, e6, e7):
        f()
    print("done")
