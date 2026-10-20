"""Interpret the reverse-engineered marriage pact model.

Answers "how does each variable move the predicted match score?":
  1. Per-feature effects: predicted match-point change when a feature goes
     from its observed min to max, everything else held at typical values.
     Full table saved to feature_effects.csv; top 20 printed.
  2. Signed ridge coefficients in ORIGINAL units: "1 more point of
     disagreement on X -> about Y fewer match points".
  3. Bootstrap 95% CIs, sign stability, and two-sided bootstrap p-values
     for every feature (CI_FEATURES), plus Benjamini-Hochberg FDR q-values
     for the 155-way multiple comparison. (Classical OLS p-values aren't
     valid here: 155 predictors on 30 points.)
  4. What-if curves for the top disagreement features
     (predicted score vs |a-b|), saved to what_if_curves.png.

This interprets the ridge model (the bake-off winner on small data).

Run:  python marriage_pact_interpret.py   (same folder as
        marriage_pact_reverse_engineer.py)
Needs: pip install pandas scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from marriage_pact_reverse_engineer import build_dataset, short

N_EFFECTS_PRINT = 20
N_CURVES = None  # None = all disagreement features; or set a number
N_BOOTSTRAP = 1000
CI_FEATURES = None  # None = CIs for every feature; or a number for top-N only
EFFECTS_CSV = "feature_effects.csv"
CURVES_PNG = "what_if_curves.png"


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        prev = min(prev, p[idx] * n / (i + 1))
        q[idx] = min(prev, 1.0)
    return q


def question_of(feature, questions, qlabels):
    keymap = {short(q): q for q in questions}
    keys = sorted(list(keymap) + list(qlabels), key=len, reverse=True)
    for cand in keys:
        if feature.startswith(cand + "__"):
            return qlabels.get(cand, keymap.get(cand, cand))
    return feature


def main():
    X, y, questions, kinds, qlabels, _pair_info = build_dataset(verbose=False)
    model = make_pipeline(StandardScaler(), Ridge()).fit(X, y)
    print(f"ridge fit on {len(y)} pairs, {X.shape[1]} features")

    # 1. ceteris-paribus effects: min -> max, all else at median
    typical = X.median()
    rows, dropped = [], []
    for c in X.columns:
        lo, hi = X[c].min(), X[c].max()
        if hi == lo:
            dropped.append(c)
            continue
        x_lo = typical.copy()
        x_hi = typical.copy()
        x_lo[c], x_hi[c] = lo, hi
        p_lo = model.predict(pd.DataFrame([x_lo]))[0]
        p_hi = model.predict(pd.DataFrame([x_hi]))[0]
        rows.append({"feature": c, "question": question_of(c, questions, qlabels),
                     "min": lo, "max": hi, "effect": p_hi - p_lo})
    eff = pd.DataFrame(rows)

    # 2. signed coefficients in original units
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    eff["per_unit"] = eff["feature"].map(
        dict(zip(X.columns, ridge.coef_ / scaler.scale_)))

    # 3. univariate correlations for context
    eff["corr"] = eff["feature"].map(
        {c: X[c].corr(y) for c in X.columns if X[c].std() > 0})

    eff["abs_effect"] = eff["effect"].abs()
    eff = eff.sort_values("abs_effect", ascending=False).reset_index(drop=True)

    # Bootstrap 95% CIs + sign stability.
    # (Resamples pairs; pairs sharing a person aren't fully independent,
    # so treat these as approximate.)
    K = len(eff) if CI_FEATURES is None else min(CI_FEATURES, len(eff))
    top_feats = eff["feature"].head(K).tolist()
    eval_rows = []
    for c in top_feats:
        lo, hi = X[c].min(), X[c].max()
        r_lo, r_hi = typical.copy(), typical.copy()
        r_lo[c], r_hi[c] = lo, hi
        eval_rows += [r_lo, r_hi]
    eval_df = pd.DataFrame(eval_rows)
    rng = np.random.default_rng(0)
    boots = {c: [] for c in top_feats}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(X), len(X))
        mb = make_pipeline(StandardScaler(), Ridge()).fit(
            X.iloc[idx], y.iloc[idx])
        preds = mb.predict(eval_df)
        for j, c in enumerate(top_feats):
            boots[c].append(preds[2 * j + 1] - preds[2 * j])
    ci_rows = []
    for c in top_feats:
        b = np.array(boots[c])
        lo95, hi95 = np.percentile(b, [2.5, 97.5])
        obs = eff.loc[eff["feature"] == c, "effect"].iloc[0]
        stable = float(np.mean(np.sign(b) == np.sign(obs)))
        # two-sided bootstrap p-value for H0: effect = 0
        p = 2 * min(np.mean(b > 0), np.mean(b < 0))
        ci_rows.append((c, lo95, hi95, stable, p))
    eff = eff.merge(pd.DataFrame(
        ci_rows, columns=["feature", "ci_lo", "ci_hi", "sign_stable",
                          "p_value"]),
        on="feature", how="left")
    eff["p_bh"] = bh_fdr(eff["p_value"].fillna(1.0))
    eff.to_csv(EFFECTS_CSV, index=False)

    print(f"\ntop {N_EFFECTS_PRINT} effects "
          f"(match-point change from min to max, all else typical;\n"
          f" 95% CI + bootstrap p-value from {N_BOOTSTRAP} bootstraps):")
    for _, r in eff.head(N_EFFECTS_PRINT).iterrows():
        print(f"  {r['effect']:+7.2f} pts [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] "
              f"p={r['p_value']:.3f} q={r['p_bh']:.3f}  {r['feature']:36} "
              f"(1u->{r['per_unit']:+.2f}, r={r['corr']:+.2f})")
    print(f"\nfull table ({len(eff)} features) saved to {EFFECTS_CSV}")
    if dropped:
        print(f"\nskipped {len(dropped)} zero-variance features "
              f"(constant across all pairs -- no effect to measure):")
        for c in dropped:
            print(f"  {c}")

    # 4. what-if curves for the disagreement features
    diff_feats = [c for c in eff["feature"] if c.endswith("__diff")]
    if N_CURVES:
        diff_feats = diff_feats[:N_CURVES]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed: pip install matplotlib "
              "to get the what-if curves plot")
        return

    n = len(diff_feats)
    fig, axes = plt.subplots((n + 1) // 2, 2,
                             figsize=(12, 3 * ((n + 1) // 2)), squeeze=False)
    for ax, feat in zip(axes.flat, diff_feats):
        lo, hi = X[feat].min(), X[feat].max()
        xs = [lo + (hi - lo) * t / 24 for t in range(25)]
        preds = []
        for v in xs:
            row = typical.copy()
            row[feat] = v
            preds.append(model.predict(pd.DataFrame([row]))[0])
        ax.plot(xs, preds, marker="o", markersize=3)
        ax.set_title(question_of(feat, questions, qlabels)[:60], fontsize=9)
        ax.set_xlabel("|a - b| (disagreement)")
        ax.set_ylabel("predicted match")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(CURVES_PNG, dpi=120)
    print(f"what-if curves saved to {CURVES_PNG}")


if __name__ == "__main__":
    main()
