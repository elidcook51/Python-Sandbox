"""Explain individual match scores: which answer differences drive each pair.

For every pair in matches.csv, decomposes the ridge model's predicted score
into per-feature contributions (exact for a linear model: coefficient times
how far this pair is from a typical pair) and reports the top drivers with
the actual answers behind them.

Run:  python marriage_pact_explain.py          (top 3 drivers per pair)
      python marriage_pact_explain.py --n 8    (top 8)
      python marriage_pact_explain.py --all    (every feature)
Needs: pip install pandas scikit-learn
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from marriage_pact_reverse_engineer import build_dataset, short

OUT_CSV = "pair_explanations.csv"


def fmt_val(v):
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (ValueError, TypeError):
        return str(v).strip()


def main():
    ap = argparse.ArgumentParser(description="Explain individual match scores.")
    ap.add_argument("--n", type=int, default=3,
                    help="drivers to show per pair (default 3)")
    ap.add_argument("--all", action="store_true",
                    help="show every feature's contribution per pair")
    args = ap.parse_args()
    n_drivers = 10 ** 9 if args.all else max(1, args.n)

    X, y, questions, kinds, qlabels, pair_info = build_dataset(verbose=True)
    model = make_pipeline(StandardScaler(), Ridge()).fit(X, y)

    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    coef_orig = ridge.coef_ / scaler.scale_
    typical = X.median()
    typical_pred = float(model.predict(pd.DataFrame([typical]))[0])
    contrib = (X - typical) * coef_orig  # rows=pairs, cols=features

    key_to_q = {short(q): q for q in questions}
    all_keys = sorted(list(key_to_q) + list(qlabels), key=len, reverse=True)

    def describe(feat, info):
        # Longest-prefix match, not split("__"): truncated keys used to end
        # in "_" which made "...___diff" parse into a bare "_diff".
        key = next((c for c in all_keys if feat.startswith(c + "__")), None)
        if key is None:
            return feat, ""
        suffix = feat[len(key) + 2:]
        question = qlabels.get(key, key_to_q.get(key, key))
        plain = {
            "care_a": "you care about their identity",
            "care_b": "they care about your identity",
            "a_violates_b": "YOUR identity is on THEIR don't-want list",
            "b_violates_a": "THEIR identity is on YOUR don't-want list",
        }
        if suffix in plain:
            return question, plain[suffix]
        q = key_to_q.get(key)
        if q is None:
            return question, suffix
        ra, rb = info["a"], info["b"]
        if suffix == "diff":
            detail = (f"you={fmt_val(ra[q])}, match={fmt_val(rb[q])}")
        elif suffix == "a":
            detail = f"you={fmt_val(ra[q])}"
        elif suffix == "b":
            detail = f"match={fmt_val(rb[q])}"
        elif suffix == "agree":
            same = str(ra[q]).strip().lower() == str(rb[q]).strip().lower()
            detail = (f"you={fmt_val(ra[q])}, match={fmt_val(rb[q])}, "
                      f"agree={'yes' if same else 'no'}")
        else:
            detail = ""
        return question, detail

    rows = []
    for i, info in enumerate(pair_info):
        pred = float(model.predict(X.iloc[[i]])[0])
        top = contrib.iloc[i].abs().sort_values(ascending=False).head(n_drivers)
        print(f"\n{info['name_a']} -> {info['name_b']}: "
              f"actual {y.iloc[i]:.1f}, predicted {pred:.1f} "
              f"(typical pair: {typical_pred:.1f})")
        row = {"your_name": info["name_a"], "match_name": info["name_b"],
               "actual": round(float(y.iloc[i]), 1),
               "predicted": round(pred, 1),
               "typical": round(typical_pred, 1)}
        for k, (feat, _) in enumerate(top.items(), start=1):
            c = float(contrib.iloc[i][feat])
            question, detail = describe(feat, info)
            print(f"  {c:+6.1f} pts  {question[:58]}"
                  + (f"  ({detail})" if detail else ""))
            row[f"driver{k}_feature"] = feat
            row[f"driver{k}_question"] = question
            row[f"driver{k}_contrib_pts"] = round(c, 2)
            row[f"driver{k}_detail"] = detail
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} "
          f"({len(rows)} pairs; contributions sum to predicted - typical)")


if __name__ == "__main__":
    main()
