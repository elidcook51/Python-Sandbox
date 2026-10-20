"""Reverse-engineer the marriage pact match algorithm.

Joins answers.csv (one row per person) with matches.csv (one row per pair),
builds pair-level features from BOTH people's answers, and trains regressors
to predict the match percentage. The winning model's feature importances hint
at what the real algorithm weights most. Likert 1-7 questions contribute each
person's score plus |a-b|; categorical questions contribute agreement
(+ one-hots unless COMPACT_FEATURES).

Run:  python marriage_pact_reverse_engineer.py
Needs: pip install pandas scikit-learn
"""

import re
import unicodedata
from collections import Counter

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ANSWERS_CSV = "answers.csv"
MATCHES_CSV = "matches.csv"
GUESSES_CSV = "name_guesses.csv"  # optional, from marriage_pact_name_match.py
TARGET_KEYWORD = "match perc"
MAX_CATEGORIES = 6  # questions with more distinct answers get agree-only
COMPACT_FEATURES = True  # categoricals get agree-only (no one-hots); best for small n
N_TOP_FEATURES = 20


def normalize(name):
    name = unicodedata.normalize("NFKD", str(name).strip())
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", name.lower()).strip()


def find_col(cols, *keywords):
    lowered = {c.strip().lower(): c for c in cols}
    for kw in keywords:
        for low, orig in lowered.items():
            if kw in low:
                return orig
    return None


def short(text, n=28):
    return re.sub(r"\W+", "_", str(text).strip().lower()).strip("_")[:n]


def is_likert(series):
    s = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
    return bool(s.notna().all() and s.between(1, 7).all())


def main():
    answers = pd.read_csv(ANSWERS_CSV, encoding="utf-8-sig")
    matches = pd.read_csv(MATCHES_CSV, encoding="utf-8-sig")

    name_col = find_col(answers.columns, "name")
    your_col = find_col(matches.columns, "your name")
    their_col = find_col(matches.columns, "match nam")
    target_col = find_col(matches.columns, TARGET_KEYWORD)
    assert all([name_col, your_col, their_col, target_col]), (
        f"missing columns: {list(answers.columns)} vs {list(matches.columns)}")

    questions = [c for c in answers.columns
                 if c != name_col and "timestamp" not in c.lower()]

    people = {}
    for _, row in answers.iterrows():
        people.setdefault(normalize(row[name_col]), row)

    # Manual name overrides from the name-matching step, if it exists.
    overrides = {}
    try:
        guesses = pd.read_csv(GUESSES_CSV, encoding="utf-8-sig")
        for _, r in guesses.iterrows():
            if str(r.get("best_guess", "")).strip():
                overrides[normalize(r["raw_name"])] = normalize(r["best_guess"])
    except FileNotFoundError:
        pass

    def lookup(raw):
        return people.get(overrides.get(normalize(raw), normalize(raw)))

    kinds = {q: ("likert" if is_likert(answers[q]) else "categorical")
             for q in questions}
    n_unique = {q: answers[q].astype(str).str.strip().str.lower().nunique()
                for q in questions if kinds[q] == "categorical"}
    n_likert = sum(k == "likert" for k in kinds.values())
    print(f"questions: {len(questions)} "
          f"({n_likert} likert 1-7, {len(questions) - n_likert} categorical)")

    def pair_features(ra, rb):
        feats = {}
        for q in questions:
            k = short(q)
            if kinds[q] == "likert":
                va = float(str(ra[q]).strip())
                vb = float(str(rb[q]).strip())
                feats[f"{k}__a"] = va
                feats[f"{k}__b"] = vb
                feats[f"{k}__diff"] = abs(va - vb)
            else:
                va = str(ra[q]).strip().lower()
                vb = str(rb[q]).strip().lower()
                feats[f"{k}__agree"] = int(va == vb)
                if not COMPACT_FEATURES and n_unique[q] <= MAX_CATEGORIES:
                    feats[f"{k}__a_{short(va, 18)}"] = 1
                    feats[f"{k}__b_{short(vb, 18)}"] = 1
        return feats

    X_rows, y, skipped_names = [], [], []
    for _, m in matches.iterrows():
        raw_a, raw_b = m[your_col], m[their_col]
        ra, rb = lookup(raw_a), lookup(raw_b)
        if ra is None or rb is None:
            missing = [str(n) for n, r in ((raw_a, ra), (raw_b, rb))
                       if r is None]
            skipped_names.extend(missing)
            continue
        try:
            target = float(str(m[target_col]).replace("%", "").strip())
        except ValueError:
            skipped += 1
            continue
        X_rows.append(pair_features(ra, rb))
        y.append(target)

    X = pd.DataFrame(X_rows).fillna(0)
    y = pd.Series(y)
    print(f"pairs used: {len(y)} (skipped {len(skipped_names)} name lookups), "
          f"features: {X.shape[1]}")
    for n, c in Counter(skipped_names).most_common(10):
        print(f"  no answers row for: {n!r} (x{c})")
    if len(y) < 2:
        raise SystemExit("not enough matched pairs to train on")

    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge()),
        "random_forest": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=0),
        "grad_boost": HistGradientBoostingRegressor(random_state=0),
        "mlp": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0)),
    }
    results = {}
    small_n = len(y) < 50
    cv_split = LeaveOneOut() if small_n else min(5, len(y))
    # R^2 is undefined for single-sample LOO folds, so small-n uses MAE only.
    scoring = ("neg_mean_absolute_error",) if small_n \
        else ("r2", "neg_mean_absolute_error")
    print(f"CV: {'leave-one-out' if small_n else '5-fold'}")
    for name, model in models.items():
        cv = cross_validate(model, X, y, cv=cv_split, scoring=scoring,
                            n_jobs=-1)
        mae = -cv["test_neg_mean_absolute_error"].mean()
        if small_n:
            results[name] = -mae
            print(f"{name:14} MAE={mae:.2f}")
        else:
            r2 = cv["test_r2"].mean()
            results[name] = r2
            print(f"{name:14} R^2={r2:.3f}  MAE={mae:.2f}")

    best = max(results, key=results.get)
    print(f"\nbest: {best} -> refit on all data for importances")
    final = models[best].fit(X, y)
    est = final[-1] if hasattr(final, "__getitem__") else final
    if hasattr(est, "feature_importances_"):
        imp = pd.Series(est.feature_importances_, index=X.columns)
    elif hasattr(est, "coef_"):
        imp = pd.Series(abs(est.coef_.ravel()), index=X.columns)
    else:
        imp = None
    if imp is not None:
        print(imp.sort_values(ascending=False).head(N_TOP_FEATURES).to_string())

    print("\nunivariate correlation with match score (top 10 by |r|):")
    corr = X.apply(lambda s: s.corr(y)).fillna(0)
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(10)
    print(top.to_string())


if __name__ == "__main__":
    main()
