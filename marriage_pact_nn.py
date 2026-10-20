"""Neural network reverse-engineering the marriage pact match score.

Inputs:  person 1's questionnaire answers + person 2's answers.
         Likert 1-7 responses enter as numbers scaled to [0, 1];
         categorical answers enter one-hot.
Output:  predicted match score in [0, 1] (match percent / 100).

Run:  python marriage_pact_nn.py
Needs: pip install pandas scikit-learn
"""

import pickle
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from marriage_pact_reverse_engineer import (
    dealbreaker_feature_names, dealbreaker_features, detect_dealbreaker_cols)

ANSWERS_CSV = "answers.csv"
MATCHES_CSV = "matches.csv"
GUESSES_CSV = "name_guesses.csv"  # optional, from marriage_pact_name_match.py
MODEL_OUT = "marriage_pact_nn.pkl"

HIDDEN_LAYERS = (16,)
ACTIVATION = "tanh"
SOLVER = "lbfgs"  # best for hundreds-few thousand pairs; use "adam" at 10k+
ALPHA = 0.1
MAX_ITER = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 0


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


def load_people(answers, name_col, questions):
    frames = []
    for q in questions:
        raw = answers[q].astype(str).str.strip()
        if is_likert(raw):
            frames.append((((pd.to_numeric(raw) - 1) / 6).rename(
                f"num__{short(q)}")))
        else:
            frames.append(pd.get_dummies(raw.str.lower(),
                                         prefix=f"cat__{short(q)}",
                                         dtype=float))
    enc = pd.concat(frames, axis=1)
    n_likert = sum(c.startswith("num__") for c in enc.columns)
    print(f"encoded answers: {enc.shape[1]} inputs/person "
          f"({n_likert} likert, {enc.shape[1] - n_likert} categorical)")
    people = {}
    for i, row in answers.iterrows():
        people.setdefault(normalize(row[name_col]), enc.iloc[i].values)
    return people, list(enc.columns)


def load_overrides():
    overrides = {}
    try:
        guesses = pd.read_csv(GUESSES_CSV, encoding="utf-8-sig")
        for _, r in guesses.iterrows():
            if str(r.get("best_guess", "")).strip():
                overrides[normalize(r["raw_name"])] = normalize(r["best_guess"])
    except FileNotFoundError:
        pass
    return overrides


def main():
    answers = pd.read_csv(ANSWERS_CSV, encoding="utf-8-sig")
    matches = pd.read_csv(MATCHES_CSV, encoding="utf-8-sig")

    name_col = find_col(answers.columns, "name")
    your_col = find_col(matches.columns, "your name")
    their_col = find_col(matches.columns, "match nam")
    target_col = find_col(matches.columns, "match perc")
    assert all([name_col, your_col, their_col, target_col]), (
        f"missing columns: {list(answers.columns)} vs {list(matches.columns)}")

    questions = [c for c in answers.columns
                 if c != name_col and "timestamp" not in c.lower()]
    people, columns = load_people(answers, name_col, questions)
    overrides = load_overrides()
    lookup = lambda raw: people.get(overrides.get(normalize(raw),
                                                  normalize(raw)))
    people_raw = {}
    for _, row in answers.iterrows():
        people_raw.setdefault(normalize(row[name_col]), row)
    lookup_raw = lambda raw: people_raw.get(overrides.get(normalize(raw),
                                                           normalize(raw)))
    db_cols = detect_dealbreaker_cols(questions)
    db_names = dealbreaker_feature_names(db_cols)
    columns = columns + db_names
    if db_names:
        print(f"dealbreaker features: {db_names}")

    X, y, skipped = [], [], 0
    for _, m in matches.iterrows():
        va, vb = lookup(m[your_col]), lookup(m[their_col])
        if va is None or vb is None:
            skipped += 1
            continue
        try:
            pct = float(str(m[target_col]).replace("%", "").strip())
        except ValueError:
            skipped += 1
            continue
        db = dealbreaker_features(lookup_raw(m[your_col]),
                                 lookup_raw(m[their_col]), db_cols)
        db_vec = np.array([db[k] for k in db_names], dtype=float)
        X.append(np.concatenate([va, vb, db_vec]))
        y.append(min(max(pct / 100.0, 0.0), 1.0))

    X, y = np.array(X), np.array(y)
    print(f"pairs: {len(y)} (skipped {skipped}), "
          f"inputs per pair: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS, activation=ACTIVATION,
                     solver=SOLVER, alpha=ALPHA, max_iter=MAX_ITER,
                     random_state=RANDOM_STATE))
    model.fit(X_train, y_train)

    pred = np.clip(model.predict(X_test), 0, 1)
    r2 = model.score(X_test, y_test)
    mae_pp = np.mean(np.abs(pred - y_test)) * 100
    print(f"test R^2={r2:.3f}  MAE={mae_pp:.2f} percentage points")
    print("\nsample predictions (predicted -> actual):")
    for p, t in zip(pred[:10], y_test[:10]):
        print(f"  {p:.3f} -> {t:.3f}")

    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"model": model, "columns": columns,
                     "questions": questions}, f)
    print(f"\nsaved trained network to {MODEL_OUT}")


if __name__ == "__main__":
    main()
