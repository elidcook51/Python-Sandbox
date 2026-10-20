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

import difflib
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

# Dealbreaker columns for religion / political identity. Set explicitly below
# (verified against the user's actual headers 2026-10-19); set to None to
# fall back to auto-detect by keywords (what it found is printed every run).
RELIGION_SELF = "What religious identity"
RELIGION_CARE = "Do you care about your partners religious identity"
RELIGION_AVOID = "What religious identity do you not want to be with?"
POLITICS_SELF = "Political Identity"
POLITICS_CARE = "Do you care about your partners political identity"
POLITICS_AVOID = "What political identity do you not want to be with"


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
    s = re.sub(r"\W+", "_", str(text).strip().lower()).strip("_")
    return s[:n].rstrip("_")


def is_likert(series):
    s = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
    return bool(s.notna().all() and s.between(1, 7).all())


def detect_dealbreaker_cols(questions):
    """Locate (self, care, avoid) columns for religion and politics.

    Returns {"religion": (self, care, avoid), "politics": (self, care, avoid)};
    any element may be None if not found.
    """
    def find(domain_key, role):
        avoid_keys = ["not want", "don't want", "dont want", "avoid",
                      "dealbreaker", "unacceptable", "rule out"]
        out = []
        for q in questions:
            ql = q.lower()
            if domain_key not in ql:
                continue
            is_avoid = any(k in ql for k in avoid_keys)
            is_care = "care" in ql
            if role == "avoid" and not is_avoid:
                continue
            if role == "care" and not is_care:
                continue
            if role == "self" and (is_avoid or is_care):
                continue
            out.append(q)
        return out[0] if out else None

    cfg = {
        "religion": (RELIGION_SELF, RELIGION_CARE, RELIGION_AVOID),
        "politics": (POLITICS_SELF, POLITICS_CARE, POLITICS_AVOID),
    }
    cols = {}
    for domain, (s, c, a) in cfg.items():
        key = "relig" if domain == "religion" else "polit"
        triple = (s or find(key, "self"),
                  c or find(key, "care"),
                  a or find(key, "avoid"))
        resolved = []
        for val, role in zip(triple, ("self", "care", "avoid")):
            r = resolve_col(val, questions)
            if val and not r:
                print(f"warning: configured {domain} {role} column {val!r} "
                      f"not found; falling back to auto-detect")
                r = resolve_col(find(key, role), questions)
            resolved.append(r)
        cols[domain] = tuple(resolved)
    return cols


def resolve_col(col, questions):
    """Match a column name against the actual headers, forgiving case/?/spaces."""
    if col is None:
        return None
    if col in questions:
        return col
    norm = {q.strip().lower().rstrip("?"): q for q in questions}
    return norm.get(col.strip().lower().rstrip("?"))


AVOID_EMPTY = {"", "nan", "none", "n/a", "no preference", "no preferences"}


def parse_avoid_list(val):
    return {p.strip().lower() for p in re.split(r"\s*(?:[;,]|\band\b)\s*", str(val))
            if p.strip().lower() not in AVOID_EMPTY}


def identity_matches(ident, avoid_set):
    """Does this identity appear on the other's don't-want list?"""
    ident = normalize(str(ident))
    if not ident or not avoid_set or ident in AVOID_EMPTY:
        return False
    if ident in avoid_set:
        return True
    no_paren = re.sub(r"\s*\(.*?\)\s*", "", ident).strip()
    for a in avoid_set:
        a_np = re.sub(r"\s*\(.*?\)\s*", "", a).strip()
        if no_paren and no_paren == a_np:
            return True
        if difflib.SequenceMatcher(None, ident, a).ratio() >= 0.88:
            return True
    return False


def care_value(val):
    s = str(val).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return 1
    if s in ("no", "n", "false", "0"):
        return 0
    try:
        return float(s)  # likert 1-7 kept raw
    except ValueError:
        return 0


def dealbreaker_features(ra, rb, db_cols):
    """care_a, care_b + violation flags, for religion and politics."""
    feats = {}
    for prefix, (self_c, care_c, avoid_c) in db_cols.items():
        if care_c:
            feats[f"{prefix}__care_a"] = care_value(ra[care_c])
            feats[f"{prefix}__care_b"] = care_value(rb[care_c])
        if self_c and avoid_c:
            feats[f"{prefix}__a_violates_b"] = int(
                identity_matches(ra[self_c], parse_avoid_list(rb[avoid_c])))
            feats[f"{prefix}__b_violates_a"] = int(
                identity_matches(rb[self_c], parse_avoid_list(ra[avoid_c])))
    return feats


def dealbreaker_feature_names(db_cols):
    names = []
    for prefix, (self_c, care_c, avoid_c) in db_cols.items():
        if care_c:
            names += [f"{prefix}__care_a", f"{prefix}__care_b"]
        if self_c and avoid_c:
            names += [f"{prefix}__a_violates_b", f"{prefix}__b_violates_a"]
    return names


def build_dataset(verbose=True):
    """Load answers.csv + matches.csv and build the pair-level feature matrix.

    Returns (X, y, questions, kinds).
    """
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
            bg = r.get("best_guess")
            # empty best_guess reads back as NaN; str(NaN)="nan" is truthy,
            # so it must be excluded explicitly or it poisons the lookup
            if bg is not None and not pd.isna(bg) and str(bg).strip():
                overrides[normalize(r["raw_name"])] = normalize(bg)
    except FileNotFoundError:
        pass

    # First-name fallback: matches.csv often uses first names only, and first
    # names are unique in this data. Mirrors the loosened rules in
    # marriage_pact_name_match.py (kept local so this script runs standalone).
    NICKNAMES = {
        "mike": "michael", "liz": "elizabeth", "beth": "elizabeth",
        "bill": "william", "jim": "james", "bob": "robert",
        "dave": "david", "kate": "katherine", "dick": "richard",
        "peggy": "margaret", "sally": "sarah", "chuck": "charles",
        "hank": "henry", "ned": "edward", "ted": "theodore",
    }
    FIRST_NAME_FLOOR = 0.70

    def first_name_score(fa, fb):
        if fa == fb:
            return 0.95
        if NICKNAMES.get(fa) == fb or NICKNAMES.get(fb) == fa:
            return 0.90
        if (len(fa) >= 3 and len(fb) >= 3
                and (fa.startswith(fb) or fb.startswith(fa))):
            return 0.90
        r = difflib.SequenceMatcher(None, fa, fb).ratio()
        return r if r >= 0.60 else 0.0

    people_by_first = {}
    for norm_name in people:
        tok = norm_name.split()
        if tok:
            people_by_first.setdefault(tok[0], []).append(norm_name)

    _lookup_cache = {}
    first_name_resolutions = []

    def lookup(raw):
        name = normalize(raw)
        if name in _lookup_cache:
            return _lookup_cache[name]
        target = overrides.get(name, name)
        result = people.get(target)
        if result is None and target == name:
            toks = name.split()
            if toks:
                f = toks[0]
                cands = []
                for pf, fulls in people_by_first.items():
                    s = first_name_score(f, pf)
                    if s >= FIRST_NAME_FLOOR:
                        cands.extend((full, s) for full in fulls)
                cands.sort(key=lambda t: -t[1])
                if cands and (len(cands) == 1 or cands[0][1] > cands[1][1]):
                    full, s = cands[0]
                    first_name_resolutions.append((str(raw), full, s))
                    result = people[full]
        _lookup_cache[name] = result
        return result

    kinds = {q: ("likert" if is_likert(answers[q]) else "categorical")
             for q in questions}
    n_unique = {q: answers[q].astype(str).str.strip().str.lower().nunique()
                for q in questions if kinds[q] == "categorical"}
    n_likert = sum(k == "likert" for k in kinds.values())
    if verbose:
        print(f"questions: {len(questions)} "
              f"({n_likert} likert 1-7, {len(questions) - n_likert} categorical)")

    db_cols = detect_dealbreaker_cols(questions)
    db_skip = set()
    for prefix, (self_c, care_c, avoid_c) in db_cols.items():
        db_skip.update(c for c in (care_c, avoid_c) if c)
        if verbose and any((self_c, care_c, avoid_c)):
            print(f"dealbreaker [{prefix}]: self={self_c!r} "
                  f"care={care_c!r} avoid={avoid_c!r}")

    def pair_features(ra, rb):
        feats = {}
        for q in questions:
            if q in db_skip:
                continue
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
        feats.update(dealbreaker_features(ra, rb, db_cols))
        return feats

    X_rows, y, skipped_names, bad_targets, pair_info = [], [], [], 0, []
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
            target = float("nan")
        if pd.isna(target):  # covers "N/A"/blank cells, which pandas reads as NaN
            bad_targets += 1
            continue
        X_rows.append(pair_features(ra, rb))
        y.append(target)
        pair_info.append({"name_a": str(raw_a), "name_b": str(raw_b),
                          "a": ra, "b": rb})

    X = pd.DataFrame(X_rows).fillna(0)
    y = pd.Series(y)
    if verbose:
        dropped_names = len(matches) - len(y) - bad_targets
        print(f"matches.csv rows: {len(matches)} -> pairs used: {len(y)} "
              f"(dropped {len(matches) - len(y)}: {dropped_names} unresolved "
              f"name lookups, {bad_targets} bad percents), "
              f"features: {X.shape[1]}")
        for n, c in Counter(skipped_names).most_common(10):
            print(f"  no answers row for: {n!r} (x{c})")
        if skipped_names:
            print("  (run marriage_pact_name_match.py and approve/fill "
                  "name_guesses.csv to resolve these)")
        seen_res = {}
        for raw, full, s in first_name_resolutions:
            seen_res.setdefault(raw, (full, s))
        if seen_res:
            print(f"resolved {len(seen_res)} name(s) by first name:")
            for raw, (full, s) in sorted(seen_res.items()):
                print(f"  {raw!r} -> {full!r} ({s:.2f})")
        for prefix in ("religion", "politics"):
            vcols = [c for c in X.columns
                     if c.startswith(prefix + "__") and "violates" in c]
            if vcols:
                n_viol = int((X[vcols].sum(axis=1) > 0).sum())
                print(f"pairs with a {prefix} dealbreaker violation: {n_viol}")
    if len(y) < 2:
        raise SystemExit("not enough matched pairs to train on")
    qlabels = {}
    if any(db_cols["religion"]):
        qlabels["religion"] = "Religion (identity + dealbreakers)"
    if any(db_cols["politics"]):
        qlabels["politics"] = "Political identity (dealbreakers)"
    return X, y, questions, kinds, qlabels, pair_info


def main():
    X, y, questions, kinds, qlabels, _pair_info = build_dataset(verbose=True)

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
    top_feats = []
    if imp is not None:
        top_feats = list(
            imp.sort_values(ascending=False).head(N_TOP_FEATURES).index)
        print(imp.sort_values(ascending=False).head(N_TOP_FEATURES).to_string())

    print("\nquestion text for top features:")
    keymap = {short(q): q for q in questions}
    all_keys = sorted(list(keymap) + list(qlabels), key=len, reverse=True)
    seen = set()
    for feat in top_feats:
        key = next((c for c in all_keys if feat.startswith(c + "__")), feat)
        if key not in seen:
            seen.add(key)
            full = qlabels.get(key, keymap.get(key, key))
            print(f"  {key:42} = {full}")

    print("\nunivariate correlation with match score (top 10 by |r|):")
    corr = pd.Series({c: X[c].corr(y) for c in X.columns if X[c].std() > 0})
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(10)
    print(top.to_string())


if __name__ == "__main__":
    main()
