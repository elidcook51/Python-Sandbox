"""Guess which names refer to the same person across the marriage pact CSVs.

Usage:
    python marriage_pact_name_match.py

Expects 'answers.csv' and 'matches.csv' in the same folder (edit the
constants below if yours are named differently). Writes 'name_guesses.csv'
with one row per distinct raw name and the best-guess canonical name.
"""

import csv
import difflib
import re
import unicodedata
from collections import defaultdict

ANSWERS_CSV = "answers.csv"
MATCHES_CSV = "matches.csv"
OUT_CSV = "name_guesses.csv"

# Anything at or above this difflib ratio counts as a probable match.
THRESHOLD = 0.80
# How many candidates to list per name in the console report.
TOP_N = 3


def normalize(name):
    name = unicodedata.normalize("NFKD", name.strip())
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"[^a-z ]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def find_col(headers, *keywords):
    lowered = [h.strip().lower() for h in headers]
    for kw in keywords:
        for i, h in enumerate(lowered):
            if kw in h:
                return headers[i]
    return None


def read_names(path, *col_keywords):
    """Return {raw_name: count} for the given name column(s)."""
    counts = defaultdict(int)
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = [find_col(reader.fieldnames, kw) for kw in col_keywords]
        cols = [c for c in cols if c]
        if not cols:
            raise ValueError(f"no matching column in {path}: {reader.fieldnames}")
        for row in reader:
            for c in cols:
                raw = (row.get(c) or "").strip()
                if raw:
                    counts[raw] += 1
    return counts


def score_pair(a_norm, b_norm):
    if a_norm == b_norm:
        return 1.0, "exact"
    a_tok, b_tok = a_norm.split(), b_norm.split()
    if a_tok and b_tok:
        # "elid" vs "Eli Cook", any caps: first-name-only against full name
        if len(a_tok) == 1 and a_tok[0] == b_tok[0]:
            return 0.95, "first-name-subset"
        if len(b_tok) == 1 and b_tok[0] == a_tok[0]:
            return 0.95, "first-name-subset"
    ratio = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio, "fuzzy"


def main():
    answer_counts = read_names(ANSWERS_CSV, "name")
    match_counts = read_names(MATCHES_CSV, "your name", "match nam")

    # Canonical pool: prefer the most common spelling of each normalized name.
    canon = {}
    for raw in answer_counts:
        norm = normalize(raw)
        if norm not in canon or answer_counts[raw] > answer_counts[canon[norm]]:
            canon[norm] = raw
    canon_norms = list(canon)

    rows = []
    for raw in sorted(match_counts, key=str.lower):
        norm = normalize(raw)
        scored = []
        for c_norm in canon_norms:
            s, rule = score_pair(norm, c_norm)
            scored.append((s, rule, canon[c_norm]))
        scored.sort(reverse=True)
        best_score, best_rule, best_name = scored[0] if scored else (0, "-", "")
        rows.append({
            "raw_name": raw,
            "times_seen": match_counts[raw],
            "normalized": norm,
            "best_guess": best_name if best_score >= THRESHOLD else "",
            "score": round(best_score, 3),
            "rule": best_rule if best_score >= THRESHOLD else "no-confident-match",
        })
        top = [(round(s, 2), n) for s, _, n in scored[:TOP_N] if s >= THRESHOLD]
        print(f"{raw!r:30} -> {top}")

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_matched = sum(1 for r in rows if r["best_guess"])
    print(f"\n{len(rows)} distinct names in matches.csv, "
          f"{n_matched} matched to answers.csv at >= {THRESHOLD}.")
    print(f"Wrote {OUT_CSV} - review the 'best_guess' column before joining.")


if __name__ == "__main__":
    main()
