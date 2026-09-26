#!/usr/bin/env python3
"""Classify all phase_2 products with the NEW rules, compare to prev PASS set.

Writes results incrementally to /tmp so we can poll. Deterministic, no LLM.
"""
import pickle
import sys
import time
from multiprocessing import Pool, cpu_count

import pandas as pd

sys.path.insert(0, "/home/sara/reference-db-system/src")

t0 = time.time()
P2 = "/home/sara/reference-db-system/data/phase_2/standardized_products.parquet"
PREV_PASS = "/tmp/phase3_backup/prev_pass.pkl"
OUT = "/tmp/phase3_backup/after_full_pass_ids.pkl"
STREAM = "/tmp/phase3_backup/after_classify_stream.parquet"

from reference_db.classification.classifier import classify_product


def classify_one(name: str) -> dict:
    try:
        r = classify_product(name)
        return {
            "taxonomy": r.taxonomy,
            "taxonomy_status": r.taxonomy_status,
            "taxonomy_rule": r.taxonomy_rule,
            "taxonomy_confidence": r.taxonomy_confidence,
            "domain": r.domain,
            "domain_status": r.domain_status,
            "domain_rule": r.domain_rule,
            "domain_confidence": r.domain_confidence,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "taxonomy": None,
            "taxonomy_status": f"ERROR:{type(e).__name__}:{e}",
            "taxonomy_rule": None,
            "taxonomy_confidence": None,
            "domain": None,
            "domain_status": "ERROR",
            "domain_rule": None,
            "domain_confidence": None,
        }


def main() -> None:
    print("loading phase_2 ...", flush=True)
    df = pd.read_parquet(P2, columns=["external_id", "product_name"])
    names = df["product_name"].tolist()
    print(f"phase_2: {len(names)} rows", flush=True)

    with Pool(cpu_count()) as pool:
        rows = list(
            pool.imap(classify_one, names, chunksize=64)
        )
    out = pd.DataFrame(rows)
    out["external_id"] = df["external_id"].astype(str).values
    out.to_parquet(STREAM, index=False)

    with open(PREV_PASS, "rb") as f:
        prev = pickle.load(f)
    new_pass = set(out.loc[out["taxonomy_status"] == "PASS", "external_id"])
    print(f"prev PASS: {len(prev)}  new PASS: {len(new_pass)}", flush=True)
    print(f"prev still PASS (all): {prev.issubset(new_pass)}", flush=True)
    dropped = prev - new_pass
    print(f"dropped: {len(dropped)}", flush=True)
    for eid in sorted(dropped):
        row = df[df["external_id"].astype(str) == eid].iloc[0]
        got = out[out["external_id"] == eid].iloc[0]
        print(f"  DROPPED {row['product_name'][:60]!r} -> {got['taxonomy_status']} {got['taxonomy_rule']}", flush=True)
    print("taxonomy_status counts:", out["taxonomy_status"].value_counts().to_dict(), flush=True)
    print("domain counts:", out["domain"].value_counts().to_dict(), flush=True)
    with open(OUT, "wb") as f:
        pickle.dump(new_pass, f)
    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
