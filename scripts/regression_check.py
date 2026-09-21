#!/usr/bin/env python3
"""Full regression: classify all phase_2 products, compare to prev PASS set."""
import pandas as pd
import pickle
import sys
from multiprocessing import Pool, cpu_count

sys.path.insert(0, "/home/sara/reference-db-system/src")
from reference_db.classification.classifier import classify_product

P2 = "/home/sara/reference-db-system/data/phase_2/standardized_products.parquet"
OUT = "/tmp/phase3_backup/after_all_classified.pkl"

def classify_one(name):
    try:
        r = classify_product(name)
        return {
            "taxonomy": r.taxonomy,
            "tax_status": r.taxonomy_status,
            "tax_check": r.taxonomy_resolution if r.taxonomy_resolution else r.taxonomy_rule,
            "domain": r.domain,
            "domain_conf": r.domain_confidence,
            "domain_rule": r.domain_rule,
            "tax_rule": r.taxonomy_rule,
        }
    except Exception as e:
        return {"taxonomy": None, "tax_status": f"ERROR:{e}", "tax_check": None,
                "domain": None, "domain_conf": 0, "domain_rule": None, "tax_rule": None}

def main():
    df = pd.read_parquet(P2)
    print("phase2:", df.shape, flush=True)
    names = df["product_name"].tolist()
    with Pool(cpu_count()) as pool:
        results = pool.map(classify_one, names)

    out = pd.DataFrame(results)
    out["external_id"] = df["external_id"].astype(str)
    print("saving", out.shape, flush=True)
    out.to_pickle(OUT)

    prev = pickle.load(open("/tmp/phase3_backup/prev_pass.pkl", "rb"))
    new_pass = set(out.loc[out["tax_status"] == "PASS", "external_id"])
    dropped = prev - new_pass
    print("new PASS:", len(new_pass), "prev:", len(prev), "dropped:", len(dropped), flush=True)
    if dropped:
        sub = df[df["external_id"].astype(str).isin(dropped)]
        for _, r in sub.iterrows():
            row = out[out["external_id"] == str(r["external_id"])].iloc[0]
            print("  DROPPED:", r["product_name"], "->", row["tax_status"], row["tax_check"], flush=True)
    print("status counts:", out["tax_status"].value_counts().to_dict(), flush=True)
    print("domain counts:", out["domain"].value_counts().to_dict(), flush=True)

if __name__ == "__main__":
    main()
