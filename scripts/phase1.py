#!/usr/bin/env python3
"""
Phase 1 — Data Quality / Cleaning / Validation / Provenance
Compliments Reference DB Pipeline

Authoritative Input:
  https://huggingface.co/datasets/saraNour/compliments-brand/blob/main/source_of_truth/products.parquet

This phase:
  1. Downloads the authoritative products.parquet from HuggingFace
  2. Validates schema, row count, nulls, duplicates
  3. Performs deterministic cleaning/standardization
  4. Preserves raw values for traceability where cleaning modifies data
  5. Analyzes UPC patterns, brand values, size fields
  6. Documents provenance of every column
  7. Drops 100% null columns with explicit documentation
  8. Produces clean Phase 1 output + validation + statistics

Cleaning operations (Phase 1 = data quality foundation):
  A. String normalization: whitespace trim, empty-to-null
  B. Brand cleaning: whitespace/case normalization, preserve raw
  C. UPC validation: format, nulls, duplicates, reused UPCs
  D. external_id validation: nulls, duplicates
  E. Title cleaning: whitespace normalization
  F. Size fields: consistency audit (semantic parsing is Phase 2)
  G. Null column handling: documented removal of 100% null columns
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_REPO = "saraNour/compliments-brand"
HF_FILE = "source_of_truth/products.parquet"
HF_REPO_TYPE = "dataset"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
VALIDATION_DIR = Path(__file__).resolve().parent.parent / "validation"
STATISTICS_DIR = Path(__file__).resolve().parent.parent / "statistics"

VERSION = "3.0.0"
TIMESTAMP = datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[Phase1] {msg}")


# ===========================================================================
# A. STRING NORMALIZATION
# ===========================================================================

def normalize_whitespace(s):
    """Trim leading/trailing whitespace, collapse repeated internal whitespace."""
    if pd.isna(s):
        return s
    s_str = str(s).strip()
    s_str = re.sub(r"\s+", " ", s_str)
    return s_str if s_str else None


def empty_to_null(s):
    """Convert empty/whitespace-only strings to None."""
    if pd.isna(s):
        return None
    s_str = str(s).strip()
    return None if s_str == "" else s_str


# ===========================================================================
# B. BRAND CLEANING
# ===========================================================================

def clean_brand(raw_brand):
    """
    Clean brand string: trim whitespace, normalize case for downstream.
    Returns (brand_clean, brand_raw).
    """
    if pd.isna(raw_brand):
        return None, None
    raw = str(raw_brand)
    cleaned = raw.strip()
    # Normalize obvious case noise: "COMPLIMENTS" -> "Compliments"
    # But preserve mixed case that might be intentional
    if cleaned.upper() == cleaned and len(cleaned) > 1:
        cleaned = cleaned.title()
    return cleaned, raw


# ===========================================================================
# C. UPC VALIDATION
# ===========================================================================

def validate_upc(upc_val):
    """
    Validate a single UPC value.
    Returns dict with validation results.
    """
    if pd.isna(upc_val):
        return {"valid": False, "reason": "null"}

    s = str(upc_val).strip()

    if s == "" or s == "nan":
        return {"valid": False, "reason": "empty"}

    # Check for non-digit characters (allow decimal point for float representation)
    # UPCs stored as floats may have .0 suffix
    s_clean = s.replace(".0", "") if s.endswith(".0") else s

    if not s_clean.isdigit():
        return {"valid": False, "reason": f"non_digit_chars: {s}"}

    # Check length (standard UPC is 12 digits, but variants exist)
    if len(s_clean) < 6 or len(s_clean) > 14:
        return {"valid": False, "reason": f"unusual_length: {len(s_clean)}"}

    return {"valid": True, "cleaned": s_clean}


def audit_upcs(df):
    """Comprehensive UPC audit."""
    upc = df["upc"]

    # Validate each UPC
    validations = upc.apply(validate_upc)
    valid_mask = validations.apply(lambda x: x["valid"])
    invalid_upcs = df[~valid_mask].copy()

    # Count reused UPCs
    upc_counts = upc.value_counts()
    reused = upc_counts[upc_counts > 1]

    # Analyze reused UPCs: do they map to different titles?
    reused_analysis = []
    for upc_val in reused.index:
        if pd.isna(upc_val):
            continue
        subset = df[df["upc"] == upc_val]
        titles = subset["title"].unique()
        brands = subset["brand"].unique()
        sizes = subset["size"].unique()
        reused_analysis.append({
            "upc": str(upc_val),
            "count": int(reused[upc_val]),
            "unique_titles": len(titles),
            "titles": [str(t) for t in titles[:5]],
            "unique_brands": len(brands),
            "brands": [str(b) for b in brands],
            "unique_sizes": len(sizes),
            "sizes": [str(s) for s in sizes[:5]],
        })

    return {
        "total_rows": len(df),
        "null_count": int(upc.isna().sum()),
        "unique_count": int(upc.nunique()),
        "invalid_format_count": int((~valid_mask).sum()),
        "invalid_format_examples": invalid_upcs[["external_id", "title", "upc"]].head(10).to_dict("records"),
        "reused_upc_count": int(len(reused)),
        "reused_upc_total_rows": int(reused.sum()),
        "reused_upc_examples": reused_analysis[:15],
    }


# ===========================================================================
# D. EXTERNAL_ID VALIDATION
# ===========================================================================

def audit_external_ids(df):
    """Validate external_id field."""
    ext = df["external_id"]
    return {
        "total_rows": len(df),
        "null_count": int(ext.isna().sum()),
        "unique_count": int(ext.nunique()),
        "duplicate_count": int(ext.duplicated().sum()),
        "pass": int(ext.duplicated().sum()) == 0 and ext.isna().sum() == 0,
    }


# ===========================================================================
# E. TITLE CLEANING
# ===========================================================================

def clean_title(raw_title):
    """
    Clean title: trim whitespace, collapse repeated spaces.
    Returns (title_clean, title_raw).
    """
    if pd.isna(raw_title):
        return None, None
    raw = str(raw_title)
    cleaned = raw.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned if cleaned else None, raw


def audit_titles(df):
    """Audit title field quality."""
    titles = df["title"]

    # Check for leading/trailing whitespace
    has_leading = titles.apply(lambda x: str(x) != str(x).strip() if pd.notna(x) else False).sum()
    has_repeated_ws = titles.apply(lambda x: bool(re.search(r"\s{2,}", str(x))) if pd.notna(x) else False).sum()
    empty_titles = titles.isna().sum() + (titles.apply(lambda x: str(x).strip() == "" if pd.notna(x) else False).sum())

    # Duplicated titles
    title_counts = titles.value_counts()
    duplicated_titles = title_counts[title_counts > 1]

    return {
        "total_rows": len(titles),
        "null_count": int(titles.isna().sum()),
        "unique_count": int(titles.nunique()),
        "leading_trailing_whitespace": int(has_leading),
        "repeated_whitespace": int(has_repeated_ws),
        "empty_titles": int(empty_titles),
        "duplicated_title_count": int(len(duplicated_titles)),
        "duplicated_title_total_rows": int(duplicated_titles.sum()),
        "duplicated_title_examples": {str(k): int(v) for k, v in list(duplicated_titles.head(10).items())},
    }


# ===========================================================================
# F. SIZE FIELDS AUDIT
# ===========================================================================

def audit_sizes(df):
    """Audit size-related fields for consistency."""
    size_str = df["size"]
    size_amount = df["size_amount"]
    size_unit = df["size_unit"]
    size_unit_norm = df["size_unit_norm"]
    size_qty = df["size_qty"]

    # Check: when size_amount is null, is size_unit also null?
    amount_null = size_amount.isna()
    unit_null = size_unit.isna()
    both_null = (amount_null & unit_null).sum()
    amount_null_unit_not = (amount_null & ~unit_null).sum()
    unit_null_amount_not = (~amount_null & unit_null).sum()

    # Check: size_qty should usually be 1
    qty_distribution = size_qty.value_counts().to_dict()

    return {
        "size_string": {
            "null_count": int(size_str.isna().sum()),
            "unique_count": int(size_str.nunique()),
        },
        "size_amount": {
            "null_count": int(amount_null.sum()),
            "null_pct": round(amount_null.sum() / len(df) * 100, 2),
        },
        "size_unit": {
            "null_count": int(unit_null.sum()),
            "null_pct": round(unit_null.sum() / len(df) * 100, 2),
            "distribution": {str(k): int(v) for k, v in size_unit.value_counts().items()},
        },
        "size_unit_norm": {
            "null_count": int(size_unit_norm.isna().sum()),
            "distribution": {str(k): int(v) for k, v in size_unit_norm.value_counts().items()},
        },
        "consistency": {
            "both_amount_and_unit_null": int(both_null),
            "amount_null_unit_not_null": int(amount_null_unit_not),
            "unit_null_amount_null_not": int(unit_null_amount_not),
        },
        "size_qty": {
            "distribution": {str(k): int(v) for k, v in qty_distribution.items()},
        },
    }


# ===========================================================================
# G. NULL COLUMN HANDLING
# ===========================================================================

def identify_null_columns(df):
    """Identify columns that are 100% null."""
    null_cols = []
    for col in df.columns:
        if df[col].isna().all():
            null_cols.append(col)
    return null_cols


# ===========================================================================
# H. DUPLICATE AUDIT
# ===========================================================================

def audit_duplicates(df):
    """Comprehensive duplicate audit."""
    full_dupes = int(df.duplicated().sum())
    ext_dupes = int(df["external_id"].duplicated().sum())
    upc_dupes = int(df["upc"].duplicated().sum())

    # Title + size + brand combinations
    if "size" in df.columns:
        combo = df["title"].fillna("") + "|" + df["size"].fillna("") + "|" + df["brand"].fillna("")
        combo_dupes = int(combo.duplicated().sum())
    else:
        combo_dupes = 0

    return {
        "full_row_duplicates": full_dupes,
        "external_id_duplicates": ext_dupes,
        "upc_duplicates": upc_dupes,
        "title_size_brand_duplicates": combo_dupes,
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    log("Starting Phase 1 (v2.0.0 — Data Quality Foundation)")

    # Ensure output dirs exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------------
    log(f"Downloading {HF_REPO}/{HF_FILE} ...")
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type=HF_REPO_TYPE)
    df_raw = pd.read_parquet(path)
    log(f"Loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # Preserve raw copy for provenance
    df_raw_copy = df_raw.copy()

    # ------------------------------------------------------------------
    # 2. SCHEMA VALIDATION
    # ------------------------------------------------------------------
    log("Validating schema ...")
    EXPECTED_COLUMNS = [
        "upc", "external_id", "brand", "title", "price", "price_currency",
        "size", "size_amount", "size_unit", "size_qty", "size_per_unit",
        "size_unit_norm", "size_total", "image_url", "source", "source_url",
    ]
    missing = [c for c in EXPECTED_COLUMNS if c not in df_raw.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"
    assert len(df_raw) == 4440, f"Expected 4440 rows, got {len(df_raw)}"
    log("  Schema: PASS")

    # ------------------------------------------------------------------
    # 3. AUDIT (before cleaning)
    # ------------------------------------------------------------------
    log("Auditing raw data ...")

    null_info = {}
    for col in df_raw.columns:
        n = int(df_raw[col].isna().sum())
        null_info[col] = {
            "null_count": n,
            "null_pct": round(n / len(df_raw) * 100, 2),
            "is_100pct_null": n == len(df_raw),
        }

    dup_info_raw = audit_duplicates(df_raw)
    upc_info_raw = audit_upcs(df_raw)
    ext_info_raw = audit_external_ids(df_raw)
    title_info_raw = audit_titles(df_raw)
    size_info_raw = audit_sizes(df_raw)
    brand_info_raw = {
        "unique_count": int(df_raw["brand"].nunique()),
        "distribution": {str(k): int(v) for k, v in df_raw["brand"].value_counts().items()},
    }

    # ------------------------------------------------------------------
    # 4. CLEANING
    # ------------------------------------------------------------------
    log("Cleaning data ...")
    df = df_raw.copy()

    # A. String normalization on all string columns
    log("  A. Normalizing whitespace on string columns ...")
    string_cols = ["upc", "external_id", "brand", "title", "size",
                   "size_unit", "size_unit_norm", "image_url", "source", "source_url"]
    cleaning_log = {}
    for col in string_cols:
        if col in df.columns:
            before_nulls = int(df[col].isna().sum())
            df[col] = df[col].apply(empty_to_null)
            after_nulls = int(df[col].isna().sum())
            new_nulls = after_nulls - before_nulls
            if new_nulls > 0:
                cleaning_log[col] = {"empty_to_null_count": new_nulls}

    # B. Brand cleaning
    log("  B. Cleaning brand field ...")
    brand_clean_results = df["brand"].apply(clean_brand)
    df["brand_clean"] = brand_clean_results.apply(lambda x: x[0])
    log(f"     Cleaned brands: {df['brand_clean'].nunique()} unique")

    # C. Title cleaning
    log("  C. Cleaning title field ...")
    title_clean_results = df["title"].apply(clean_title)
    df["title_clean"] = title_clean_results.apply(lambda x: x[0])

    # D. Generate source_product_id (composite identifier)
    log("  D. Generating source_product_id ...")
    df["source_product_id"] = df.apply(
        lambda row: f"{row['source']}|{row['upc']}|{row['external_id']}"
        if pd.notna(row['source']) and pd.notna(row['upc']) and pd.notna(row['external_id'])
        else f"{row['source']}|{row['external_id']}",
        axis=1
    )
    log(f"     source_product_id unique: {df['source_product_id'].nunique()}")

    # E. Drop 100% null columns
    null_100pct = [c for c, v in null_info.items() if v["is_100pct_null"]]
    log(f"  F. Dropping 100% null columns: {null_100pct}")
    df = df.drop(columns=null_100pct)

    # ------------------------------------------------------------------
    # 5. REORDER COLUMNS
    # ------------------------------------------------------------------
    log("Reordering columns ...")
    new_order = [
        "source_product_id",
        "upc", "external_id",
        "brand", "brand_clean",
        "title", "title_clean",
        "price", "price_currency",
        "size", "size_amount", "size_unit", "size_qty", "size_unit_norm",
        "image_url", "source", "source_url",
    ]
    df = df[new_order]

    # ------------------------------------------------------------------
    # 6. POST-CLEANING AUDIT
    # ------------------------------------------------------------------
    log("Auditing cleaned data ...")
    dup_info_clean = audit_duplicates(df)
    upc_info_clean = audit_upcs(df)
    ext_info_clean = audit_external_ids(df)
    title_info_clean = audit_titles(df)
    size_info_clean = audit_sizes(df)
    brand_info_clean = {
        "unique_count": int(df["brand_clean"].nunique()),
        "distribution": {str(k): int(v) for k, v in df["brand_clean"].value_counts().items()},
    }

    # ------------------------------------------------------------------
    # 7. STATISTICS
    # ------------------------------------------------------------------
    statistics = {
        "version": VERSION,
        "timestamp": TIMESTAMP,
        "input": {
            "source": f"{HF_REPO}/{HF_FILE}",
            "row_count": len(df_raw),
            "column_count": len(df_raw.columns),
        },
        "output": {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns_dropped": null_100pct,
            "columns_added": ["brand_clean", "title_clean"],
        },
        "cleaning_summary": cleaning_log,
        "nulls_before_cleaning": {c: v for c, v in null_info.items() if v["null_count"] > 0},
        "duplicates_raw": dup_info_raw,
        "duplicates_cleaned": dup_info_clean,
        "upc": upc_info_clean,
        "external_id": ext_info_clean,
        "title": title_info_clean,
        "size": size_info_raw,
        "brand": brand_info_clean,
    }

    # ------------------------------------------------------------------
    # 8. VALIDATION
    # ------------------------------------------------------------------
    all_pass = True
    failures = []

    # external_id must be unique
    if not ext_info_clean["pass"]:
        all_pass = False
        failures.append("external_id_not_unique")

    # No full-row duplicates
    if dup_info_clean["full_row_duplicates"] > 0:
        all_pass = False
        failures.append(f"full_row_duplicates: {dup_info_clean['full_row_duplicates']}")

    # Check no unexpected 100% null columns
    expected_null = {"size_per_unit", "size_total"}
    for col in df.columns:
        if df[col].isna().all() and col not in expected_null:
            all_pass = False
            failures.append(f"unexpected_100pct_null: {col}")

    validation = {
        "version": VERSION,
        "timestamp": TIMESTAMP,
        "result": "PASS" if all_pass else "FAIL",
        "failures": failures,
        "checks": {
            "schema": {"pass": True, "note": "All 16 expected columns present"},
            "external_id_uniqueness": ext_info_clean,
            "duplicates": dup_info_clean,
            "null_columns": {"expected_100pct_null": list(expected_null), "dropped": null_100pct},
        },
    }

    # ------------------------------------------------------------------
    # 9. SAVE OUTPUTS
    # ------------------------------------------------------------------
    log("Saving outputs ...")
    df.to_parquet(OUTPUT_DIR / "phase1_output.parquet", index=False)
    log(f"  Saved phase1_output.parquet ({df.shape[0]} rows, {df.shape[1]} cols)")

    with open(VALIDATION_DIR / "phase1_validation.json", "w") as f:
        json.dump(validation, f, indent=2, default=str)
    log("  Saved phase1_validation.json")

    with open(STATISTICS_DIR / "phase1_statistics.json", "w") as f:
        json.dump(statistics, f, indent=2, default=str)
    log("  Saved phase1_statistics.json")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    log("")
    log("=== PHASE 1 COMPLETE (v2.0.0) ===")
    log(f"Input:  {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    log(f"Output: {df.shape[0]} rows, {df.shape[1]} columns")
    log(f"Columns dropped: {null_100pct}")
    log(f"Columns added: ['brand_clean', 'title_clean']")
    log(f"Validation: {validation['result']}")
    if failures:
        log(f"Failures: {failures}")
    log("========================")

    return df, validation, statistics


if __name__ == "__main__":
    main()
