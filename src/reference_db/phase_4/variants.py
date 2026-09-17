from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def parse_variant_attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    return {}


def build_variant_key(row: dict[str, Any]) -> str:
    attributes = parse_variant_attributes(row.get("variant_attributes"))

    amount = attributes.get("amount")
    unit = attributes.get("unit")
    qty = attributes.get("qty")
    count = attributes.get("count")
    multiplier = attributes.get("multiplier")
    raw = attributes.get("raw")

    parts = []

    if count is not None:
        parts.append(f"count{count}")
    elif amount is not None:
        parts.append(f"amt{amount}")
        if unit:
            parts.append(f"unit{unit}")
        if qty is not None and qty != 1:
            parts.append(f"qty{qty}")
        if multiplier is not None and multiplier != 1:
            parts.append(f"mult{multiplier}")
    elif raw:
        parts.append(f"raw{str(raw).strip().lower()}")
    else:
        parts.append("nosize")

    return "|".join(parts)


def generate_variant_id(group_id: str, variant_key: str) -> str:
    value = f"{group_id}||{variant_key}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def assign_variants(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["variant_key"] = result.apply(
        lambda row: build_variant_key(row.to_dict()),
        axis=1,
    )

    result["variant_id"] = result.apply(
        lambda row: generate_variant_id(
            str(row["group_id"]),
            row["variant_key"],
        ),
        axis=1,
    )

    return result


def build_variant_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for variant_id, group in df.groupby("variant_id", sort=False):
        row = group.iloc[0]

        rows.append(
            {
                "variant_id": variant_id,
                "group_id": row["group_id"],
                "variant_key": row["variant_key"],
                "core_title": row.get("core_title"),
                "size": row.get("size"),
                "product_count": len(group),
            }
        )

    return pd.DataFrame(rows)


def build_variant_mapping(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "external_id",
        "group_id",
        "variant_id",
        "variant_key",
    ]

    return df[columns].copy()


def build_variant_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for group_id, group in df.groupby("group_id", sort=False):
        rows.append(
            {
                "group_id": group_id,
                "variant_count": group["variant_id"].nunique(),
                "product_count": len(group),
                "variant_ids": ", ".join(
                    sorted(group["variant_id"].unique())
                ),
            }
        )

    return pd.DataFrame(rows)
