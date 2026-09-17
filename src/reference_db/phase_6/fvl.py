from __future__ import annotations

import pandas as pd


def add_fvl_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add FVL provenance fields.

    Product-level FVL percentage is not available in the current
    Reference DB nutrition source. Therefore V1 does not invent an
    FVL percentage from taxonomy.

    FVL will be populated in a later version when a defensible
    ingredient-based source/calculation is available.
    """

    result = df.copy()

    result["fvl_percent"] = pd.NA
    result["fvl_method"] = "not_available"
    result["fvl_confidence"] = pd.NA
    result["fvl_provenance"] = "NOT_AVAILABLE"

    return result
