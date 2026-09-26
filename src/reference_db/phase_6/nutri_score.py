from __future__ import annotations

import pandas as pd

ALGORITHM_VERSION = "Nutri-Score-2023"


# Nutri-Score 2023 — General Foods
#
# Source:
# Santé publique France, updated Nutri-Score algorithm (2023)
#
# All values are per 100 g for solid foods.


# Negative component thresholds.
#
# Each tuple is:
#   (upper_threshold_inclusive, points)
#
# Values above the final threshold receive the maximum number of points.

ENERGY_THRESHOLDS_KJ = [
    (335, 0),
    (670, 1),
    (1005, 2),
    (1340, 3),
    (1675, 4),
    (2010, 5),
    (2345, 6),
    (2680, 7),
    (3015, 8),
    (3350, 9),
]

SUGAR_THRESHOLDS_G = [
    (3.4, 0),
    (6.8, 1),
    (10.0, 2),
    (14.0, 3),
    (17.0, 4),
    (20.0, 5),
    (24.0, 6),
    (27.0, 7),
    (31.0, 8),
    (34.0, 9),
    (37.0, 10),
    (41.0, 11),
    (44.0, 12),
    (48.0, 13),
    (51.0, 14),
]

SATURATED_FAT_THRESHOLDS_G = [
    (1.0, 0),
    (2.0, 1),
    (3.0, 2),
    (4.0, 3),
    (5.0, 4),
    (6.0, 5),
    (7.0, 6),
    (8.0, 7),
    (9.0, 8),
    (10.0, 9),
]

SALT_THRESHOLDS_G = [
    (0.2, 0),
    (0.4, 1),
    (0.6, 2),
    (0.8, 3),
    (1.0, 4),
    (1.2, 5),
    (1.4, 6),
    (1.6, 7),
    (1.8, 8),
    (2.0, 9),
    (2.2, 10),
    (2.4, 11),
    (2.6, 12),
    (2.8, 13),
    (3.0, 14),
    (3.2, 15),
    (3.4, 16),
    (3.6, 17),
    (3.8, 18),
    (4.0, 19),
]


# Positive component thresholds.

FIBRE_THRESHOLDS_G = [
    (3.0, 0),
    (4.1, 1),
    (5.2, 2),
    (6.3, 3),
    (7.4, 4),
]

PROTEIN_THRESHOLDS_G = [
    (2.4, 0),
    (4.8, 1),
    (7.2, 2),
    (9.6, 3),
    (12.0, 4),
    (14.0, 5),
    (17.0, 6),
]


def _threshold_points(
    value: float,
    thresholds: list[tuple[float, int]],
    maximum_points: int,
) -> int:
    """
    Apply Nutri-Score threshold tables.

    Boundary behavior follows the official tables:
    a value exactly equal to a threshold remains in that point band;
    points increase only when the value is above the threshold.
    """

    if pd.isna(value):
        return 0

    value = float(value)

    points = 0

    for threshold, threshold_points in thresholds:
        if value > threshold:
            points = threshold_points + 1
        else:
            break

    return min(points, maximum_points)


def calculate_energy_kj(calories_kcal) -> float | None:
    """Convert kcal to kJ."""

    if pd.isna(calories_kcal):
        return None

    return float(calories_kcal) * 4.184


def calculate_salt_g_from_sodium(sodium_mg) -> float | None:
    """
    Convert sodium to salt.

    Nutri-Score 2023 uses salt rather than sodium.
    Molecular conversion:
        salt = sodium × 2.5
    """

    if pd.isna(sodium_mg):
        return None

    return float(sodium_mg) / 1000.0 * 2.5


def calculate_negative_points(row: pd.Series) -> dict[str, int | None]:
    """Calculate N-component points for general foods."""

    energy_kj = row.get("energy_kj_per_100g")
    sugars = row.get("sugars_g_per_100g")
    saturated_fat = row.get("saturated_fat_g_per_100g")
    salt = row.get("salt_g_per_100g")

    if any(
        pd.isna(value)
        for value in [
            energy_kj,
            sugars,
            saturated_fat,
            salt,
        ]
    ):
        return {
            "energy_points": None,
            "sugar_points": None,
            "saturated_fat_points": None,
            "salt_points": None,
            "negative_points": None,
        }

    energy_points = _threshold_points(
        energy_kj,
        ENERGY_THRESHOLDS_KJ,
        maximum_points=10,
    )

    sugar_points = _threshold_points(
        sugars,
        SUGAR_THRESHOLDS_G,
        maximum_points=15,
    )

    saturated_fat_points = _threshold_points(
        saturated_fat,
        SATURATED_FAT_THRESHOLDS_G,
        maximum_points=10,
    )

    salt_points = _threshold_points(
        salt,
        SALT_THRESHOLDS_G,
        maximum_points=20,
    )

    negative_points = (
        energy_points
        + sugar_points
        + saturated_fat_points
        + salt_points
    )

    return {
        "energy_points": energy_points,
        "sugar_points": sugar_points,
        "saturated_fat_points": saturated_fat_points,
        "salt_points": salt_points,
        "negative_points": negative_points,
    }


def calculate_fibre_points(fibre_g) -> int | None:
    if pd.isna(fibre_g):
        return None

    return _threshold_points(
        fibre_g,
        FIBRE_THRESHOLDS_G,
        maximum_points=5,
    )


def calculate_protein_points(protein_g) -> int | None:
    if pd.isna(protein_g):
        return None

    return _threshold_points(
        protein_g,
        PROTEIN_THRESHOLDS_G,
        maximum_points=7,
    )


def calculate_fvl_points(fvl_percent) -> int | None:
    """
    Calculate FVL points.

    Updated 2023 general-food scale:
        <= 40% : 0
        > 40%  : 1
        > 60%  : 2
        > 80%  : 5

    FVL cannot be inferred from taxonomy alone.
    """

    if pd.isna(fvl_percent):
        return None

    value = float(fvl_percent)

    if value <= 40:
        return 0

    if value <= 60:
        return 1

    if value <= 80:
        return 2

    return 5


def calculate_positive_points(
    row: pd.Series,
    *,
    protein_cap: int | None = None,
) -> dict[str, int | None]:

    fibre_points = calculate_fibre_points(
        row.get("fibre_g_per_100g")
    )

    protein_points = calculate_protein_points(
        row.get("protein_g_per_100g")
    )

    fvl_points = calculate_fvl_points(
        row.get("fvl_percent")
    )

    if protein_points is not None and protein_cap is not None:
        protein_points = min(
            protein_points,
            protein_cap,
        )

    if (
        fibre_points is None
        or protein_points is None
        or fvl_points is None
    ):
        positive_points = None
    else:
        positive_points = (
            fibre_points
            + protein_points
            + fvl_points
        )

    return {
        "protein_points": protein_points,
        "fibre_points": fibre_points,
        "fvl_points": fvl_points,
        "positive_points": positive_points,
    }


def calculate_general_score(
    negative_points: int,
    fibre_points: int,
    fvl_points: int,
    protein_points: int,
) -> int:
    """
    Updated Nutri-Score general-food score.

    If N < 11:
        score = N - P

    If N >= 11:
        protein points are excluded and:
        score = N - fibre points - FVL points
    """

    if negative_points < 11:
        return (
            negative_points
            - fibre_points
            - protein_points
            - fvl_points
        )

    return (
        negative_points
        - fibre_points
        - fvl_points
    )


def grade_general_score(score: int) -> str:
    """
    General foods / cheese / red meat grading.

    A: score <= 0
    B: 1–2
    C: 3–10
    D: 11–18
    E: >= 19
    """

    if score <= 0:
        return "A"

    if score <= 2:
        return "B"

    if score <= 10:
        return "C"

    if score <= 18:
        return "D"

    return "E"


def calculate_general_food_components(
    row: pd.Series,
) -> dict[str, object]:

    negative = calculate_negative_points(row)

    positive = calculate_positive_points(row)

    result = {
        **negative,
        **positive,
        "nutri_score_raw": None,
        "nutri_score_grade": None,
        "nutri_score_calculated": False,
        "nutri_score_status": "FVL_NOT_AVAILABLE",
        "nutri_score_algorithm": ALGORITHM_VERSION,
    }

    if negative["negative_points"] is None:
        result["nutri_score_status"] = "MISSING_REQUIRED_INPUT"
        return result

    if positive["positive_points"] is None:
        return result

    score = calculate_general_score(
        negative_points=negative["negative_points"],
        fibre_points=positive["fibre_points"],
        fvl_points=positive["fvl_points"],
        protein_points=positive["protein_points"],
    )

    result["nutri_score_raw"] = score
    result["nutri_score_grade"] = grade_general_score(score)
    result["nutri_score_calculated"] = True
    result["nutri_score_status"] = "CALCULATED"

    return result


def add_nutri_score_components(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add Nutri-Score 2023 component columns.

    Important:
    - We calculate all available components.
    - Final score requires FVL.
    - Because current Reference DB has no ingredient/FVL data,
      current eligible rows remain FVL_NOT_AVAILABLE.
    """

    result = df.copy()

    result["energy_kj_per_100g"] = result[
        "calories_per_100g"
    ].apply(calculate_energy_kj)

    result["salt_g_per_100g"] = result[
        "sodium_mg_per_100g"
    ].apply(calculate_salt_g_from_sodium)

    # Initialize output columns.
    component_columns = [
        "energy_points",
        "sugar_points",
        "saturated_fat_points",
        "salt_points",
        "negative_points",
        "protein_points",
        "fibre_points",
        "fvl_points",
        "positive_points",
        "nutri_score_raw",
        "nutri_score_grade",
        "nutri_score_calculated",
        "nutri_score_status",
        "nutri_score_algorithm",
    ]

    for column in component_columns:
        result[column] = None

    for index, row in result.iterrows():

        if not bool(row.get("score_eligibility", False)):
            result.at[
                index,
                "nutri_score_status"
            ] = "NOT_ELIGIBLE"

            result.at[
                index,
                "nutri_score_calculated"
            ] = False

            result.at[
                index,
                "nutri_score_algorithm"
            ] = ALGORITHM_VERSION

            continue

        calculated = calculate_general_food_components(
            result.loc[index]
        )

        for column, value in calculated.items():
            result.at[index, column] = value

    return result
