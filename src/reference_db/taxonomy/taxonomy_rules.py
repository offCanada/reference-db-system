from __future__ import annotations

import re
from dataclasses import dataclass

from reference_db.taxonomy.reference_taxonomy import ReferenceCategory


@dataclass(frozen=True)
class TaxonomyMatch:
    category: ReferenceCategory | None
    confidence: float
    matched_rule: str | None
    status: str


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


DISAMBIGUATION_RULES = [
    (
        r"\bpaper\s+bag\b.*\b(?:potato|potatoes|apple|banana|bananas|orange|carrot|onion|garlic)\b",
        ReferenceCategory.PRODUCE,
        0.95,
        "paper_bag_food_product",
    ),
    (
        r"\bice\s+cream\s+cups?\b",
        None,
        0.50,
        "ice_cream_cups_ambiguous",
    ),
    (
        r"\b(?:dog|cat)\s+(?:food|treat|treats|dental|dental\s+sticks?)\b",
        ReferenceCategory.PET_FOOD,
        0.95,
        "pet_food",
    ),
    (
        r"\b(?:bird\s+seed|bird\s+feed|wild\s+bird\s+seed)\b",
        ReferenceCategory.PET_FOOD,
        0.95,
        "bird_food",
    ),
    (
        r"\b(?:castor\s+oil|furniture\s+polish|disinfectant)\b",
        ReferenceCategory.HOUSEHOLD_CLEANING,
        0.95,
        "non_food_oil_or_cleaning_product",
    ),
    (
        r"\bwater\s+bottle\b",
        ReferenceCategory.HOUSEHOLD_SUPPLIES,
        0.95,
        "water_bottle",
    ),
    (
        r"\b(?:coffee\s+filters?|cone\s+coffee\s+filters?)\b",
        ReferenceCategory.HOUSEHOLD_SUPPLIES,
        0.95,
        "coffee_filters",
    ),
    (
        r"\bchristmas\s+crackers?\b",
        ReferenceCategory.HOUSEHOLD_SUPPLIES,
        0.90,
        "christmas_crackers",
    ),
    (
        r"\bwhitening\s+wraps?\b",
        ReferenceCategory.PERSONAL_CARE,
        0.90,
        "whitening_wraps",
    ),
    (
        r"\bflour\s+sack\s+towels?\b",
        ReferenceCategory.HOUSEHOLD_SUPPLIES,
        0.95,
        "flour_sack_towels",
    ),
    (
        r"\bdenture\s+cleanser\b",
        ReferenceCategory.PERSONAL_CARE,
        0.95,
        "denture_cleanser",
    ),
    (
        r"\bwater\s+enhancer\b",
        ReferenceCategory.BEVERAGES,
        0.90,
        "water_enhancer",
    ),
    (
        r"\bgrilling\s+plank\b",
        ReferenceCategory.HOUSEHOLD_SUPPLIES,
        0.90,
        "grilling_plank",
    ),
    (
        r"\bsnack\s+bags?\b",
        ReferenceCategory.HOUSEHOLD_SUPPLIES,
        0.95,
        "snack_bags",
    ),
    (
        r"\bgingerbread\s+house\s+kit\b",
        ReferenceCategory.HOUSEHOLD_SUPPLIES,
        0.80,
        "gingerbread_house_kit",
    ),
    (
        r"\b(?:vitamin|vitamins|supplement|supplements)\b.*\b(?:capsules?|tablets?)\b",
        ReferenceCategory.HEALTH_REMEDIES,
        0.95,
        "health_supplement",
    ),
]


CATEGORY_RULES = [
    (
        ReferenceCategory.DAIRY,
        [
            r"\bmilk\b",
            r"\bcheese\b",
            r"\byogurt\b",
            r"\byoghurt\b",
            r"\bbutter\b",
            r"\bsour\s+cream\b",
            r"\bcottage\s+cheese\b",
            r"\bcream\s+cheese\b",
        ],
    ),
    (
        ReferenceCategory.MEAT_SEAFOOD,
        [
            r"\bbeef\b",
            r"\bpork\b",
            r"\bchicken\b",
            r"\bturkey\b",
            r"\blamb\b",
            r"\bveal\b",
            r"\bsausage\b",
            r"\bbacon\b",
            r"\bham\b",
            r"\bsalami\b",
            r"\bpepperoni\b",
            r"\bsalmon\b",
            r"\btuna\b",
            r"\bshrimp\b",
            r"\bprawn\b",
            r"\bcrab\b",
            r"\blobster\b",
            r"\bseafood\b",
        ],
    ),
    (
        ReferenceCategory.ALCOHOLIC_BEVERAGES,
        [
            r"\bbeer\b",
            r"\blager\b",
            r"\bale\b",
            r"\bwine\b",
            r"\bchampagne\b",
            r"\bprosecco\b",
            r"\bcider\b",
            r"\bvodka\b",
            r"\bwhisky\b",
            r"\bwhiskey\b",
            r"\brum\b",
            r"\bgin\b",
            r"\btequila\b",
            r"\bbrandy\b",
            r"\bliqueur\b",
        ],
    ),
    (
        ReferenceCategory.BEVERAGES,
        [
            r"\bjuice\b",
            r"\bdrink\b",
            r"\bbeverage\b",
            r"\bsoda\b",
            r"\bpop\b",
            r"\bcola\b",
            r"\bcoffee\b",
            r"\btea\b",
            r"\blemonade\b",
            r"\bsmoothie\b",
            r"\benergy\s+drink\b",
            r"\bsports\s+drink\b",
        ],
    ),
    (
        ReferenceCategory.BAKERY,
        [
            r"\bbread\b",
            r"\bbuns?\b",
            r"\bbagel\b",
            r"\bcroissant\b",
            r"\bmuffin\b",
            r"\bcake\b",
            r"\bcookies?\b",
            r"\bpastr(?:y|ies)\b",
            r"\bdonuts?\b",
            r"\bdoughnuts?\b",
            r"\btortilla\b",
            r"\bflatbread\b",
            r"\bnaan\b",
        ],
    ),
    (
        ReferenceCategory.BAKING_INGREDIENTS,
        [
            r"\bflour\b",
            r"\bbaking\s+powder\b",
            r"\bbaking\s+soda\b",
            r"\byeast\b",
            r"\bcake\s+mix\b",
            r"\bicing\s+sugar\b",
            r"\bpowdered\s+sugar\b",
            r"\bvanilla\s+extract\b",
            r"\bcocoa\s+powder\b",
            r"\bcornstarch\b",
        ],
    ),
    (
        ReferenceCategory.FROZEN,
        [
            r"\bfrozen\b",
            r"\bice\s+cream\b",
            r"\bfrozen\s+pizza\b",
            r"\bfrozen\s+vegetables?\b",
            r"\bfrozen\s+fruit\b",
            r"\bfrozen\s+meal\b",
            r"\bfrozen\s+fries\b",
        ],
    ),
    (
        ReferenceCategory.PRODUCE,
        [
            r"\bapples?\b",
            r"\bbananas?\b",
            r"\boranges?\b",
            r"\blemons?\b",
            r"\blimes?\b",
            r"\bgrapes?\b",
            r"\bstrawberries?\b",
            r"\bblueberries?\b",
            r"\braspberries?\b",
            r"\bpotatoes?\b",
            r"\btomatoes?\b",
            r"\bonions?\b",
            r"\bgarlic\b",
            r"\bcarrots?\b",
            r"\blettuce\b",
            r"\bspinach\b",
            r"\bbroccoli\b",
            r"\bcucumbers?\b",
            r"\bpeppers?\b",
            r"\bmushrooms?\b",
        ],
    ),
    (
        ReferenceCategory.CONDIMENTS_SAUCES,
        [
            r"\bsauce\b",
            r"\bsalsa\b",
            r"\bketchup\b",
            r"\bmustard\b",
            r"\bmayonnaise\b",
            r"\bmayo\b",
            r"\bdressing\b",
            r"\brelish\b",
            r"\bdip\b",
            r"\bspread\b",
        ],
    ),
    (
        ReferenceCategory.OILS_VINEGARS,
        [
            r"\bolive\s+oil\b",
            r"\bvegetable\s+oil\b",
            r"\bcanola\s+oil\b",
            r"\bavocado\s+oil\b",
            r"\bcoconut\s+oil\b",
            r"\bsunflower\s+oil\b",
            r"\bsesame\s+oil\b",
            r"\bgrapeseed\s+oil\b",
            r"\bvinegar\b",
            r"\bbalsamic\b",
        ],
    ),
    (
        ReferenceCategory.CONFECTIONERY,
        [
            r"\bchocolate\b",
            r"\bcandy\b",
            r"\bcaramel\b",
            r"\btoffee\b",
            r"\bfudge\b",
            r"\bgumm(?:y|ies)\b",
            r"\blollipop\b",
            r"\bmarshmallow\b",
            r"\blicorice\b",
            r"\btruffle\b",
        ],
    ),
    (
        ReferenceCategory.SNACKS,
        [
            r"\bchips\b",
            r"\bcrisps\b",
            r"\bpopcorn\b",
            r"\bcrackers?\b",
            r"\bpretzels?\b",
            r"\bsnacks?\b",
            r"\btrail\s+mix\b",
            r"\bgranola\s+bar\b",
            r"\bprotein\s+bar\b",
        ],
    ),
    (
        ReferenceCategory.PASTA_RICE,
        [
            r"\bpasta\b",
            r"\bspaghetti\b",
            r"\bmacaroni\b",
            r"\bpenne\b",
            r"\bnoodles?\b",
            r"\brice\b",
            r"\brisotto\b",
            r"\bcouscous\b",
        ],
    ),
    (
        ReferenceCategory.BREAKFAST,
        [
            r"\bcereal\b",
            r"\boatmeal\b",
            r"\boats\b",
            r"\bgranola\b",
            r"\bmuesli\b",
            r"\bbreakfast\b",
        ],
    ),
    (
        ReferenceCategory.CANNED_GOODS,
        [
            r"\bcanned\b",
            r"\bchickpeas\b",
            r"\blentils\b",
            r"\btomato\s+paste\b",
            r"\bcanned\s+soup\b",
            r"\bcanned\s+fruit\b",
            r"\bcanned\s+vegetables?\b",
        ],
    ),
    (
        ReferenceCategory.GENERAL_GROCERY,
        [
            r"\bsoup\b",
            r"\bbroth\b",
            r"\bstock\b",
            r"\bhoney\b",
            r"\bjam\b",
            r"\bjelly\b",
            r"\bpeanut\s+butter\b",
            r"\bnut\s+butter\b",
            r"\bsugar\b",
            r"\bsalt\b",
            r"\bspices?\b",
            r"\bseasoning\b",
            r"\bherbs?\b",
        ],
    ),
    (
        ReferenceCategory.HOUSEHOLD_CLEANING,
        [
            r"\bcleaner\b",
            r"\bcleaning\b",
            r"\bdisinfectant\b",
            r"\bdish\s+soap\b",
            r"\bdishwasher\b",
            r"\blaundry\s+detergent\b",
            r"\bdetergent\b",
            r"\bbleach\b",
            r"\bpolish\b",
            r"\bsurface\s+cleaner\b",
        ],
    ),
    (
        ReferenceCategory.HEALTH_REMEDIES,
        [
            r"\bvitamin\b",
            r"\bvitamins\b",
            r"\bsupplement\b",
            r"\bsupplements\b",
            r"\bcapsules?\b",
            r"\btablets?\b",
            r"\bpain\s+relief\b",
            r"\bcold\s+remedy\b",
            r"\bcough\s+syrup\b",
        ],
    ),
    (
        ReferenceCategory.PERSONAL_CARE,
        [
            r"\bshampoo\b",
            r"\bconditioner\b",
            r"\bsoap\b",
            r"\bbody\s+wash\b",
            r"\btoothpaste\b",
            r"\btoothbrush\b",
            r"\bdeodorant\b",
            r"\blotion\b",
            r"\brazor\b",
            r"\bshaving\b",
        ],
    ),
    (
        ReferenceCategory.BABY_CARE,
        [
            r"\bbaby\b",
            r"\binfant\b",
            r"\bdiapers?\b",
            r"\bformula\b",
            r"\bbaby\s+food\b",
            r"\bbaby\s+wipes?\b",
            r"\bpacifier\b",
        ],
    ),
    (
        ReferenceCategory.PET_FOOD,
        [
            r"\bdog\s+food\b",
            r"\bcat\s+food\b",
            r"\bpet\s+food\b",
            r"\bdog\s+treats?\b",
            r"\bcat\s+treats?\b",
            r"\bpet\s+treats?\b",
            r"\bbird\s+seed\b",
            r"\bbird\s+feed\b",
        ],
    ),
    (
        ReferenceCategory.HOUSEHOLD_SUPPLIES,
        [
            r"\bpaper\s+towels?\b",
            r"\btoilet\s+paper\b",
            r"\bgarbage\s+bags?\b",
            r"\btrash\s+bags?\b",
            r"\bstorage\s+bags?\b",
            r"\bfood\s+containers?\b",
            r"\bplastic\s+wrap\b",
            r"\baluminum\s+foil\b",
            r"\bpaper\s+bags?\b",
        ],
    ),
]


def classify_taxonomy(product_name: str) -> TaxonomyMatch:
    text = normalize_text(product_name)

    if not text:
        return TaxonomyMatch(
            category=None,
            confidence=0.0,
            matched_rule=None,
            status="AMBIGUOUS",
        )

    for pattern, category, confidence, rule_name in DISAMBIGUATION_RULES:
        if re.search(pattern, text):
            return TaxonomyMatch(
                category=category,
                confidence=confidence,
                matched_rule=rule_name,
                status="AMBIGUOUS" if category is None else "PASS",
            )

    matches = []

    for category, patterns in CATEGORY_RULES:
        for pattern in patterns:
            if re.search(pattern, text):
                matches.append((category, pattern))

    if not matches:
        return TaxonomyMatch(
            category=None,
            confidence=0.0,
            matched_rule=None,
            status="AMBIGUOUS",
        )

    categories = list(dict.fromkeys(category for category, _ in matches))

    if len(categories) > 1:
        return TaxonomyMatch(
            category=None,
            confidence=0.0,
            matched_rule="multiple_category_matches",
            status="AMBIGUOUS",
        )

    category, pattern = matches[0]

    return TaxonomyMatch(
        category=category,
        confidence=0.90,
        matched_rule=pattern,
        status="PASS",
    )