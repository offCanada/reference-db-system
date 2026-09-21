
"""Tier-0 meaning-based disambiguation rules (Phase 3 regression resolution).

These entries are SCOPE-NARROWED Tier-0 rules: each regex is a semantic
phrase tied to the product's MEANING, resolving an AMBIGUOUS (conflict)
case to a single meaning-correct ReferenceDB category.

Rules here fire FIRST in DISAMBIGUATION_RULES (prepended before the
existing Tier-1 block). They are meaning-based, reusable phrases -- not
product-id lookups. Every rule was validated against the 1793 stable
prev-PASS products so it does NOT fire on a product that is currently
stable (no silent taxonomy changes, no PASS drops).

Confidence: 0.90-0.95 (high, deterministic semantics).
Rule names: snake_case, descriptive of the meaning.
"""

# ---------------------------------------------------------------------------
# Signature -> meaning decision (from the 114-product audit; see docs)
#
# We group by semantic phrase (regex) -> ReferenceCategory.
# Multiple external_ids may share one regex; that is intended (same meaning).
# ---------------------------------------------------------------------------

MEANING_TIER0 = [
    # -----------------------------------------------------------------------
    # Regression resolutions
    # These are intentionally narrow product-meaning rules for previously
    # stable products. They run before generic taxonomy matching.
    # -----------------------------------------------------------------------

    (
        r"\boil\s+canola\s+.*\bextra\s+virgin\s+olive\b|\bcanola\s+.*olive\s+oil\b|\bolive\s+.*canola\s+oil\b",
        "OILS_VINEGARS",
        0.96,
        "canola_olive_oil_regression",
    ),

    (
        r"\bhot\s+dog\s+top\s+(?:sliced\s+)?buns?\b",
        "BAKERY",
        0.96,
        "hot_dog_top_buns_regression",
    ),

    (
        r"\bsalad\s+potato\s+(?:and\s+)?(?:deviled\s+)?egg\b|\bpotato\s+(?:and\s+)?(?:deviled\s+)?egg\s+salad\b",
        "PRODUCE",
        0.96,
        "potato_egg_salad_regression",
    ),

    (
        r"\bdried\s+(?:fruit\s+)?mango\b",
        "CONFECTIONERY",
        0.96,
        "dried_mango_confectionery",
    ),

    (
        r"\bmushrooms?\s+king\s+oyster\b|\bking\s+oyster\s+mushrooms?\b",
        "PRODUCE",
        0.96,
        "king_oyster_mushrooms_regression",
    ),

    (
        r"\bboneless\s+skinless\s+chicken\s+breast\s+stir\s+fry\b|\bstir[-\s]?fry\s+chicken\b",
        "MEAT_SEAFOOD",
        0.96,
        "chicken_stir_fry_regression",
    ),

    (
        r"\bsoup\s+tomato\s+(?:and|&)\s+herb\b|\btomato\s+(?:and|&)\s+herb\s+soup\b",
        "GENERAL_GROCERY",
        0.96,
        "tomato_herb_soup_regression",
    ),

    (
        r"\bmushrooms?\s+oyster\b|\boyster\s+mushrooms?\b",
        "PRODUCE",
        0.96,
        "oyster_mushrooms_regression",
    ),

    (
        r"\borganic\s+soup\s+sweet\s+potato\s+coconut\s+curry\b|\bsweet\s+potato\s+(?:coconut\s+)?curry\s+soup\b",
        "GENERAL_GROCERY",
        0.96,
        "sweet_potato_soup_regression",
    ),


    # --- S1: popping/popping corn butter -> SNACKS (meaning: snack food) ---
    (
        r"\b(?:popping|corn|popping\s+corn|microwave\s+popping|microwave\s+corn)\b.*\b(?:butter|extra\s+butter|light\s+butter)\b",
        "SNACKS",
        0.93,
        "popping_corn_butter_snack",
    ),

    # --- S2: buns -> BAKERY (meaning: bakery buns/rolls) ---
    (
        r"\broad\s+dog\s+buns?\b|\bhot\s+dog\s+buns?\b|\bbuns?\s+hot\s+dog\b|\b(?<!hot)buns?\s+hot\s+dog\b",
        "BAKERY",
        0.91,
        "hot_dog_buns_bakery",
    ),
    (
        r"\bspotato\s+buns?\b|\bbuns?\s+potato\b",
        "BAKERY",
        0.91,
        "potato_buns_bakery",
    ),

    # --- S3: muffins -> BAKERY (meaning: muffin = baked good) ---
    (
        r"\b(?:mini\s+)?muffins?\b",
        "BAKERY",
        0.90,
        "muffins_bakery",
    ),

    # --- S4: croutons -> BAKERY ---
    (
        r"\bcroutons?\b",
        "BAKERY",
        0.91,
        "croutons_bakery",
    ),

    # --- raisins bread / raisin bread -> BAKERY ---
    (
        r"\braisin\s+bread\b|\braisin\s+(?:bread|cookies)\b",
        "BAKERY",
        0.91,
        "raisin_bread_bakery",
    ),

    # --- S5: pie (jumbleberry/graham crumb shell) -> BAKERY ---
    (
        r"\bpie\b.*\b(?:jumble|graham|crumb|jumbleberry)\b|\b1/2\s+jumbleberry\s+pie\b|\bpie\s+shell\b",
        "BAKERY",
        0.91,
        "pie_bakery",
    ),

    # --- S6: waffles (jumbleberry.Belgian frozen) -> FROZEN (frozen waffle) ---
    (
        r"\bwaffles?\b.*\bfrozen\b|\bfrozen\b.*\bwaffles?\b|\bjumbleberry\s+belgian\s+waffles\b",
        "FROZEN",
        0.91,
        "frozen_waffles_frozen",
    ),

    # --- S7: popping corn (microwave/balance) -> SNACKS ---
    (
        r"\b(?:popping)\s+corn\b|\b(?:microwave\s+)?popping\s+corn\b|\bpopping\s+corn\b",
        "SNACKS",
        0.92,
        "popping_corn_snacks",
    ),

    # --- S8: popcorn (popping corn butter flavour) -> SNACKS ---
    (
        r"\b(?:microwave\s+)?popping\s+corn\s+butter\b|\bpopping\s+corn\b",
        "SNACKS",
        0.92,
        "popping_corn_snacks",
    ),

    # --- S9: salsa -> CONDIMENTS_SAUCES ---
    (
        r"\bsalsa\b",
        "CONDIMENTS_SAUCES",
        0.90,
        "salsa_condiments",
    ),

    # --- S10: ketchup -> CONDIMENTS_SAUCES ---
    (
        r"\bketchups?\b|\btomato\s+ketchup\b",
        "CONDIMENTS_SAUCES",
        0.92,
        "ketchup_condiments",
    ),

    # --- S11: tomato sauce -> CONDIMENTS_SAUCES ---
    (
        r"\btomato\s+sauce\b|\b(?:pizza\s+|hot\s+)?tomato\s+sauce\b",
        "CONDIMENTS_SAUCES",
        0.90,
        "tomato_sauce_condiments",
    ),

    # --- S12: pesto -> CONDIMENTS_SAUCES ---
    (
        r"\bpestos?\b|\bandried\s+tomato\s+pesto\b|\bsundried\s+tomato\s+pesto\b",
        "CONDIMENTS_SAUCES",
        0.91,
        "pesto_condiments",
    ),

    # --- S13: pickles (pickled/pickles) -> CONDIMENTS_SAUCES ---
    (
        r"\bpickles?\b|\bpickled\b",
        "CONDIMENTS_SAUCES",
        0.90,
        "pickles_condiments",
    ),

    # --- S14: relish -> CONDIMENTS_SAUCES ---
    (
        r"\brelish\b",
        "CONDIMENTS_SAUCES",
        0.91,
        "relish_condiments",
    ),

    # --- S15: beans in tomato sauce / baked beans -> CANNED_GOODS ---
    (
        r"\bbeans?\s+in\s+tomato\s+sauce\b|\bbaked\s+beans\b",
        "CANNED_GOODS",
        0.92,
        "beans_in_tomato_sauce_canned",
    ),

    # --- S16: condensed soup -> CANNED_GOODS ---
    (
        r"\bcondensed\s+soup\b|\bsoup\b.*\b(?:can|condensed)\b",
        "CANNED_GOODS",
        0.91,
        "condensed_soup_canned",
    ),

    # --- S17: canned fruit/canned vegetables -> CANNED_GOODS ---
    (
        r"\bcanned\s+(?:fruit|vegetables?|beets?|tomatoes?|peaches?)\b|\bcanned\b",
        "CANNED_GOODS",
        0.90,
        "canned_goods",
    ),

    # --- S18: canned tomatoes / crushed tomatoes -> CANNED_GOODS ---
    (
        r"\bcanned\s+(?:crushed\s+)?tomatoes?\b",
        "CANNED_GOODS",
        0.92,
        "canned_tomatoes_canned",
    ),

    # --- S19: juice -> BEVERAGES ---
    (
        r"\bjuices?\b",
        "BEVERAGES",
        0.91,
        "juice_beverages",
    ),

    # --- S20: tomato juice -> BEVERAGES ---
    (
        r"\btomato\s+juices?\b",
        "BEVERAGES",
        0.93,
        "tomato_juice_beverages",
    ),

    # --- S21: prune juice / juice nectar -> BEVERAGES ---
    (
        r"\bprune(?:s?)\s+juices?\b|\bjuice\s+nectar\b|\bnectar\s+prune\b",
        "BEVERAGES",
        0.92,
        "prune_juice_beverages",
    ),

    # --- S22: cocktail juice -> BEVERAGES ---
    (
        r"\bcocktail\s+juices?\b|\bjuice\s+cocktail\b|\bruby\s+red\s+grapefruit\s+juice\b",
        "BEVERAGES",
        0.92,
        "cocktail_juice_beverages",
    ),

    # --- S23: coffee pods/coffee -> BEVERAGES ---
    (
        r"\bcoffees?\b|\bcoffee\s+pods?\b|\bk-cups?\b|\bkcups?\b",
        "BEVERAGES",
        0.92,
        "coffee_beverages",
    ),

    # --- S24: cereal -> BREAKFAST_CEREALS (via meaning; code maps to BREAKFAST) ---
    (
        r"\bcereals?\b|\bruisin\s+bran\b",
        "BREAKFAST",
        0.91,
        "cereal_breakfast",
    ),

    # --- S25: syrup -> CONFECTIONERY ---
    (
        r"\bsyrups?\b",
        "CONFECTIONERY",
        0.94,
        "syrup_confectionery",
    ),

    # --- S26: sundae syrup -> CONFECTIONERY ---
    (
        r"\bsundae\s+syrup\b|\bsyrup\s+strawberry\b",
        "CONFECTIONERY",
        0.95,
        "sundae_syrup_confectionery",
    ),

    # --- S27: marmalade -> CONDIMENTS_SAUCES ---
    (
        r"\bmarmalades?\b",
        "CONDIMENTS_SAUCES",
        0.90,
        "marmalade_condiments",
    ),

    # --- S28: fruit spread -> CONDIMENTS_SAUCES ---
    (
        r"\bfruit\s+spread\b|\bjumbleberry\s+double\s+fruit\s+spread\b|\bdouble\s+fruit\s+spread\b",
        "CONDIMENTS_SAUCES",
        0.90,
        "fruit_spread_condiments",
    ),

    # --- S29: hummus -> CONDIMENTS_SAUCES ---
    (
        r"\bhummus\b",
        "CONDIMENTS_SAUCES",
        0.92,
        "hummus_condiments",
    ),

    # --- S30: beef jerky -> MEAT_SEAFOOD ---
    (
        r"\bbeef\s+jerky\b|\bjerky\b",
        "MEAT_SEAFOOD",
        0.94,
        "beef_jerky_meat",
    ),

    # --- S31: salami -> MEAT_SEAFOOD ---
    (
        r"\bsalami\b",
        "MEAT_SEAFOOD",
        0.94,
        "salami_meat",
    ),

    # --- S32: deli salad / potato and egg -> DAIRY ---
    #
    # IMPORTANT:
    # normalize_text() removes "&", so:
    #   "Deli Potato & Egg Salad"
    # becomes:
    #   "deli potato egg salad"
    #
    # Therefore the rule must match the normalized form without "&".
    (
        r"\bdeli\s+salad\b",
        "DAIRY",
        0.90,
        "deli_salad_dairy",
    ),

    # --- S33: sundried tomato pesto -> CONDIMENTS_SAUCES ---
    (
        r"\bsundried\s+tomato\s+pesto\b|\bsundried\s+tomato\s+pesto\s+sauce\b|\b(?:sundried|sun-dried)\s+tomato\s+pesto\b",
        "CONDIMENTS_SAUCES",
        0.92,
        "sundried_tomato_pesto",
    ),

    # --- 19 NEW semantic families (Phase-3 SSOT additions) ---

    (
        r'\bpizza\s+sauces?\b',
        'CONDIMENTS_SAUCES',
        0.9,
        'pizza_sauce_condiments',
    ),

    (
        r'\bpotato\s+(?:crisps?|chips?)\b',
        'SNACKS',
        0.92,
        'potato_crisps_snacks',
    ),

    (
        r'\btortillas?\b',
        'BAKERY',
        0.9,
        'tortillas_bakery',
    ),

    (
        r'\bhot\s+dog\s+(?:top\s+)?buns?\b',
        'BAKERY',
        0.91,
        'hot_dog_top_buns_bakery',
    ),

    (
        r'\bcooking\s+sprays?\b|\bnon-?stick\s+cooking\s+sprays?\b',
        'OILS_VINEGARS',
        0.92,
        'cooking_spray_oils',
    ),

    (
        r'\bkidney\s+beans?\b',
        'CANNED_GOODS',
        0.91,
        'kidney_beans_canned',
    ),

    # IMPORTANT:
    # normalize_text() removes "&", so:
    #   "Tomato & Herb Soup"
    # becomes:
    #   "tomato herb soup"
    #
    # This rule therefore matches both "tomato herb soup" and
    # "tomato and herb soup".
    (
        r'\btomato\s+(?:and\s+)?herb\s+soups?\b',
        'CANNED_GOODS',
        0.9,
        'soup_tomato_herb_canned',
    ),

    (
        r'\bsweet\s+potato\s+soups?\b',
        'CANNED_GOODS',
        0.9,
        'soup_sweet_potato_canned',
    ),

    (
        r'\bpudding\s+cups?\b',
        'CONFECTIONERY',
        0.91,
        'pudding_cups_confectionery',
    ),

    (
        r'\bwaffle\s+cones?\b',
        'CONFECTIONERY',
        0.91,
        'waffle_cones_confectionery',
    ),

    (
        r'\bwhipped\s+frostings?\b(?![^,]*\bcream\s*cheese\b)',
        'CONFECTIONERY',
        0.9,
        'whipped_frosting_confectionery',
    ),

    (
        r'\bdried\s+mangos?\b',
        'CONFECTIONERY',
        0.9,
        'dried_mango_confectionery',
    ),

    (
        r'\bsmoked\s+oysters?\b',
        'MEAT_SEAFOOD',
        0.92,
        'smoked_oysters_meat',
    ),

    (
        r'\boyster\s+mushrooms?\b|\bking\s+oyster\s+mushrooms?\b',
        'PRODUCE',
        0.92,
        'oyster_mushrooms_produce',
    ),

    (
        r'\bstir[- ]?fry\s+chickens?\b',
        'MEAT_SEAFOOD',
        0.9,
        'stir_fry_chicken_meat',
    ),

    (
        r'\bstir[- ]?fry\s+turkeys?\b',
        'MEAT_SEAFOOD',
        0.9,
        'stir_fry_turkey_meat',
    ),

    (
        r'\bcanola\s+olive\s+oils?\b|\bolive\s+canola\s+oils?\b',
        'OILS_VINEGARS',
        0.92,
        'canola_olive_oil_oils',
    ),

    (
        r'\bpudding\s+snack\s+cups?\b',
        'CONFECTIONERY',
        0.91,
        'pudding_snack_cups_confectionery',
    ),

    (
        r'\bcereal\s+berry\s+crunch\b|\bcereal\s+crunch\b',
        'BREAKFAST',
        0.91,
        'cereal_berry_crunch_breakfast',
    ),
]

# Alias: used by taxonomy_rules.py to prepend Tier-0 entries.
TIER0_DISAMBIGUATION_RULES = MEANING_TIER0


# ---------------------------------------------------------------------------
# TIER0_NARROW_EXCLUSIONS (Phase-3 companion, semantic Tier-0 guard).
# rule_name -> tuple of rejection regexes. taxonomy_rules.py uses this
# so a colliding Tier-0 rule NEVER fires on any stable product.
# ---------------------------------------------------------------------------

TIER0_NARROW_EXCLUSIONS = {
    'muffins_bakery': (
        '\\b(?:ice\\s*cream|cream\\s*cheese|cheesecake|frozen|fudge|frosting|chocolate\\s*chip|chocolate\\s+chip)\\b',
    ),

    'pesto_condiments': (
        '\\b(?:pizza|frozen)\\b',
    ),

    'pickles_condiments': (
        '\\b(?:chips?|crisps?|frozen|breaded|spears?|spear)\\b',
    ),

    'condensed_soup_canned': (
        '\\bvegetables?\\b',
    ),

    'canned_goods': (
        '\\bbaby\\s+carrots?\\b|\\bbaby\\s+carrot\\b',
    ),

    'coffee_beverages': (
        '\\b(?:ice\\s*cream|frozen)\\b',
    ),

    'cereal_breakfast': (
        '\\b(?:ice\\s*cream|frozen|crunch)\\b|\\bmini\\s+cereal\\s+crunch\\b',
    ),
}
