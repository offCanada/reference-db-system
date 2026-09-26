
from __future__ import annotations

import re
from dataclasses import dataclass

from reference_db.taxonomy.reference_taxonomy import ReferenceCategory
from reference_db.taxonomy.tier1_resolutions import (
    TIER0_DISAMBIGUATION_RULES,
    TIER0_NARROW_EXCLUSIONS,
)


@dataclass(frozen=True)
class TaxonomyMatch:
    category: ReferenceCategory | None
    confidence: float
    matched_rule: str | None
    status: str
    resolution: str = "unknown"
    candidates: tuple[str, ...] = ()
    other_candidates: tuple[str, ...] = ()


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Explicit disambiguation rules
#
# These are product-level rules that override generic keyword matching.
# They run after the semantic Tier-0 rules from tier1_resolutions.py.
#

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


# Specific phrase rules
#
# More specific product phrases run before generic keyword rules.
#

SPECIFIC_PHRASE_RULES = [
    (
        r"\bchocolate\s+milk\b",
        ReferenceCategory.DAIRY,
        0.85,
        "specific_chocolate_milk",
    ),
    (
        r"\bstrawberry\s+milk\b",
        ReferenceCategory.DAIRY,
        0.85,
        "specific_strawberry_milk",
    ),
    (
        r"\bflavou?red\s+milk\b",
        ReferenceCategory.DAIRY,
        0.85,
        "specific_flavoured_milk",
    ),
    (
        r"\borange\s+juice\b",
        ReferenceCategory.BEVERAGES,
        0.85,
        "specific_orange_juice",
    ),
    (
        r"\bapple\s+juice\b",
        ReferenceCategory.BEVERAGES,
        0.85,
        "specific_apple_juice",
    ),
    (
        r"\bcranberry\s+juice\b",
        ReferenceCategory.BEVERAGES,
        0.85,
        "specific_cranberry_juice",
    ),
    (
        r"\bgrape\s+juice\b",
        ReferenceCategory.BEVERAGES,
        0.85,
        "specific_grape_juice",
    ),
    (
        r"\bapple\s+cider\b",
        ReferenceCategory.ALCOHOLIC_BEVERAGES,
        0.85,
        "specific_apple_cider",
    ),
    (
        r"\broot\s+beer\b",
        ReferenceCategory.BEVERAGES,
        0.85,
        "specific_root_beer",
    ),
    (
        r"\bginger\s+beer\b",
        ReferenceCategory.BEVERAGES,
        0.85,
        "specific_ginger_beer",
    ),
    (
        r"\bpasta\s+sauce\b",
        ReferenceCategory.CONDIMENTS_SAUCES,
        0.85,
        "specific_pasta_sauce",
    ),
    (
        r"\bbarbecue\s+sauce\b",
        ReferenceCategory.CONDIMENTS_SAUCES,
        0.85,
        "specific_barbecue_sauce",
    ),
    (
        r"\bhot\s+sauce\b",
        ReferenceCategory.CONDIMENTS_SAUCES,
        0.85,
        "specific_hot_sauce",
    ),
    (
        r"\bchicken\s+soup\b",
        ReferenceCategory.GENERAL_GROCERY,
        0.85,
        "specific_chicken_soup",
    ),
    (
        r"\bbeef\s+soup\b",
        ReferenceCategory.GENERAL_GROCERY,
        0.85,
        "specific_beef_soup",
    ),
    (
        r"\bchicken\s+noodle\s+soup\b",
        ReferenceCategory.GENERAL_GROCERY,
        0.85,
        "specific_chicken_noodle_soup",
    ),
    (
        r"\bbaby\s+food\b",
        ReferenceCategory.BABY_CARE,
        0.90,
        "specific_baby_food",
    ),
    (
        r"\bbaby\s+formula\b",
        ReferenceCategory.BABY_CARE,
        0.90,
        "specific_baby_formula",
    ),
    (
        r"\bbaby\s+spinach\b",
        ReferenceCategory.PRODUCE,
        0.85,
        "specific_baby_spinach",
    ),
    (
        r"\bbaby\s+carrots?\b",
        ReferenceCategory.PRODUCE,
        0.85,
        "specific_baby_carrots",
    ),
    (
        r"\bbaking\s+soda\b",
        ReferenceCategory.GENERAL_GROCERY,
        0.85,
        "specific_baking_soda",
    ),
    (
        r"\bbaking\s+powder\b",
        ReferenceCategory.GENERAL_GROCERY,
        0.85,
        "specific_baking_powder",
    ),
    (
        r"\bprotein\s+bar\b",
        ReferenceCategory.SNACKS,
        0.85,
        "specific_protein_bar",
    ),
    (
        r"\bgranola\s+bar\b",
        ReferenceCategory.SNACKS,
        0.85,
        "specific_granola_bar",
    ),
    (
        r"\bpeanut\s+butter\b",
        ReferenceCategory.GENERAL_GROCERY,
        0.85,
        "specific_peanut_butter",
    ),
    (
        r"\bcream\s+cheese\b",
        ReferenceCategory.DAIRY,
        0.90,
        "specific_cream_cheese",
    ),
    (
        r"\bcottage\s+cheese\b",
        ReferenceCategory.DAIRY,
        0.90,
        "specific_cottage_cheese",
    ),
    (
        r"\bice\s+cream\b",
        ReferenceCategory.FROZEN,
        0.90,
        "specific_ice_cream",
    ),
    (
        r"\bmilk\s+chocolate\b",
        ReferenceCategory.CONFECTIONERY,
        0.85,
        "specific_milk_chocolate",
    ),
    (
        r"\bchocolate\s+chips?\b.*\b(?:cookies?|muffins?|brownies?|cakes?|scones?)\b",
        ReferenceCategory.BAKERY,
        0.90,
        "specific_chocolate_chip_bakery",
    ),
    (
        r"\b(?:cookies?|muffins?|brownies?|cakes?|scones?)\b.*\bchocolate\s+chips?\b",
        ReferenceCategory.BAKERY,
        0.90,
        "specific_chocolate_chip_bakery",
    ),
    (
        r"\bchocolate\s+chips?\b",
        ReferenceCategory.CONFECTIONERY,
        0.85,
        "specific_chocolate_chips",
    ),
    (
        r"\b(?:potato|tortilla|corn|pita)\s+chips?\b",
        ReferenceCategory.SNACKS,
        0.90,
        "specific_plain_chips",
    ),
    (
        r"\bcranberry\s+cocktail\b",
        ReferenceCategory.BEVERAGES,
        0.85,
        "specific_cranberry_cocktail",
    ),
    (
        r"\biced\s+tea\b",
        ReferenceCategory.BEVERAGES,
        0.85,
        "specific_iced_tea",
    ),
    (
        r"\bsparkling\s+water\b",
        ReferenceCategory.BEVERAGES,
        0.90,
        "specific_sparkling_water",
    ),
    (
        r"\bvegetable\s+juice\b",
        ReferenceCategory.BEVERAGES,
        0.85,
        "specific_vegetable_juice",
    ),
    (
        r"\bjelly\s+beans?\b",
        ReferenceCategory.CONFECTIONERY,
        0.90,
        "specific_jelly_beans",
    ),
    (
        r"\bchili\s+powder\b|\bpowder\s+chili\b",
        ReferenceCategory.GENERAL_GROCERY,
        0.85,
        "specific_chili_powder",
    ),
    (
        r"\bmaple\s+syrup\b",
        ReferenceCategory.GENERAL_GROCERY,
        0.85,
        "specific_maple_syrup",
    ),
    (
        r"\b(?:pumpkin|sunflower|chia|flax|sesame|hemp)\s+seeds?\b",
        ReferenceCategory.SNACKS,
        0.85,
        "specific_seed_snacks",
    ),
    (
        r"\bmixed\s+nuts?\b",
        ReferenceCategory.SNACKS,
        0.85,
        "specific_mixed_nuts",
    ),
    (
        r"\b(?:taco|fajita)\s+kits?\b",
        ReferenceCategory.CONDIMENTS_SAUCES,
        0.85,
        "specific_taco_fajita_kit",
    ),
    (
        r"\bwhip(?:ped)?\s+cream\b",
        ReferenceCategory.DAIRY,
        0.85,
        "specific_whipped_cream",
    ),
    (
        r"\bextract\s+vanilla\b",
        ReferenceCategory.BAKING_INGREDIENTS,
        0.85,
        "specific_extract_vanilla",
    ),
]


# Generic category rules

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
            r"\beggs?\b",
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
            r"\bsausages?\b",
            r"\bwieners?\b",
            r"\bfranks?\b",
            r"\bhot\s+dogs?\b",
            r"\bbacon\b",
            r"\bham\b",
            r"\bsalami\b",
            r"\bpepperoni\b",
            r"\bbologna\b",
            r"\bprosciutto\b",
            r"\bpastrami\b",
            r"\bcharcuterie\b",
            r"\bdeli\s+meats?\b",
            r"\bsmoked\s+meat\b",
            r"\bluncheon\s+meat\b",
            r"\bmeats?\b",
            r"\bsalmon\b",
            r"\btuna\b",
            r"\bshrimp\b",
            r"\bprawn\b",
            r"\bcrab\b",
            r"\blobster\b",
            r"\bmussels?\b",
            r"\bclams?\b",
            r"\boysters?\b",
            r"\bscallops?\b",
            r"\boctopus\b",
            r"\bsquid\b",
            r"\bcod\b",
            r"\btilapia\b",
            r"\btrout\b",
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
            r"\bwater\b",
        ],
    ),
    (
        ReferenceCategory.BAKERY,
        [
            r"\bbread\b",
            r"\bbuns?\b",
            r"\bbagels?\b",
            r"\bcroissants?\b",
            r"\bmuffins?\b",
            r"\bcake\b",
            r"\bcookies?\b",
            r"\bpastr(?:y|ies)\b",
            r"\bdonuts?\b",
            r"\bdoughnuts?\b",
            r"\btortillas?\b",
            r"\bflatbreads?\b",
            r"\bnaan\b",
            r"\bpies?\b",
            r"\bcroutons?\b",
            r"\bmacaroons?\b",
            r"\bdessert\s+(?:shells?|squares?)\b",
            r"\bwaffle\s+bowls?\b",
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
            r"\bcream\s+of\s+tartar\b",
            r"\btartar\b",
            r"\bextracts?\b",
            r"\bshortening\b",
            r"\blard\b",
            r"\bfrostings?\b",
            r"\bgraham\s+crumbs?\b",
            r"\bdesiccated\b",
            r"\b(?:shredded|flaked)\s+coconut\b",
            r"\bcoconut\s+flakes?\b",
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
            r"\bstrawberr(?:y|ies)\b",
            r"\bblueberr(?:y|ies)\b",
            r"\braspberr(?:y|ies)\b",
            r"\bpotato(?:es)?\b",
            r"\btomato(?:es)?\b",
            r"\bonions?\b",
            r"\bgarlic\b",
            r"\bcarrots?\b",
            r"\blettuce\b",
            r"\bspinach\b",
            r"\bbroccoli\b",
            r"\bcucumbers?\b",
            r"\bpeppers?\b",
            r"\bmushrooms?\b",
            r"\bromaine\b",
            r"\bkale\b",
            r"\barugula\b",
            r"\bmango(?:es)?\b",
            r"\bcranberr(?:y|ies)\b",
            r"\bcoleslaw\b",
            r"\bhearts?\b",
            r"\bcherries\b",
            r"\bpeach(?:es)?\b",
            r"\bpears?\b",
            r"\bavocados?\b",
            r"\bcelery\b",
            r"\bcabbage\b",
            r"\bcauliflower\b",
            r"\basparagus\b",
            r"\byams?\b",
            r"\bpumpkin\b",
            r"\bsquash\b",
            r"\bzucchini\b",
            r"\bjumbleberr(?:y|ies)\b",
            r"\bstir[-\s]?fry\b",
            r"\bchil(?:e|i)?es?\b",
            r"\bspring\s+mix\b",
            r"\bmixed\s+greens?\b",
            r"\bbrussel\s+sprouts?\b",
            r"\bgrapefruit\b",
            r"\bclementines?\b",
            r"\bmandarins?\b",
            r"\bbeets?\b",
            r"\bradish(?:es)?\b",
            r"\bturnips?\b",
            r"\bparsnips?\b",
            r"\bsweet\s+potatoes?\b",
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
            r"\bhummus\b",
            r"\bguacamole\b",
            r"\bmarinade\b",
            r"\btacos?\b",
            r"\bolives?\b",
            r"\bpickles?\b",
            r"\bgherkins?\b",
            r"\bpickled\b",
            r"\bhorseradish\b",
            r"\bsoy\s+sauce\b",
            r"\bteriyaki\b",
            r"\bsriracha\b",
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
            r"\bsafflower\s+oil\b",
            r"\boils?\b.*\b(?:canola|sunflower|vegetable|coconut|avocado|olive|safflower|grapeseed|sesame|peanut)\b",
            r"\bcooking\s+spray\b",
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
            r"\bmarshmallows?\b",
            r"\blicorice\b",
            r"\bliquorice\b",
            r"\btruffle\b",
            r"\bpuddings?\b",
            r"\braisins?\b",
            r"\bsultanas?\b",
            r"\bprunes?\b",
            r"\bjujubes?\b",
            r"\bbrittle\b",
            r"\bconfection\b",
            r"\blollipops?\b",
            r"\bdried\s+fruit\b",
        ],
    ),
    (
        ReferenceCategory.SNACKS,
        [
            r"\bchips?\b",
            r"\bcrisps?\b",
            r"\bpopcorn\b",
            r"\bcrackers?\b",
            r"\bpretzels?\b",
            r"\bsnacks?\b",
            r"\btrail\s+mix\b",
            r"\bgranola\s+bar\b",
            r"\bprotein\s+bar\b",
            r"\balmonds?\b",
            r"\bwalnuts?\b",
            r"\bcashews?\b",
            r"\bpecans?\b",
            r"\bpeanuts?\b",
            r"\bpistachios?\b",
            r"\bsunflower\s+seeds?\b",
            r"\bsesame\s+seeds?\b",
            r"\bmixed\s+nuts?\b",
            r"\bnut\s+mix(?:es)?\b",
            r"\bpopping\s+corn\b",
            r"\bparty\s+mix\b",
            r"\bpub\s+mix\b",
            r"\bkettle\s+corn\b",
            r"\bmacadamia\b",
            r"\bhazelnuts?\b",
            r"\bpine\s+nuts?\b",
            r"\bflax\s+seeds?\b",
            r"\bchia\s+seeds?\b",
            r"\bhemp\s+seeds?\b",
            r"\bsesame\s+seeds?\b",
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
            r"\blasagna\b",
            r"\bgnocchi\b",
            r"\bquinoa\b",
            r"\bbarley\b",
            r"\bfarro\b",
            r"\bsplit\s+peas?\b",
            r"\bpolenta\b",
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
            r"\bwaffles?\b",
            r"\bpancakes?\b",
            r"\bfrench\s+toast\b",
            r"\bcrepes?\b",
        ],
    ),
    (
        ReferenceCategory.CANNED_GOODS,
        [
            r"\bcanned\b",
            r"\bchickpeas\b",
            r"\bchick\s+peas?\b",
            r"\blentils\b",
            r"\bblack\s+beans?\b",
            r"\bpinto\s+beans?\b",
            r"\bkidney\s+beans?\b",
            r"\bnavy\s+beans?\b",
            r"\brefried\s+beans?\b",
            r"\bbaked\s+beans?\b",
            r"\bgarbanzo\s+beans?\b",
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
            r"\bthyme\b",
            r"\bcinnamon\b",
            r"\boregano\b",
            r"\bbasil\b",
            r"\bpaprika\b",
            r"\bcumin\b",
            r"\bginger\b",
            r"\bnutmeg\b",
            r"\bparsley\b",
            r"\bsage\b",
            r"\bdill\b",
            r"\brosemary\b",
            r"\bchives?\b",
            r"\bcoriander\b",
            r"\bturmeric\b",
            r"\bpeppercorns?\b",
            r"\bcloves?\b",
            r"\bbay\s+leaves?\b",
            r"\bstevia\b",
            r"\bsyrups?\b",
            r"\bmolasses\b",
            r"\bmarmalade\b",
            r"\bsweeteners?\b",
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
    """
    Classify a product using a tiered evidence strategy.

    Resolution order:

    1. Empty input -> AMBIGUOUS
    2. Semantic Tier-0 rules + narrow exclusions
    3. Explicit disambiguation rules
    4. Specific product phrases
    5. Generic category rules
    6. One category -> PASS
    7. FROZEN precedence for exactly two candidates
    8. Multiple categories -> AMBIGUOUS
    9. No evidence -> AMBIGUOUS
    """

    text = normalize_text(product_name)

    if not text:
        return TaxonomyMatch(
            category=None,
            confidence=0.0,
            matched_rule=None,
            status="AMBIGUOUS",
            resolution="no_evidence",
            candidates=(),
        )

    # Tier 0: semantic meaning-based disambiguation
    #
    # These rules were created from the Phase-3 conflict audit.
    # They intentionally run before all generic and specific rules.
    #
    # Narrow exclusions protect stable products from accidental
    # taxonomy changes when a semantic phrase collides with another
    # product meaning.
    #

    for pattern, category_value, confidence, rule_name in TIER0_DISAMBIGUATION_RULES:
        exclusions = TIER0_NARROW_EXCLUSIONS.get(rule_name, ())

        # A matching exclusion vetoes this Tier-0 rule.
        if any(re.search(exclusion, text) for exclusion in exclusions):
            continue

        if not re.search(pattern, text):
            continue

        category = (
            ReferenceCategory(category_value)
            if category_value is not None
            else None
        )

        if category is None:
            return TaxonomyMatch(
                category=None,
                confidence=confidence,
                matched_rule=rule_name,
                status="AMBIGUOUS",
                resolution="tier0_disambiguation",
                candidates=(),
            )

        return TaxonomyMatch(
            category=category,
            confidence=confidence,
            matched_rule=rule_name,
            status="PASS",
            resolution="tier0_disambiguation",
            candidates=(category.value,),
        )

    # Tier 1: explicit disambiguation rules

    for pattern, category, confidence, rule_name in DISAMBIGUATION_RULES:
        if re.search(pattern, text):
            if category is None:
                return TaxonomyMatch(
                    category=None,
                    confidence=confidence,
                    matched_rule=rule_name,
                    status="AMBIGUOUS",
                    resolution="disambiguation_rule",
                    candidates=(),
                )

            return TaxonomyMatch(
                category=category,
                confidence=confidence,
                matched_rule=rule_name,
                status="PASS",
                resolution="disambiguation_rule",
                candidates=(category.value,),
            )

    # Tier 2: specific product phrases

    for pattern, category, confidence, rule_name in SPECIFIC_PHRASE_RULES:
        if re.search(pattern, text):
            return TaxonomyMatch(
                category=category,
                confidence=confidence,
                matched_rule=rule_name,
                status="PASS",
                resolution="specific_phrase",
                candidates=(category.value,),
            )

    # Tier 3: generic category evidence

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
            resolution="no_evidence",
            candidates=(),
        )

    categories = list(dict.fromkeys(category for category, _ in matches))

    # Tier 4: one unopposed category

    if len(categories) == 1:
        category = categories[0]
        pattern = matches[0][1]

        return TaxonomyMatch(
            category=category,
            confidence=0.90,
            matched_rule=pattern,
            status="PASS",
            resolution="unopposed",
            candidates=(category.value,),
        )

    # Tier 5: FROZEN precedence
    #
    # Frozen products dominate when FROZEN is one of exactly two
    # supporting categories.
    #

    if len(categories) == 2 and ReferenceCategory.FROZEN in categories:
        other = next(
            category
            for category in categories
            if category is not ReferenceCategory.FROZEN
        )

        return TaxonomyMatch(
            category=ReferenceCategory.FROZEN,
            confidence=0.90,
            matched_rule="frozen_precedence",
            status="PASS",
            resolution="precedence",
            candidates=(
                ReferenceCategory.FROZEN.value,
                other.value,
            ),
            other_candidates=(other.value,),
        )

    # Tier 6: genuine conflict
    #
    # Do not arbitrarily select a category.
    # Preserve ambiguity when multiple generic categories remain.
    #

    candidate_values = tuple(
        category.value for category in categories
    )

    return TaxonomyMatch(
        category=None,
        confidence=0.0,
        matched_rule="multiple_category_matches",
        status="AMBIGUOUS",
        resolution="conflict",
        candidates=candidate_values,
    )
