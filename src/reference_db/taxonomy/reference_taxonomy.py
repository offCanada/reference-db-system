from enum import Enum


class ReferenceCategory(str, Enum):
    DAIRY = "DAIRY"
    MEAT_SEAFOOD = "MEAT_SEAFOOD"
    BEVERAGES = "BEVERAGES"
    ALCOHOLIC_BEVERAGES = "ALCOHOLIC_BEVERAGES"
    BAKERY = "BAKERY"
    BAKING_INGREDIENTS = "BAKING_INGREDIENTS"
    FROZEN = "FROZEN"
    PRODUCE = "PRODUCE"
    CONDIMENTS_SAUCES = "CONDIMENTS_SAUCES"
    OILS_VINEGARS = "OILS_VINEGARS"
    CONFECTIONERY = "CONFECTIONERY"
    SNACKS = "SNACKS"
    PASTA_RICE = "PASTA_RICE"
    BREAKFAST = "BREAKFAST"
    CANNED_GOODS = "CANNED_GOODS"
    GENERAL_GROCERY = "GENERAL_GROCERY"
    HOUSEHOLD_CLEANING = "HOUSEHOLD_CLEANING"
    HEALTH_REMEDIES = "HEALTH_REMEDIES"
    PERSONAL_CARE = "PERSONAL_CARE"
    BABY_CARE = "BABY_CARE"
    PET_FOOD = "PET_FOOD"
    HOUSEHOLD_SUPPLIES = "HOUSEHOLD_SUPPLIES"


TAXONOMY_DEFINITIONS = {
    ReferenceCategory.DAIRY: {
        "description": "Milk, cheese, yogurt, butter, cream and other dairy products."
    },
    ReferenceCategory.MEAT_SEAFOOD: {
        "description": "Meat, poultry, fish and seafood products."
    },
    ReferenceCategory.BEVERAGES: {
        "description": "Non-alcoholic beverages and drinks."
    },
    ReferenceCategory.ALCOHOLIC_BEVERAGES: {
        "description": "Beer, wine, spirits and other alcoholic beverages."
    },
    ReferenceCategory.BAKERY: {
        "description": "Bread, buns, pastries, cakes, cookies and bakery products."
    },
    ReferenceCategory.BAKING_INGREDIENTS: {
        "description": "Flour, baking powder, yeast, cocoa and other baking ingredients."
    },
    ReferenceCategory.FROZEN: {
        "description": "Frozen food products."
    },
    ReferenceCategory.PRODUCE: {
        "description": "Fresh fruits and vegetables."
    },
    ReferenceCategory.CONDIMENTS_SAUCES: {
        "description": "Sauces, dressings, spreads and condiments."
    },
    ReferenceCategory.OILS_VINEGARS: {
        "description": "Cooking oils, vinegars and related products."
    },
    ReferenceCategory.CONFECTIONERY: {
        "description": "Candy, chocolate and other confectionery products."
    },
    ReferenceCategory.SNACKS: {
        "description": "Chips, crackers, popcorn, snack bars and similar products."
    },
    ReferenceCategory.PASTA_RICE: {
        "description": "Pasta, noodles, rice and related staple products."
    },
    ReferenceCategory.BREAKFAST: {
        "description": "Breakfast cereals, oatmeal, granola and similar products."
    },
    ReferenceCategory.CANNED_GOODS: {
        "description": "Canned, preserved and shelf-stable food products."
    },
    ReferenceCategory.GENERAL_GROCERY: {
        "description": "General grocery products that do not fit a more specific category."
    },
    ReferenceCategory.HOUSEHOLD_CLEANING: {
        "description": "Cleaning products and household cleaning supplies."
    },
    ReferenceCategory.HEALTH_REMEDIES: {
        "description": "Vitamins, supplements, remedies and health-related products."
    },
    ReferenceCategory.PERSONAL_CARE: {
        "description": "Personal hygiene, grooming and cosmetic care products."
    },
    ReferenceCategory.BABY_CARE: {
        "description": "Baby food, diapers and baby care products."
    },
    ReferenceCategory.PET_FOOD: {
        "description": "Food and treats intended for pets."
    },
    ReferenceCategory.HOUSEHOLD_SUPPLIES: {
        "description": "General household supplies not primarily used for cleaning."
    },
}


def get_all_categories() -> list[ReferenceCategory]:
    return list(ReferenceCategory)


def get_taxonomy_definition(
    category: ReferenceCategory,
) -> dict:
    return TAXONOMY_DEFINITIONS[category]