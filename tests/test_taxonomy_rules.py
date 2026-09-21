"""Tests for taxonomy classification rules."""
from __future__ import annotations

import pytest

from reference_db.taxonomy.reference_taxonomy import ReferenceCategory
from reference_db.taxonomy.taxonomy_rules import classify_taxonomy


class TestClearSingleCategory:
    """Products that should classify to exactly one category."""

    def test_dairy_milk(self):
        result = classify_taxonomy("Milk 2% 1L")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.DAIRY

    def test_meat_chicken(self):
        result = classify_taxonomy("Chicken Breast Boneless")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.MEAT_SEAFOOD

    def test_beverages_juice(self):
        result = classify_taxonomy("Orange Juice From Concentrate")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.BEVERAGES

    def test_bakery_bread(self):
        result = classify_taxonomy("Whole Wheat Bread")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.BAKERY

    def test_produce_apple(self):
        result = classify_taxonomy("Organic Apples")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.PRODUCE

    def test_snacks_chips(self):
        result = classify_taxonomy("Kettle Cooked Potato Chips")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.SNACKS

    def test_frozen_ice_cream(self):
        result = classify_taxonomy("Vanilla Ice Cream 1.5L")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.FROZEN


class TestPreviouslyNoEvidence:
    """Products that previously returned no_evidence and should now classify."""

    def test_sliced_almonds(self):
        result = classify_taxonomy("Sliced Almonds 250 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.SNACKS

    def test_sunflower_seeds(self):
        result = classify_taxonomy("Roasted Salted Sunflower Seeds 375 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.SNACKS

    def test_olives(self):
        result = classify_taxonomy("Kalamata Whole Olives 375 ml")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.CONDIMENTS_SAUCES

    def test_water(self):
        result = classify_taxonomy("Natural Spring Water 4 L")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.BEVERAGES

    def test_safflower_oil(self):
        result = classify_taxonomy("Safflower Oil 100% Pure 946 ml")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.OILS_VINEGARS

    def test_tacos(self):
        result = classify_taxonomy("Taco Kit Hard And Soft 393 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.CONDIMENTS_SAUCES

    def test_romaine(self):
        result = classify_taxonomy("Romaine Hearts 510 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.PRODUCE

    def test_cream_of_tartar(self):
        result = classify_taxonomy("Cream Of Tartar 100 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.BAKING_INGREDIENTS

    def test_eggs(self):
        result = classify_taxonomy("Brown Eggs Large Carton 12 Count")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.DAIRY

    def test_thyme(self):
        result = classify_taxonomy("Ground Thyme 80 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.GENERAL_GROCERY

    def test_cinnamon(self):
        result = classify_taxonomy("Cinnamon Sticks 105 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.GENERAL_GROCERY

    def test_guacamole(self):
        result = classify_taxonomy("Guacamole 454 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.CONDIMENTS_SAUCES

    def test_hummus(self):
        result = classify_taxonomy("Mini Chick Pea Hummus 228 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.CONDIMENTS_SAUCES

    def test_dried_mango(self):
        # Tier-0 meaning rule dried_mango_confectionery resolves dried-mango
        # slices to CONFECTIONERY before generic produce keywords fire.
        result = classify_taxonomy("Dried Mango Slices 200 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.CONFECTIONERY
        assert result.matched_rule == "dried_mango_confectionery"

    def test_pecan_halves(self):
        result = classify_taxonomy("Pecan Halves 155 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.SNACKS

    def test_walnut_pieces(self):
        result = classify_taxonomy("Walnut Pieces 250 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.SNACKS

    def test_cranberry(self):
        result = classify_taxonomy("Cocktail White Cranberry 1.89 L")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.PRODUCE

    def test_blueberries(self):
        result = classify_taxonomy("Fresh Blueberries 340 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.PRODUCE

    def test_strawberries(self):
        result = classify_taxonomy("Fresh Strawberries 454 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.PRODUCE


class TestGenuineAmbiguity:
    """Products that are genuinely ambiguous and should remain AMBIGUOUS."""

    def test_frozen_chicken(self):
        """Frozen + chicken resolves to FROZEN by frozen_precedence."""
        result = classify_taxonomy("Frozen Chicken Breast 4 kg")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.FROZEN
        assert result.matched_rule == "frozen_precedence"
        candidates = set(result.candidates)
        assert ReferenceCategory.FROZEN in candidates
        assert ReferenceCategory.MEAT_SEAFOOD in candidates

    def test_milk_chocolate(self):
        """Milk chocolate resolves to CONFECTIONERY by specific phrase."""
        result = classify_taxonomy("Milk Chocolate Bar 100 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.CONFECTIONERY
        assert result.matched_rule == "specific_milk_chocolate"
        candidates = set(result.candidates)
        assert ReferenceCategory.CONFECTIONERY in candidates

    def test_apple_juice_is_beverages(self):
        """Apple juice is handled by Tier-0 juice rule → PASS/BEVERAGES."""
        result = classify_taxonomy("Apple Juice 1 L")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.BEVERAGES
        assert result.resolution == "tier0_disambiguation"


class TestWordBoundaryBehavior:
    """Rules should not overmatch due to missing word boundaries."""

    def test_almond_not_in_compliments(self):
        """'almond' inside 'compliments' should not match almonds rule."""
        result = classify_taxonomy("Compliments Organic Peanut Butter")
        # Should not match SNACKS via almonds
        if result.category == ReferenceCategory.SNACKS:
            assert "almond" not in result.matched_rule or False

    def test_tea_not_in_team(self):
        """'tea' in 'team' should not match beverages rule."""
        result = classify_taxonomy("Team Sports Water Bottle")
        # Should not match BEVERAGES via tea
        if result.status == "PASS" and result.category == ReferenceCategory.BEVERAGES:
            assert False, "tea matched inside 'team'"

    def test_bean_not_in_beauty(self):
        """'bean' substring should not match unrelated products."""
        result = classify_taxonomy("Beauty Bar Soap")
        # Should not match food categories
        if result.status == "PASS" and result.category in (
            ReferenceCategory.PRODUCE,
            ReferenceCategory.SNACKS,
        ):
            assert False, "bean matched inside 'beauty'"

    def test_ice_cream_cups_ambiguous(self):
        """Ice cream cups should be disambiguated to ambiguous."""
        result = classify_taxonomy("Ice Cream Cups 18 Pack 75 g")
        assert result.status == "AMBIGUOUS"
        assert result.resolution == "disambiguation_rule"


class TestSpecificPhraseBehavior:
    """Specific multi-word phrases should override generic keyword matches."""

    def test_chocolate_milk_is_dairy(self):
        result = classify_taxonomy("Chocolate Milk 500 ml")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.DAIRY

    def test_orange_juice_is_beverages(self):
        result = classify_taxonomy("Orange Juice 1 L")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.BEVERAGES

    def test_peanut_butter_is_general_grocery(self):
        result = classify_taxonomy("Peanut Butter Creamy 500 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.GENERAL_GROCERY

    def test_ice_cream_is_frozen(self):
        result = classify_taxonomy("Chocolate Ice Cream 1.5L")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.FROZEN
class TestRegressionResolutions:
    """Previously-PASS products must remain classified after taxonomy improvements."""

    def test_canola_olive_oil(self):
        result = classify_taxonomy("Oil Canola & Extra Virgin Olive 1.89 L")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.OILS_VINEGARS

    def test_hot_dog_buns(self):
        result = classify_taxonomy("Hot Dog Top Sliced Buns 12 x 45 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.BAKERY

    def test_potato_deviled_egg_salad(self):
        result = classify_taxonomy("Salad Potato & Deviled Egg 908 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.PRODUCE

    def test_dried_fruit_mango(self):
        result = classify_taxonomy("Dried Fruit Mango 325 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.CONFECTIONERY

    def test_king_oyster_mushrooms(self):
        result = classify_taxonomy("Mushrooms King Oyster 170 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.PRODUCE

    def test_potato_egg_salad_large(self):
        result = classify_taxonomy("Salad Potato & Egg 1.81 kg")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.PRODUCE

    def test_potato_egg_salad_small(self):
        result = classify_taxonomy("Salad Potato & Egg 454 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.PRODUCE

    def test_chicken_stir_fry(self):
        result = classify_taxonomy("Boneless Skinless Chicken Breast Stir Fry")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.MEAT_SEAFOOD

    def test_tomato_soup(self):
        result = classify_taxonomy("Soup Tomato And Herb 540 ml")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.GENERAL_GROCERY

    def test_oyster_mushrooms(self):
        result = classify_taxonomy("Mushrooms Oyster 85 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.PRODUCE

    def test_sweet_potato_soup(self):
        result = classify_taxonomy(
            "Organic Soup Sweet Potato Coconut Curry 398 ml"
        )
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.GENERAL_GROCERY

    def test_potato_deviled_egg_salad_small(self):
        result = classify_taxonomy("Salad Potato & Deviled Egg 454 g")
        assert result.status == "PASS"
        assert result.category == ReferenceCategory.PRODUCE
