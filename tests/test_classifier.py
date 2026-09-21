"""Tests for the classify_product orchestrator."""
from __future__ import annotations

import pytest

from reference_db.classification.classifier import classify_product
from reference_db.taxonomy.reference_taxonomy import ReferenceCategory


class TestClassifierIntegration:
    """End-to-end tests for classify_product."""

    def test_baby_greens_is_food(self):
        """Baby Greens should be classified as food."""
        result = classify_product("Baby Greens Mixed 142 g")
        assert result.domain == "food"

    def test_without_antibiotics_is_food(self):
        """'Without Antibiotics' should not negate food classification."""
        result = classify_product("Compliments Raised Without Antibiotics Chicken Wings Split")
        assert result.domain == "food"

    def test_antibiotic_free_is_food(self):
        """'Antibiotic Free' should not negate food classification."""
        result = classify_product("Antibiotic Free Chicken Wings")
        assert result.domain == "food"

    def test_free_range_is_food(self):
        """'Free Range' should not negate food classification."""
        result = classify_product("Free Range Chicken Breast")
        assert result.domain == "food"

    def test_zero_sugar_is_food(self):
        """'Zero Sugar' should not negate food classification."""
        result = classify_product("Zero Sugar Ginger Ale 2 L (bottle)")
        assert result.domain == "food"

    def test_fabric_softener_is_non_food(self):
        """Fabric softener should be classified as non_food."""
        result = classify_product("Fragrance Free Fabric Softener Sheets 120 Sheets")
        assert result.domain == "non_food"

    def test_chicken_wings_is_food(self):
        """Simple food product should classify as food."""
        result = classify_product("Chicken Wings")
        assert result.domain == "food"

    def test_apple_juice_is_food(self):
        """Juice should classify as food."""
        result = classify_product("Apple Juice")
        assert result.domain == "food"

    def test_sliced_almonds_has_taxonomy(self):
        """Sliced almonds should get a taxonomy category."""
        result = classify_product("Sliced Almonds 250 g")
        assert result.domain == "food"
        assert result.taxonomy is not None
        assert result.taxonomy_status in ("PASS", "AMBIGUOUS")

    def test_water_has_taxonomy(self):
        """Water should classify as BEVERAGES."""
        result = classify_product("Natural Spring Water 4 L")
        assert result.domain == "food"
        assert result.taxonomy is not None
        if result.taxonomy_status == "PASS":
            assert result.taxonomy == "BEVERAGES"
