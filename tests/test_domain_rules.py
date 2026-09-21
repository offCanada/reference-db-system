"""Tests for domain classification rules."""
from __future__ import annotations

import pytest

from reference_db.classification.domain_rules import _is_negated


class TestNegationLogic:
    """Tests that _is_negated correctly handles food vs non-food negation.

    Note: _is_negated checks the prefix BEFORE match_start.
    So for "no chicken soup" with chicken at position 3,
    prefix = "no " and the function checks if "no" directly precedes the match.
    """

    def test_no_chicken_is_negated(self):
        """'no chicken' negates chicken (food word, directly after 'no')."""
        assert _is_negated("no chicken soup", 3, 10) is True

    def test_without_antibiotics_not_negated(self):
        """'without antibiotics' is a product attribute, not negation."""
        # "chicken" is at position 22 in "without antibiotics chicken wings"
        assert _is_negated("without antibiotics chicken wings", 22, 29) is False

    def test_antibiotic_free_not_negated(self):
        """'antibiotic free' is a product attribute, not negation."""
        # "chicken" is at position 14 in "antibiotic free chicken"
        assert _is_negated("antibiotic free chicken", 14, 21) is False

    def test_free_range_not_negated(self):
        """'free range' is a farming attribute, not negation."""
        # "chicken" is at position 11 in "free range chicken"
        assert _is_negated("free range chicken", 11, 18) is False

    def test_zero_sugar_not_negated(self):
        """'zero sugar' is a nutritional attribute, not negation."""
        # "ginger" is at position 10 in "zero sugar ginger ale"
        assert _is_negated("zero sugar ginger ale", 10, 16) is False

    def test_gluten_free_not_negated(self):
        """'gluten free bread' is still bread, not negated bread."""
        # "bread" is at position 12 in "gluten free bread"
        assert _is_negated("gluten free bread", 12, 17) is False

    def test_fragrance_free_not_negated(self):
        """'fragrance free' is a product attribute."""
        # "fabric" is at position 14 in "fragrance free fabric softener"
        assert _is_negated("fragrance free fabric softener", 14, 20) is False

    def test_non_food_prefix(self):
        """'non-' prefix is always negation."""
        # "dairy" is at position 4 in "non-dairy creamer"
        assert _is_negated("non-dairy creamer", 4, 9) is True

    def test_no_fish_is_negated(self):
        """'no fish' negates fish (food word, directly after 'no')."""
        assert _is_negated("no fish sticks", 3, 7) is True
