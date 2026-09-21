"""Integration tests for Phase 3 pipeline outputs."""
from __future__ import annotations

import pandas as pd
import pytest

from reference_db.phase_3.pipeline import run_phase3


def _sample_rows(n: int = 50) -> list[dict]:
    df = pd.read_parquet("data/phase_2/standardized_products.parquet")
    return df.head(n).to_dict(orient="records")


class TestPhase3Pipeline:
    """Phase 3 pipeline should produce correct taxonomy counts."""

    def test_pipeline_runs(self):
        """Phase 3 pipeline should run without errors."""
        result = run_phase3(_sample_rows())
        assert result is not None
        assert "product_groups" in result

    def test_pipeline_no_new_unknown_domain(self):
        """Pipeline should not produce new unknown domain products."""
        result = run_phase3(_sample_rows())
        assert result is not None
        unknown = [
            r
            for r in result["product_classification"]
            if r["domain"] == "unknown"
        ]
        assert len(unknown) == 0


class TestTaxonomyCounts:
    """Verify taxonomy classification counts are within expected ranges."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.df = pd.read_parquet("data/phase_3/product_classification.parquet")

    def test_total_count(self):
        assert len(self.df) > 0

    def test_no_new_conflicts(self):
        """Number of conflict AMBIGUOUS should not increase."""
        ambiguous = self.df[self.df["taxonomy_status"] == "AMBIGUOUS"]
        conflicts = ambiguous[ambiguous["taxonomy_resolution"] == "conflict"]
        # Should be <= 937 (the original count)
        assert len(conflicts) <= 937, f"Conflicts increased to {len(conflicts)}"

    def test_no_evidence_reduced(self):
        """Number of no_evidence AMBIGUOUS should decrease."""
        ambiguous = self.df[self.df["taxonomy_status"] == "AMBIGUOUS"]
        no_evidence = ambiguous[ambiguous["taxonomy_resolution"] == "no_evidence"]
        # Should be < 621 (the original count)
        assert len(no_evidence) < 621, f"no_evidence still at {len(no_evidence)}"

    def test_unknown_domain_stable(self):
        """Number of unknown domain should not increase."""
        unknown = self.df[self.df["domain"] == "unknown"]
        assert len(unknown) <= 14, f"Unknown domain increased to {len(unknown)}"
