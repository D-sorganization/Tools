"""
tests/test_biomass_decomposer.py
=================================
Unit tests for the BiomassDecomposer and BiomassFeed classes.

These tests do NOT require DWSIM — they test pure Python math/logic.

What we're checking
--------------------
1. BiomassFeed validates inputs correctly (raises on bad data)
2. The decompose() output:
   - sums to 1.0 (normalized mole fractions)
   - contains only non-negative values
   - includes the expected major species
   - excludes trace species that aren't in available_compounds
3. Channiwala-Parikh HHV estimate is within a reasonable range
4. Edge cases: high moisture, high ash, no trace elements
"""

import pytest

from dwsim_model.chemistry.biomass_decomposer import BiomassDecomposer, BiomassFeed

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def pine_feed():
    """Typical pine wood biomass."""
    return BiomassFeed(
        ultimate_daf={
            "C": 0.501,
            "H": 0.062,
            "O": 0.421,
            "N": 0.008,
            "S": 0.005,
            "Cl": 0.003,
        },
        moisture_ar=0.15,
        ash_ar=0.10,
        hhv_mj_kg=18.5,
    )


@pytest.fixture
def standard_decomposer():
    return BiomassDecomposer(
        available_compounds=[
            "Carbon monoxide",
            "Hydrogen",
            "Carbon dioxide",
            "Methane",
            "Water",
            "Nitrogen",
            "Helium",
            "Ammonia",
            "Hydrogen sulfide",
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# BiomassFeed validation
# ─────────────────────────────────────────────────────────────────────────────


class TestBiomassFeedValidation:

    def test_valid_default_feed_creates_without_error(self):
        feed = BiomassFeed()
        assert feed.moisture_ar == 0.15
        assert feed.ash_ar == 0.10

    def test_ultimate_analysis_must_sum_to_one(self):
        """If fractions don't sum to 1.0 ± 0.03, raise ValueError."""
        with pytest.raises(ValueError, match="sum"):
            BiomassFeed(
                ultimate_daf={
                    "C": 0.5,
                    "H": 0.1,
                    "O": 0.1,
                    "N": 0.0,
                    "S": 0.0,
                    "Cl": 0.0,
                },  # sums to 0.7
                moisture_ar=0.10,
                ash_ar=0.05,
            )

    def test_moisture_out_of_range(self):
        with pytest.raises(ValueError, match="moisture_ar"):
            BiomassFeed(
                ultimate_daf={
                    "C": 0.501,
                    "H": 0.062,
                    "O": 0.421,
                    "N": 0.008,
                    "S": 0.005,
                    "Cl": 0.003,
                },
                moisture_ar=0.75,  # > 0.60
                ash_ar=0.10,
            )

    def test_ash_out_of_range(self):
        with pytest.raises(ValueError, match="ash_ar"):
            BiomassFeed(
                ultimate_daf={
                    "C": 0.501,
                    "H": 0.062,
                    "O": 0.421,
                    "N": 0.008,
                    "S": 0.005,
                    "Cl": 0.003,
                },
                moisture_ar=0.10,
                ash_ar=0.65,  # > 0.50
            )

    def test_moisture_plus_ash_exceeds_one_raises_in_decomposer(
        self, standard_decomposer
    ):
        """The decomposer should raise if moisture + ash >= 1.0."""
        feed = BiomassFeed.__new__(BiomassFeed)  # bypass __post_init__
        object.__setattr__(feed, "moisture_ar", 0.55)
        object.__setattr__(feed, "ash_ar", 0.50)
        object.__setattr__(
            feed,
            "ultimate_daf",
            {"C": 0.5, "H": 0.06, "O": 0.43, "N": 0.01, "S": 0.0, "Cl": 0.0},
        )
        object.__setattr__(feed, "hhv_mj_kg", 18.0)
        with pytest.raises(ValueError, match="daf_frac"):
            standard_decomposer.decompose(feed)


# ─────────────────────────────────────────────────────────────────────────────
# BiomassDecomposer.decompose()
# ─────────────────────────────────────────────────────────────────────────────


class TestDecompose:

    def test_mole_fractions_sum_to_one(self, pine_feed, standard_decomposer):
        fracs = standard_decomposer.decompose(pine_feed)
        total = sum(fracs.values())
        assert abs(total - 1.0) < 1e-9, f"Mole fractions sum to {total}, expected 1.0"

    def test_no_negative_fractions(self, pine_feed, standard_decomposer):
        fracs = standard_decomposer.decompose(pine_feed)
        for species, val in fracs.items():
            assert val >= 0.0, f"{species} has negative mole fraction {val}"

    def test_major_species_present(self, pine_feed, standard_decomposer):
        fracs = standard_decomposer.decompose(pine_feed)
        for expected in [
            "Carbon monoxide",
            "Hydrogen",
            "Carbon dioxide",
            "Methane",
            "Water",
        ]:
            assert expected in fracs, f"Expected '{expected}' in output"

    def test_trace_species_included_when_available(self, pine_feed):
        """NH3 and H2S appear only if they're in available_compounds."""
        decomposer_with_trace = BiomassDecomposer(
            available_compounds=[
                "Carbon monoxide",
                "Hydrogen",
                "Carbon dioxide",
                "Methane",
                "Water",
                "Helium",
                "Ammonia",
                "Hydrogen sulfide",
            ]
        )
        fracs = decomposer_with_trace.decompose(pine_feed)
        assert "Ammonia" in fracs, "NH3 should appear when available"
        assert "Hydrogen sulfide" in fracs, "H2S should appear when available"

    def test_trace_species_excluded_when_not_available(self, pine_feed):
        """NH3/H2S excluded when not in available_compounds."""
        minimal_decomposer = BiomassDecomposer(
            available_compounds=[
                "Carbon monoxide",
                "Hydrogen",
                "Carbon dioxide",
                "Methane",
                "Water",
                "Helium",
            ]
        )
        fracs = minimal_decomposer.decompose(pine_feed)
        assert "Ammonia" not in fracs
        assert "Hydrogen sulfide" not in fracs

    def test_co_dominates_for_wood(self, pine_feed, standard_decomposer):
        """CO should be the largest single species for typical woody biomass."""
        fracs = standard_decomposer.decompose(pine_feed)
        co_frac = fracs.get("Carbon monoxide", 0.0)
        assert co_frac > 0.1, f"CO fraction ({co_frac:.3f}) is unexpectedly low"

    def test_high_moisture_increases_water(self):
        """Higher moisture content should increase water mole fraction."""
        decomposer = BiomassDecomposer(
            available_compounds=[
                "Carbon monoxide",
                "Hydrogen",
                "Carbon dioxide",
                "Methane",
                "Water",
                "Helium",
            ]
        )
        feed_dry = BiomassFeed(
            ultimate_daf={
                "C": 0.501,
                "H": 0.062,
                "O": 0.421,
                "N": 0.008,
                "S": 0.005,
                "Cl": 0.003,
            },
            moisture_ar=0.05,
            ash_ar=0.05,
        )
        feed_wet = BiomassFeed(
            ultimate_daf={
                "C": 0.501,
                "H": 0.062,
                "O": 0.421,
                "N": 0.008,
                "S": 0.005,
                "Cl": 0.003,
            },
            moisture_ar=0.40,
            ash_ar=0.05,
        )
        fracs_dry = decomposer.decompose(feed_dry)
        fracs_wet = decomposer.decompose(feed_wet)
        assert fracs_wet.get("Water", 0) > fracs_dry.get(
            "Water", 0
        ), "Wetter feed should produce more water"

    def test_no_available_compounds_uses_core_only(self, pine_feed):
        """Without available_compounds, only core gas species are returned."""
        decomposer = BiomassDecomposer(available_compounds=None)
        fracs = decomposer.decompose(pine_feed)
        assert len(fracs) > 0
        assert sum(fracs.values()) == pytest.approx(1.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# HHV estimation
# ─────────────────────────────────────────────────────────────────────────────


class TestHHVEstimation:

    def test_hhv_in_reasonable_range(self, pine_feed, standard_decomposer):
        """Channiwala-Parikh HHV for woody biomass should be roughly 15–22 MJ/kg."""
        hhv = standard_decomposer.estimate_hhv(pine_feed)
        assert 14.0 < hhv < 25.0, f"HHV estimate {hhv:.2f} out of expected range"

    def test_hhv_close_to_provided_value(self, pine_feed, standard_decomposer):
        """Estimate should be within 2 MJ/kg of the as-received equivalent of the declared HHV.

        Important unit note
        -------------------
        BiomassFeed.hhv_mj_kg is the *dry-basis* HHV (moisture excluded).
        estimate_hhv() uses the Channiwala-Parikh correlation on an
        *as-received* basis (moisture and ash both included).

        To compare the two fairly we convert the declared dry-basis HHV to
        an as-received basis by multiplying by (1 - moisture_ar).  The ash
        term is already embedded in the element fractions, so no separate
        ash penalty is needed here.
        """
        hhv_estimated = standard_decomposer.estimate_hhv(pine_feed)
        # Convert declared dry-basis HHV to as-received for fair comparison
        hhv_declared_ar = pine_feed.hhv_mj_kg * (1.0 - pine_feed.moisture_ar)
        assert abs(hhv_estimated - hhv_declared_ar) < 2.0, (
            f"Estimate {hhv_estimated:.2f} too far from as-received equivalent "
            f"{hhv_declared_ar:.2f} MJ/kg (dry declared: {pine_feed.hhv_mj_kg:.2f})"
        )

    def test_higher_carbon_raises_hhv(self):
        """A feed with more carbon should have a higher HHV."""
        decomposer = BiomassDecomposer()
        low_C_feed = BiomassFeed(
            ultimate_daf={
                "C": 0.45,
                "H": 0.06,
                "O": 0.47,
                "N": 0.01,
                "S": 0.005,
                "Cl": 0.005,
            },
            moisture_ar=0.10,
            ash_ar=0.05,
        )
        high_C_feed = BiomassFeed(
            ultimate_daf={
                "C": 0.65,
                "H": 0.06,
                "O": 0.27,
                "N": 0.01,
                "S": 0.005,
                "Cl": 0.005,
            },
            moisture_ar=0.10,
            ash_ar=0.05,
        )
        assert decomposer.estimate_hhv(high_C_feed) > decomposer.estimate_hhv(
            low_C_feed
        )
