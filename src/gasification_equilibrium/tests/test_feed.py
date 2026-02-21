"""Tests for the feed composition builder module.

Tests verify:
1. FeedComposition creation and conversion
2. Mass fraction to moles conversion
3. Injection element contributions
4. OxidantConfig (pure O2 vs air)
5. ProcessInputs with all injection types
6. build_total_feed aggregation
7. Feed presets
"""

import pytest

from gasification_equilibrium.python.feed import (
    AIR_N2_FRACTION,
    AIR_O2_FRACTION,
    COMPOUND_ELEMENTS,
    FEED_PRESETS,
    FeedComposition,
    Injection,
    OxidantConfig,
    ProcessInputs,
    _merge_elements,
    build_total_feed,
    feed_from_preset,
)


class TestFeedComposition:
    """Test FeedComposition dataclass."""

    def test_default_zeros(self):
        f = FeedComposition()
        assert f.C == 0.0
        assert f.H == 0.0
        assert f.O == 0.0
        assert f.N == 0.0
        assert f.S == 0.0

    def test_custom_values(self):
        f = FeedComposition(C=1.0, H=2.0, O=0.5, N=0.1, S=0.05)
        assert f.C == 1.0
        assert f.H == 2.0

    def test_as_dict_excludes_zeros(self):
        f = FeedComposition(C=1.0, H=2.0, O=0.0)
        d = f.as_dict()
        assert "C" in d
        assert "H" in d
        assert "O" not in d

    def test_as_dict_values(self):
        f = FeedComposition(C=1.0, H=2.0, O=0.5, N=0.1, S=0.05)
        d = f.as_dict()
        assert d["C"] == 1.0
        assert d["S"] == 0.05

    def test_total_moles(self):
        f = FeedComposition(C=1.0, H=2.0, O=0.5)
        assert abs(f.total_moles() - 3.5) < 1e-10

    def test_from_dict(self):
        f = FeedComposition.from_dict({"C": 1.0, "H": 4.0})
        assert f.C == 1.0
        assert f.H == 4.0
        assert f.O == 0.0

    def test_from_dict_ignores_unknown(self):
        f = FeedComposition.from_dict({"C": 1.0, "X": 99.0})
        assert f.C == 1.0
        assert f.O == 0.0


class TestFeedFromMassFractions:
    """Test mass-to-moles conversion."""

    def test_pure_carbon(self):
        f = FeedComposition.from_mass_fractions({"C": 1.0})
        assert f.C > 0
        # 1 kg C / 12.011 g/mol * 1000 g/kg
        expected = 1.0 / (12.011 / 1000.0)
        assert abs(f.C - expected) < 0.1

    def test_ash_excluded(self):
        f = FeedComposition.from_mass_fractions({"C": 0.5, "Ash": 0.5})
        d = f.as_dict()
        assert "Ash" not in d
        assert f.C > 0

    def test_coal_composition(self):
        fracs = {"C": 0.75, "H": 0.05, "O": 0.08, "N": 0.015, "S": 0.01, "Ash": 0.095}
        f = FeedComposition.from_mass_fractions(fracs)
        assert f.C > 0
        assert f.H > 0
        assert f.O > 0
        assert f.N > 0
        assert f.S > 0

    def test_empty_mass_fractions_returns_zeros(self):
        f = FeedComposition.from_mass_fractions({})
        assert f.total_moles() == 0.0

    def test_hydrogen_has_more_moles_than_carbon(self):
        """Due to low atomic weight, H should have more moles per kg."""
        fracs = {"C": 0.50, "H": 0.50}
        f = FeedComposition.from_mass_fractions(fracs)
        assert f.H > f.C


class TestInjection:
    """Test Injection dataclass."""

    def test_zero_flow_no_contribution(self):
        inj = Injection("test", flow=0.0, elements={"H": 2, "O": 1})
        contrib = inj.element_contribution()
        assert all(v == 0 for v in contrib.values())

    def test_steam_injection(self):
        inj = Injection("steam", flow=1.0, elements=COMPOUND_ELEMENTS["H2O"])
        contrib = inj.element_contribution()
        assert contrib["H"] == 2.0
        assert contrib["O"] == 1.0

    def test_ch4_injection(self):
        inj = Injection("CH4", flow=2.0, elements=COMPOUND_ELEMENTS["CH4"])
        contrib = inj.element_contribution()
        assert contrib["C"] == 2.0
        assert contrib["H"] == 8.0

    def test_c3h8_injection(self):
        inj = Injection("C3H8", flow=1.0, elements=COMPOUND_ELEMENTS["C3H8"])
        contrib = inj.element_contribution()
        assert contrib["C"] == 3.0
        assert contrib["H"] == 8.0

    def test_natural_gas_injection(self):
        inj = Injection("NG", flow=1.0, elements=COMPOUND_ELEMENTS["natural_gas"])
        contrib = inj.element_contribution()
        assert contrib["C"] == pytest.approx(1.05)
        assert contrib["H"] == pytest.approx(4.16)
        assert contrib["N"] == pytest.approx(0.04)


class TestOxidantConfig:
    """Test OxidantConfig for pure O2 and air modes."""

    def test_zero_flow_no_contribution(self):
        ox = OxidantConfig(use_air=False, o2_flow=0.0)
        assert ox.element_contribution() == {}

    def test_pure_o2(self):
        ox = OxidantConfig(use_air=False, o2_flow=1.0)
        contrib = ox.element_contribution()
        assert contrib["O"] == 2.0
        assert "N" not in contrib

    def test_air_mode_adds_nitrogen(self):
        ox = OxidantConfig(use_air=True, o2_flow=1.0)
        contrib = ox.element_contribution()
        assert contrib["O"] == 2.0
        assert "N" in contrib
        assert contrib["N"] > 0

    def test_air_n2_o2_ratio(self):
        """Air has N2/O2 ~ 3.73 by moles, so N atoms/O atoms ~ 3.73."""
        ox = OxidantConfig(use_air=True, o2_flow=1.0)
        contrib = ox.element_contribution()
        air_moles = 1.0 / AIR_O2_FRACTION
        expected_n = air_moles * AIR_N2_FRACTION * 2
        assert abs(contrib["N"] - expected_n) < 0.01

    def test_air_mode_double_flow(self):
        ox = OxidantConfig(use_air=True, o2_flow=2.0)
        contrib = ox.element_contribution()
        assert contrib["O"] == 4.0


class TestProcessInputs:
    """Test ProcessInputs with all injection streams."""

    def test_default_all_zero_flow(self):
        pi = ProcessInputs()
        assert pi.oxidant.o2_flow == 0.0
        assert pi.steam.flow == 0.0
        assert pi.n2_purge.flow == 0.0
        assert pi.ch4_injection.flow == 0.0
        assert pi.c3h8_injection.flow == 0.0
        assert pi.natural_gas.flow == 0.0

    def test_all_injections_returns_list(self):
        pi = ProcessInputs()
        injections = pi.all_injections()
        assert len(injections) == 5

    def test_modify_stream_flow(self):
        pi = ProcessInputs()
        pi.steam.flow = 2.0
        assert pi.steam.flow == 2.0

    def test_default_feed_rate(self):
        pi = ProcessInputs()
        assert pi.feed_rate_kg_hr == 100.0


class TestBuildTotalFeed:
    """Test build_total_feed aggregation."""

    def test_base_only_no_injections(self):
        base = FeedComposition(C=1.0, H=1.0, O=0.5)
        pi = ProcessInputs()
        total = build_total_feed(base, pi)
        assert total["C"] == 1.0
        assert total["H"] == 1.0
        assert total["O"] == 0.5

    def test_with_steam_injection(self):
        base = FeedComposition(C=1.0, O=0.5)
        pi = ProcessInputs()
        pi.steam.flow = 1.0
        total = build_total_feed(base, pi)
        assert total["C"] == 1.0
        assert total["H"] == 2.0  # from steam
        assert total["O"] == 1.5  # 0.5 base + 1.0 steam

    def test_with_pure_o2(self):
        base = FeedComposition(C=1.0)
        pi = ProcessInputs()
        pi.oxidant.o2_flow = 0.5
        total = build_total_feed(base, pi)
        assert total["O"] == 1.0  # 0.5 mol O2 = 1.0 mol O

    def test_with_air(self):
        base = FeedComposition(C=1.0)
        pi = ProcessInputs()
        pi.oxidant.use_air = True
        pi.oxidant.o2_flow = 0.5
        total = build_total_feed(base, pi)
        assert total["O"] == 1.0  # same O as pure O2
        assert "N" in total  # but also has N from air
        assert total["N"] > 0

    def test_with_n2_purge(self):
        base = FeedComposition(C=1.0)
        pi = ProcessInputs()
        pi.n2_purge.flow = 1.0
        total = build_total_feed(base, pi)
        assert total["N"] == 2.0  # 1 mol N2 = 2 mol N

    def test_with_ch4_injection(self):
        base = FeedComposition(C=1.0)
        pi = ProcessInputs()
        pi.ch4_injection.flow = 1.0
        total = build_total_feed(base, pi)
        assert total["C"] == 2.0
        assert total["H"] == 4.0

    def test_with_c3h8_injection(self):
        base = FeedComposition(C=1.0)
        pi = ProcessInputs()
        pi.c3h8_injection.flow = 1.0
        total = build_total_feed(base, pi)
        assert total["C"] == 4.0  # 1 base + 3 from C3H8
        assert total["H"] == 8.0

    def test_with_natural_gas(self):
        base = FeedComposition(C=1.0)
        pi = ProcessInputs()
        pi.natural_gas.flow = 1.0
        total = build_total_feed(base, pi)
        assert total["C"] == pytest.approx(2.05)

    def test_multiple_injections_combine(self):
        base = FeedComposition(C=1.0, H=1.0, O=0.5)
        pi = ProcessInputs()
        pi.steam.flow = 1.0
        pi.oxidant.o2_flow = 0.5
        pi.ch4_injection.flow = 0.5
        total = build_total_feed(base, pi)
        # C: 1.0 base + 0.5 CH4 = 1.5
        assert total["C"] == pytest.approx(1.5)
        # H: 1.0 base + 2.0 steam + 2.0 CH4 = 5.0
        assert total["H"] == pytest.approx(5.0)
        # O: 0.5 base + 1.0 steam + 1.0 O2 = 2.5
        assert total["O"] == pytest.approx(2.5)

    def test_all_values_nonnegative(self):
        base = FeedComposition(C=1.0, H=1.0, O=0.5, N=0.01, S=0.005)
        pi = ProcessInputs()
        pi.steam.flow = 2.0
        pi.oxidant.use_air = True
        pi.oxidant.o2_flow = 0.5
        pi.ch4_injection.flow = 0.3
        pi.c3h8_injection.flow = 0.1
        pi.natural_gas.flow = 0.2
        pi.n2_purge.flow = 0.5
        total = build_total_feed(base, pi)
        for val in total.values():
            assert val >= 0


class TestMergeElements:
    """Test _merge_elements helper."""

    def test_merge_into_empty(self):
        target = {}
        _merge_elements(target, {"C": 1.0, "H": 2.0})
        assert target == {"C": 1.0, "H": 2.0}

    def test_merge_additive(self):
        target = {"C": 1.0, "O": 0.5}
        _merge_elements(target, {"C": 0.5, "H": 2.0})
        assert target["C"] == 1.5
        assert target["O"] == 0.5
        assert target["H"] == 2.0


class TestFeedPresets:
    """Test feed preset compositions."""

    def test_all_presets_have_description(self):
        for name, preset in FEED_PRESETS.items():
            assert "description" in preset, f"{name} missing description"

    def test_all_presets_have_feed_data(self):
        for name, preset in FEED_PRESETS.items():
            assert (
                "mass_fractions" in preset or "elements" in preset
            ), f"{name} needs 'mass_fractions' or 'elements'"

    def test_mass_fractions_sum_to_one(self):
        for name, preset in FEED_PRESETS.items():
            if "mass_fractions" in preset:
                total = sum(preset["mass_fractions"].values())
                assert (
                    abs(total - 1.0) < 0.01
                ), f"{name} mass fractions sum to {total}, expected ~1.0"

    def test_feed_from_preset_bituminous(self):
        f = feed_from_preset("Bituminous Coal")
        assert f.C > 0
        assert f.H > 0
        assert f.O > 0

    def test_feed_from_preset_custom(self):
        f = feed_from_preset("Custom")
        assert f.C == 1.0
        assert f.H == 1.0
        assert f.O == 0.5

    def test_feed_from_preset_natural_gas(self):
        f = feed_from_preset("Natural Gas (CH4)")
        assert f.C == 1.0
        assert f.H == 4.0

    def test_all_presets_produce_valid_feed(self):
        for name in FEED_PRESETS:
            f = feed_from_preset(name)
            assert f.total_moles() > 0, f"Preset '{name}' produces zero-moles feed"

    def test_preset_not_found_raises(self):
        with pytest.raises(KeyError):
            feed_from_preset("NonexistentFuel")


class TestCompoundElements:
    """Test COMPOUND_ELEMENTS lookup table."""

    def test_h2o_composition(self):
        assert COMPOUND_ELEMENTS["H2O"] == {"H": 2, "O": 1}

    def test_o2_composition(self):
        assert COMPOUND_ELEMENTS["O2"] == {"O": 2}

    def test_n2_composition(self):
        assert COMPOUND_ELEMENTS["N2"] == {"N": 2}

    def test_ch4_composition(self):
        assert COMPOUND_ELEMENTS["CH4"] == {"C": 1, "H": 4}

    def test_c3h8_composition(self):
        assert COMPOUND_ELEMENTS["C3H8"] == {"C": 3, "H": 8}

    def test_natural_gas_composition(self):
        ng = COMPOUND_ELEMENTS["natural_gas"]
        assert "C" in ng
        assert "H" in ng
        assert "N" in ng
