import pytest
from dwsim_model.chemistry.biomass_decomposer import BiomassDecomposer, BiomassFeed


def test_biomass_feed_init_valid() -> None:
    # A valid instantiation
    feed = BiomassFeed(ultimate_daf={"C": 0.5, "O": 0.5}, moisture_ar=0.2, ash_ar=0.1)
    assert feed.moisture_ar == 0.2
    assert feed.ash_ar == 0.1


def test_biomass_feed_init_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        BiomassFeed(ultimate_daf={"C": 0.5, "O": 0.4})  # Sums to 0.9


def test_biomass_feed_init_invalid_moisture() -> None:
    with pytest.raises(ValueError, match="moisture_ar"):
        BiomassFeed(ultimate_daf={"C": 0.5, "O": 0.5}, moisture_ar=-0.1)

    with pytest.raises(ValueError, match="moisture_ar"):
        BiomassFeed(ultimate_daf={"C": 0.5, "O": 0.5}, moisture_ar=0.7)


def test_biomass_feed_init_invalid_ash() -> None:
    with pytest.raises(ValueError, match="ash_ar"):
        BiomassFeed(ultimate_daf={"C": 0.5, "O": 0.5}, ash_ar=-0.1)

    with pytest.raises(ValueError, match="ash_ar"):
        BiomassFeed(ultimate_daf={"C": 0.5, "O": 0.5}, ash_ar=0.6)


def test_biomass_decomposer_estimate_hhv() -> None:
    feed = BiomassFeed(
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
    )
    dec = BiomassDecomposer()
    hhv = dec.estimate_hhv(feed)

    # Check it calculates a reasonable HHV ~15-20 MJ/kg
    assert 12.0 < hhv < 25.0


def test_biomass_decomposer_decompose_invalid_daf() -> None:
    feed = BiomassFeed(
        ultimate_daf={"C": 1.0}, moisture_ar=0.6, ash_ar=0.45
    )  # daf_frac = -0.05
    # Bypass post_init checks just for this test
    object.__setattr__(feed, "moisture_ar", 0.6)
    object.__setattr__(feed, "ash_ar", 0.45)

    dec = BiomassDecomposer()
    with pytest.raises(ValueError, match="daf_frac .* ≤ 0"):
        dec.decompose(feed)


def test_biomass_decomposer_decompose_core() -> None:
    feed = BiomassFeed(
        ultimate_daf={"C": 0.4, "H": 0.1, "O": 0.5}, moisture_ar=0.1, ash_ar=0.1
    )
    # Available compounds missing trace species
    dec = BiomassDecomposer(available_compounds=[])

    fractions = dec.decompose(feed)

    # Check that keys are what we expect
    assert set(fractions.keys()).issubset(
        {"Carbon monoxide", "Hydrogen", "Carbon dioxide", "Methane", "Water", "Helium"}
    )
    # Check fractions sum to 1.0
    assert sum(fractions.values()) == pytest.approx(1.0)

    # Since C is 0.4 and daf is 0.8, C_ar = 0.32
    # mol_C = 0.32 / 12.011 > 0
    assert fractions.get("Carbon monoxide", 0) > 0


def test_biomass_decomposer_decompose_trace() -> None:
    feed = BiomassFeed(
        ultimate_daf={"C": 0.5, "H": 0.1, "O": 0.35, "N": 0.02, "S": 0.02, "Cl": 0.01},
        moisture_ar=0.0,
        ash_ar=0.0,
    )

    dec = BiomassDecomposer(
        available_compounds=["Ammonia", "Hydrogen sulfide", "Hydrogen chloride"]
    )

    fractions = dec.decompose(feed)
    assert fractions.get("Ammonia", 0) > 0
    assert fractions.get("Hydrogen sulfide", 0) > 0
    assert fractions.get("Hydrogen chloride", 0) > 0


def test_biomass_decomposer_not_enough_H() -> None:
    # Not enough H for N, S, Cl (lots of trace elements, almost no H)
    feed = BiomassFeed(
        ultimate_daf={"C": 0.5, "H": 0.001, "O": 0.1, "N": 0.199, "S": 0.1, "Cl": 0.1},
        moisture_ar=0.0,
        ash_ar=0.0,
    )
    dec = BiomassDecomposer(available_compounds=["Ammonia"])
    fractions = dec.decompose(feed)

    # Should warn and set NH3, H2S, HCl to zero due to negative H
    assert "Ammonia" not in fractions or fractions["Ammonia"] == 0.0
