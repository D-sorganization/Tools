"""Registry completeness: all 7 literature models with citation metadata."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from shared.python.swing_sim.flight import (
    BallFlightModel,
    FlightModelRegistry,
    FlightModelType,
    MacDonaldHanzelyModel,
    WaterlooPennerModel,
)

EXPECTED_MODELS = {
    FlightModelType.WATERLOO_PENNER: "Penner (2003); McPhee et al. (Waterloo)",
    FlightModelType.MACDONALD_HANZELY: "MacDonald & Hanzely (1991)",
    FlightModelType.NATHAN: "Nathan et al. (2018)",
    FlightModelType.BALLANTYNE: "Ballantyne et al. (2012)",
    FlightModelType.JCOLE: "Cole (2016)",
    FlightModelType.ROSPIE_DL: "Rospie & Layton (2014)",
    FlightModelType.CHARRY_L3: "Charry et al. (2017)",
}


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Prevent cross-test pollution of the class-level model dict."""
    FlightModelRegistry.reset()
    yield
    FlightModelRegistry.reset()


@pytest.mark.unit
def test_registry_has_all_seven_models() -> None:
    models = FlightModelRegistry.get_all_models()
    assert len(models) == 7
    assert len(FlightModelType) == 7


@pytest.mark.unit
@pytest.mark.parametrize("model_type", list(FlightModelType))
def test_every_model_resolves_with_metadata(model_type: FlightModelType) -> None:
    model = FlightModelRegistry.get_model(model_type)
    assert isinstance(model, BallFlightModel)
    assert model.name
    assert model.description
    assert model.reference == EXPECTED_MODELS[model_type]


@pytest.mark.unit
def test_named_models_use_dedicated_classes() -> None:
    assert isinstance(
        FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER),
        WaterlooPennerModel,
    )
    assert isinstance(
        FlightModelRegistry.get_model(FlightModelType.MACDONALD_HANZELY),
        MacDonaldHanzelyModel,
    )


@pytest.mark.unit
def test_get_model_requires_model_type() -> None:
    with pytest.raises(ValueError, match="model_type"):
        FlightModelRegistry.get_model(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_reset_forces_reinitialization() -> None:
    first = FlightModelRegistry.get_model(FlightModelType.NATHAN)
    FlightModelRegistry.reset()
    second = FlightModelRegistry.get_model(FlightModelType.NATHAN)
    assert first is not second
