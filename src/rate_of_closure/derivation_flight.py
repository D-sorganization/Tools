"""Ball-flight derivation content for the Calculation Description tab.

Sectioned V4 coverage (#4120): the flight equations of motion with
drag / lift (Magnus), the ACTIVE literature model's coefficient law
with its citation (pulled live from the ``swing_sim.flight`` registry
metadata), and spin decay. The coefficient step substitutes the
selected model's actual parameters, so switching the flight model in
the Simulation tab rewrites this section.
"""

from __future__ import annotations

from shared.python.swing_sim.flight.registry import (
    _CONSTANT_COEFFICIENT_SPECS,
    FlightModelRegistry,
    FlightModelType,
)

from ._contracts import ensure
from .derivation import DerivationStep

__all__ = ["flight_steps"]


def _coefficient_step(model_type: FlightModelType) -> DerivationStep:
    """The active model's coefficient law, cited from the registry."""
    model = FlightModelRegistry.get_model(model_type)
    if model_type is FlightModelType.WATERLOO_PENNER:
        latex = (
            r"$C_d = c_{d0} + c_{d1} s + c_{d2} s^2,\qquad "
            r"C_l = \min(C_{l,max},\ c_{l1}\,s^{c_{l2}}),\qquad "
            r"s = \frac{R\,\omega}{v}$"
        )
        values = (
            r"$c_{d} = (0.21,\ 0.05,\ 0.02),\ "
            r"c_{l} = (0.70,\ 0.645),\ "
            r"\mathrm{spin\ ratio\ } s\ \mathrm{drives\ both}$"
        )
    elif model_type is FlightModelType.MACDONALD_HANZELY:
        latex = (
            r"$C_d = const,\qquad C_l \propto s,\qquad "
            r"\omega(t) = \omega_0\, e^{-t/\tau}$"
        )
        values = r"$\mathrm{exponential\ spin\ decay\ sets\ late\ lift}$"
    else:
        spec = _CONSTANT_COEFFICIENT_SPECS[model_type]
        latex = (
            r"$C_d = const,\qquad C_l = const,\qquad "
            r"\omega(t) = \omega_0\,e^{-\lambda t}$"
        )
        values = (
            rf"$C_d = {spec.cd:.2f},\ C_l = {spec.cl:.2f},\ "
            rf"\lambda = {spec.spin_decay:.2f}\ \mathrm{{s^{{-1}}}}$"
        )
    return DerivationStep(
        title=f"Active Model Coefficient Law — {model.name}",
        latex=latex,
        values=values,
        narrative=(
            f"{model.description}. The literature flight models differ "
            "mainly in how the drag and lift coefficients depend on the "
            "spin ratio s = Rω/v. "
            f"Citation: {model.reference} "
            "(swing_sim.flight registry metadata)."
        ),
    )


def flight_steps(flight_model: str) -> tuple[DerivationStep, ...]:
    """Ball-flight derivation steps for the selected literature model.

    Args:
        flight_model: Registry key of the active model (e.g.
            ``"waterloo_penner"``).

    Returns:
        Ordered steps: the flight EOM, the active model's coefficient
        law with its citation, and spin decay.
    """
    model_type = FlightModelType(flight_model)

    steps = (
        DerivationStep(
            title="Equations of Motion — Drag, Lift, Gravity",
            latex=(
                r"$m\dot{\vec{v}} = -\frac{1}{2}\rho A C_d "
                r"|\vec{v}|\,\vec{v} + \frac{1}{2}\rho A C_l "
                r"|\vec{v}|^2\,(\hat{\omega} \times \hat{v}) + m\vec{g}$"
            ),
            values=(
                r"$A = \pi R^2,\ \rho \approx 1.225\ \mathrm{kg/m^3},\ "
                r"\vec{g} = (0,\ -9.81,\ 0)\ \mathrm{m/s^2}$"
            ),
            narrative=(
                "Three forces act in flight: drag opposing the velocity, "
                "the Magnus lift perpendicular to both the spin axis and "
                "the velocity (backspin lifts, tilted spin curves the "
                "shot sideways), and gravity. The trajectory integrates "
                "this ODE with scipy RK45 to a terminal ground event "
                "(swing_sim.flight.models base loop)."
            ),
        ),
        _coefficient_step(model_type),
        DerivationStep(
            title="Spin Decay and the Terminal Ground Event",
            latex=(
                r"$\omega(t) = \omega_0\,e^{-\lambda t},\qquad "
                r"y(t^*) = 0 \Rightarrow \mathrm{carry},\ "
                r"\mathrm{apex},\ \mathrm{landing\ angle}$"
            ),
            values=(
                r"$\mathrm{carry} = x(t^*),\ \mathrm{lateral} = z(t^*),\ "
                r"\mathrm{landing} = \arctan\!\frac{-v_y(t^*)}"
                r"{\sqrt{v_x^2 + v_z^2}}$"
            ),
            narrative=(
                "Aerodynamic torque bleeds spin during flight (modeled as "
                "an exponential decay in the spin-decay model families), "
                "reducing late lift. Integration stops at the ground "
                "event; the reported carry, apex, flight time, landing "
                "angle, and lateral offset are read off that terminal "
                "state (swing_sim.flight metrics)."
            ),
        ),
    )
    ensure(len(steps) == 3, "flight derivation must cover EOM + law + decay")
    return steps
