"""Gasification Equilibrium Calculator - Interactive Application.

SRP: Tab layout, widget wiring, and callback routing only.
     All plotting delegated to plots.py, theming to theme.py,
     feed logic to feed.py, calculations to engine.py.

Single-page, 4-tab interface with process injection controls.
"""

from collections.abc import Callable
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import warnings

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

from . import plots  # noqa: E402
from .engine import GasificationEngine  # noqa: E402
from .feed import (  # noqa: E402
    FEED_PRESETS,
    FeedComposition,
    ProcessInputs,
)
from .theme import COLORS, apply_theme, style_slider  # noqa: E402
from .thermo_data import SPECIES_DB  # noqa: E402

# ─── Surface parameter constants ────────────────────────────────────────────────

LBL_STEAM_CARBON = "Steam/Carbon"
LBL_O2_CARBON = "O\u2082/Carbon"
LBL_PRESSURE = "Pressure [atm]"

_SURFACE_PARAMS = [
    (LBL_STEAM_CARBON, "steam_carbon_ratio", (0.0, 3.0)),
    (LBL_O2_CARBON, "oxygen_carbon_ratio", (0.0, 1.5)),
    (LBL_PRESSURE, "pressure", (101325 * 0.5, 101325 * 30)),
]

SURFACE_PARAM_MAP = {lbl: (key, rng) for lbl, key, rng in _SURFACE_PARAMS}
SURFACE_PARAM_LABELS = {key: lbl for lbl, key, _ in _SURFACE_PARAMS}


# ─── Widget factory helpers ─────────────────────────────────────────────────────


def make_slider(
    fig: Any,
    rect: list[float],
    label: str,
    vmin: float,
    vmax: float,
    vinit: float,
    vstep: float,
    color: str,
    callback: Callable[..., Any],
) -> Slider:
    """Create a styled slider. Returns the Slider widget."""
    ax = fig.add_axes(rect)
    ax.set_facecolor(COLORS["panel"])
    sl = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=vstep, color=color)
    style_slider(sl, color)
    sl.on_changed(callback)
    return sl


def make_button(
    fig: Any, rect: list[float], label: str, color: str, callback: Callable[..., Any]
) -> Button:
    """Create a styled button. Returns the Button widget."""
    ax = fig.add_axes(rect)
    btn = Button(ax, label, color=color, hovercolor=COLORS["grid"])
    btn.label.set_color(COLORS["bg"])
    btn.label.set_fontweight("bold")
    btn.on_clicked(callback)
    return btn


# ─── Application state ─────────────────────────────────────────────────────────


class AppState:
    """Mutable application state, separated from UI.

    SRP: Holds values only. No logic, no widgets.
    """

    def __init__(self) -> None:
        self.base_feed = FeedComposition(C=1.0, H=1.0, O=0.5, N=0.0, S=0.0)
        self.process = ProcessInputs()
        self.temperature = 1000.0
        self.pressure = 101325.0
        self.selected_species = ["H2", "CO", "CO2", "H2O", "CH4"]
        self.surface_param = "steam_carbon_ratio"
        self.last_sweep: list[Any] | None = None
        self.last_surface: dict[str, Any] | None = None
        self.last_result: Any = None


# ─── Tab builders ───────────────────────────────────────────────────────────────


def build_single_point_tab(
    fig: Any, state: AppState, on_recalc: Callable[[], None]
) -> dict[str, Any]:
    """Build Tab 1: single-point equilibrium with process controls.

    Returns dict of {name: widget_or_axes} for visibility toggling.
    """
    widgets = {}

    widgets["ax_bar"] = fig.add_axes([0.06, 0.35, 0.38, 0.55])
    widgets["ax_pie"] = fig.add_axes([0.50, 0.42, 0.22, 0.48])
    widgets["ax_info"] = fig.add_axes([0.74, 0.08, 0.24, 0.82])
    widgets["ax_info"].set_facecolor(COLORS["panel"])

    # Process condition sliders (left column)
    x, w = 0.12, 0.28
    widgets["sl_T"] = make_slider(
        fig,
        [x, 0.26, w, 0.022],
        "T [K]",
        300,
        2000,
        1000,
        10,
        COLORS["accent"],
        lambda v: _set_and_recalc(state, "temperature", v, on_recalc),
    )
    widgets["sl_P"] = make_slider(
        fig,
        [x, 0.23, w, 0.022],
        "P [atm]",
        0.1,
        50,
        1.0,
        0.1,
        COLORS["accent2"],
        lambda v: _set_pressure(state, v, on_recalc),
    )

    # Oxidant controls
    widgets["sl_O2"] = make_slider(
        fig,
        [x, 0.19, w, 0.022],
        "O\u2082 [mol]",
        0,
        3,
        0,
        0.05,
        "#e040fb",
        lambda v: _set_oxidant(state, v, on_recalc),
    )

    # Steam injection
    widgets["sl_steam"] = make_slider(
        fig,
        [x, 0.15, w, 0.022],
        "Steam",
        0,
        5,
        0,
        0.1,
        COLORS["accent3"],
        lambda v: _set_injection(state, "steam", v, on_recalc),
    )

    # N2 purge
    widgets["sl_N2"] = make_slider(
        fig,
        [x, 0.11, w, 0.022],
        "N\u2082 purge",
        0,
        3,
        0,
        0.1,
        COLORS["error"],
        lambda v: _set_injection(state, "n2_purge", v, on_recalc),
    )

    # CH4 injection
    widgets["sl_CH4"] = make_slider(
        fig,
        [x, 0.07, w, 0.022],
        "CH\u2084 inj",
        0,
        3,
        0,
        0.1,
        COLORS["warning"],
        lambda v: _set_injection(state, "ch4_injection", v, on_recalc),
    )

    # C3H8 injection
    widgets["sl_C3H8"] = make_slider(
        fig,
        [x, 0.03, w, 0.022],
        "C\u2083H\u2088",
        0,
        2,
        0,
        0.05,
        COLORS["success"],
        lambda v: _set_injection(state, "c3h8_injection", v, on_recalc),
    )

    # Air/O2 toggle + natural gas
    ax_radio = fig.add_axes([0.44, 0.03, 0.10, 0.10])
    ax_radio.set_facecolor(COLORS["panel"])
    widgets["radio_ox"] = RadioButtons(ax_radio, ["Pure O\u2082", "Air"], active=0)
    for lb in widgets["radio_ox"].labels:
        lb.set_color(COLORS["text"])
        lb.set_fontsize(8)
    widgets["radio_ox"].on_clicked(lambda lbl: _set_air_mode(state, lbl, on_recalc))

    widgets["sl_NG"] = make_slider(
        fig,
        [x, -0.005, w, 0.022],
        "Nat Gas",
        0,
        3,
        0,
        0.1,
        "#64dd17",
        lambda v: _set_injection(state, "natural_gas", v, on_recalc),
    )

    return widgets


def build_sweep_tab(
    fig: Any, state: AppState, on_sweep: Callable[[], None]
) -> dict[str, Any]:
    """Build Tab 2: temperature sweep with species selection."""
    widgets = {}

    widgets["ax_comp"] = fig.add_axes([0.07, 0.52, 0.55, 0.38])
    widgets["ax_metrics"] = fig.add_axes([0.07, 0.08, 0.55, 0.35])

    widgets["btn_run"] = make_button(
        fig,
        [0.72, 0.08, 0.20, 0.05],
        "RUN SWEEP",
        COLORS["accent"],
        lambda _: on_sweep(),
    )

    widgets["sl_Ts"] = make_slider(
        fig,
        [0.77, 0.35, 0.18, 0.022],
        "T start",
        200,
        1500,
        400,
        50,
        COLORS["accent"],
        lambda _: None,
    )
    widgets["sl_Te"] = make_slider(
        fig,
        [0.77, 0.30, 0.18, 0.022],
        "T end",
        500,
        2500,
        1600,
        50,
        COLORS["accent2"],
        lambda _: None,
    )
    widgets["sl_np"] = make_slider(
        fig,
        [0.77, 0.25, 0.18, 0.022],
        "Points",
        10,
        100,
        50,
        5,
        COLORS["accent3"],
        lambda _: None,
    )
    widgets["sl_Pp"] = make_slider(
        fig,
        [0.77, 0.20, 0.18, 0.022],
        "P [atm]",
        0.1,
        50,
        1.0,
        0.5,
        COLORS["warning"],
        lambda _: None,
    )

    sp_labels = [
        "H\u2082",
        "CO",
        "CO\u2082",
        "H\u2082O",
        "CH\u2084",
        "N\u2082",
        "C\u2082H\u2084",
        "C\u2082H\u2086",
        "C\u2083H\u2088",
        "C(s)",
    ]
    sp_keys = ["H2", "CO", "CO2", "H2O", "CH4", "N2", "C2H4", "C2H6", "C3H8", "C_solid"]
    widgets["species_map"] = dict(zip(sp_labels, sp_keys, strict=True))
    defaults = [k in state.selected_species for k in sp_keys]

    ax_chk = fig.add_axes([0.68, 0.50, 0.28, 0.40])
    ax_chk.set_facecolor(COLORS["panel"])
    widgets["check"] = CheckButtons(ax_chk, sp_labels, defaults)
    for lb in widgets["check"].labels:
        lb.set_color(COLORS["text"])
        lb.set_fontsize(9)
    for _i, rect in enumerate(widgets["check"].rectangles):
        rect.set_edgecolor(COLORS.get("accent", "#888"))
        rect.set_linewidth(2)

    return widgets


def build_surface_tab(fig: Any, on_surface: Callable[[], None]) -> dict[str, Any]:
    """Build Tab 3: 3D surface plots."""
    widgets = {}

    widgets["ax_3d"] = fig.add_axes([0.02, 0.12, 0.55, 0.78], projection="3d")
    widgets["ax_3d"].set_facecolor(COLORS["panel"])
    widgets["ax_contour"] = fig.add_axes([0.60, 0.52, 0.36, 0.38])

    ax_r = fig.add_axes([0.62, 0.22, 0.32, 0.25])
    ax_r.set_facecolor(COLORS["panel"])
    widgets["radio_param"] = RadioButtons(
        ax_r, [LBL_STEAM_CARBON, LBL_O2_CARBON, LBL_PRESSURE], active=0
    )
    for lb in widgets["radio_param"].labels:
        lb.set_color(COLORS["text"])
        lb.set_fontsize(9)

    ax_sp = fig.add_axes([0.62, 0.08, 0.15, 0.12])
    ax_sp.set_facecolor(COLORS["panel"])
    widgets["radio_species"] = RadioButtons(
        ax_sp, ["H\u2082", "CO", "CO\u2082", "CH\u2084", "H\u2082/CO"], active=0
    )
    for lb in widgets["radio_species"].labels:
        lb.set_color(COLORS["text"])
        lb.set_fontsize(9)

    widgets["btn_run"] = make_button(
        fig,
        [0.82, 0.08, 0.14, 0.06],
        "GENERATE\nSURFACE",
        COLORS["accent2"],
        lambda _: on_surface(),
    )
    widgets["btn_run"].label.set_fontsize(9)

    widgets["cbar"] = None
    return widgets


def build_feed_tab(
    fig: Any, state: AppState, on_apply: Callable[[], None]
) -> dict[str, Any]:
    """Build Tab 4: CHONS feed editor with presets and injection summary."""
    widgets = {}

    widgets["ax_comp"] = fig.add_axes([0.06, 0.42, 0.38, 0.48])
    widgets["ax_eq"] = fig.add_axes([0.50, 0.42, 0.46, 0.48])

    # Preset radio
    ax_pre = fig.add_axes([0.06, 0.06, 0.22, 0.30])
    ax_pre.set_facecolor(COLORS["panel"])
    widgets["radio_preset"] = RadioButtons(ax_pre, list(FEED_PRESETS.keys()), active=7)
    for lb in widgets["radio_preset"].labels:
        lb.set_color(COLORS["text"])
        lb.set_fontsize(7)

    # CHONS sliders
    elements = ["C", "H", "O", "N", "S"]
    colors = [
        COLORS["accent2"],
        COLORS["accent"],
        COLORS["success"],
        COLORS["accent3"],
        COLORS["warning"],
    ]
    defaults = [
        state.base_feed.C,
        state.base_feed.H,
        state.base_feed.O,
        state.base_feed.N,
        state.base_feed.S,
    ]
    widgets["elem_sliders"] = {}
    for i, (elem, default, clr) in enumerate(
        zip(elements, defaults, colors, strict=True)
    ):
        widgets["elem_sliders"][elem] = make_slider(
            fig,
            [0.38, 0.31 - i * 0.05, 0.25, 0.022],
            elem,
            0.0,
            10.0,
            default,
            0.05,
            clr,
            lambda v, e=elem: _set_feed_element(state, e, v),
        )

    widgets["btn_apply"] = make_button(
        fig,
        [0.70, 0.06, 0.18, 0.05],
        "APPLY FEED",
        COLORS["success"],
        lambda _: on_apply(),
    )

    widgets["ax_desc"] = fig.add_axes([0.70, 0.14, 0.26, 0.18])
    widgets["ax_desc"].set_facecolor(COLORS["panel"])
    widgets["ax_desc"].set_xticks([])
    widgets["ax_desc"].set_yticks([])

    return widgets


# ─── Callback helpers (pure state mutations) ────────────────────────────────────


def _set_and_recalc(
    state: AppState, attr: str, value: float, recalc: Callable[[], None]
) -> None:
    setattr(state, attr, value)
    recalc()


def _set_pressure(
    state: AppState, atm_value: float, recalc: Callable[[], None]
) -> None:
    state.pressure = atm_value * 101325.0
    recalc()


def _set_oxidant(state: AppState, o2_moles: float, recalc: Callable[[], None]) -> None:
    state.process.oxidant.o2_flow = o2_moles
    recalc()


def _set_air_mode(state: AppState, label: str, recalc: Callable[[], None]) -> None:
    state.process.oxidant.use_air = "Air" in label
    recalc()


def _set_injection(
    state: AppState, stream_name: str, flow: float, recalc: Callable[[], None]
) -> None:
    getattr(state.process, stream_name).flow = flow
    recalc()


def _set_feed_element(state: AppState, elem: str, value: float) -> None:
    setattr(state.base_feed, elem, value)


# ─── Main application ──────────────────────────────────────────────────────────


class GasificationApp:
    """Thin application shell: tab management and callback routing.

    SRP: Layout + wiring only. Plotting delegated to plots module.
    DIP: Depends on engine interface, not implementation.
    """

    def __init__(self, engine: GasificationEngine | None = None) -> None:
        apply_theme()
        self.engine = engine or GasificationEngine()
        self.state = AppState()

        self.fig = plt.figure(figsize=(16, 9.5))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(
                "Gasification Equilibrium Calculator"
            )
        self.fig.patch.set_facecolor(COLORS["bg"])

        self._build_tab_bar()
        self.tabs = [
            build_single_point_tab(self.fig, self.state, self._run_single_point),
            build_sweep_tab(self.fig, self.state, self._run_sweep),
            build_surface_tab(self.fig, self._run_surface),
            build_feed_tab(self.fig, self.state, self._apply_feed),
        ]
        self._setup_sweep_callbacks()

        self._show_tab(0)
        self._run_single_point()

    def _build_tab_bar(self) -> None:
        """Create 4 tab navigation buttons."""
        names = [
            "  Single Point  ",
            " Temp Sweep ",
            " Surface Plots ",
            "  Feed Editor  ",
        ]
        self.tab_buttons: list[Button] = []
        self.tab_axes: list[Any] = []
        for i, name in enumerate(names):
            ax = self.fig.add_axes((0.01 + i * 0.245, 0.955, 0.24, 0.04))
            ax.set_facecolor(COLORS["panel"])
            btn = Button(ax, name, color=COLORS["panel"], hovercolor=COLORS["grid"])
            btn.label.set_color(COLORS["text"])
            btn.label.set_fontsize(10)
            btn.label.set_fontweight("bold")
            btn.on_clicked(lambda _, idx=i: self._show_tab(idx))  # type: ignore[misc]
            self.tab_buttons.append(btn)
            self.tab_axes.append(ax)

    def _show_tab(self, idx: int) -> None:
        """Switch visible tab."""
        for i, ax in enumerate(self.tab_axes):
            active = i == idx
            ax.set_facecolor(COLORS["accent"] if active else COLORS["panel"])
            self.tab_buttons[i].label.set_color(
                COLORS["bg"] if active else COLORS["text"]
            )

        for ti, tab in enumerate(self.tabs):
            vis = ti == idx
            for widget in tab.values():
                if hasattr(widget, "set_visible"):
                    widget.set_visible(vis)
                elif hasattr(widget, "ax") and hasattr(widget.ax, "set_visible"):
                    widget.ax.set_visible(vis)
        self.fig.canvas.draw_idle()

    def _setup_sweep_callbacks(self) -> None:
        """Wire species checkbuttons in sweep tab."""
        sw = self.tabs[1]
        sw["check"].on_clicked(lambda lbl: self._toggle_species(lbl, sw))

    def _toggle_species(self, label: str, sw: dict[str, Any]) -> None:
        sp_key = sw["species_map"].get(label, label)
        if sp_key in self.state.selected_species:
            self.state.selected_species.remove(sp_key)
        else:
            self.state.selected_species.append(sp_key)
        if self.state.last_sweep:
            self._plot_sweep_results()

    # ─── Calculation callbacks ──────────────────────────────

    def _run_single_point(self) -> None:
        """Solve single-point equilibrium and update plots."""
        from .feed import build_total_feed

        feed_elements = build_total_feed(self.state.base_feed, self.state.process)

        result = self.engine.solve(
            temperature=self.state.temperature,
            pressure=self.state.pressure,
            feed=feed_elements,
        )
        self.state.last_result = result

        t = self.tabs[0]
        plots.plot_composition_bars(
            t["ax_bar"], result.composition_dict(), result.species
        )
        plots.plot_pie_chart(t["ax_pie"], result.composition_dict())
        plots.plot_info_panel(t["ax_info"], result)
        self.fig.canvas.draw_idle()

    def _run_sweep(self) -> None:
        """Run temperature sweep and plot."""
        sw = self.tabs[1]
        t_s = sw["sl_Ts"].val
        t_e = max(sw["sl_Te"].val, t_s + 100)
        n_pts = int(sw["sl_np"].val)
        pressure = sw["sl_Pp"].val * 101325.0

        from .feed import build_total_feed

        feed_elements = build_total_feed(self.state.base_feed, self.state.process)

        results = self.engine.temperature_sweep(
            t_start=t_s,
            t_end=t_e,
            n_points=n_pts,
            pressure=pressure,
            feed=feed_elements,
        )
        self.state.last_sweep = results
        self._plot_sweep_results()

    def _plot_sweep_results(self) -> None:
        sw = self.tabs[1]
        results = self.state.last_sweep
        if results is None:
            return
        plots.plot_sweep_composition(
            sw["ax_comp"], results, self.state.selected_species
        )
        plots.plot_sweep_metrics(sw["ax_metrics"], results)
        self.fig.canvas.draw_idle()

    def _run_surface(self) -> None:
        """Run 2D parameter surface sweep."""
        sf = self.tabs[2]
        param_label = sf["radio_param"].value_selected
        param_name, param_range = SURFACE_PARAM_MAP.get(
            param_label, ("steam_carbon_ratio", (0, 3))
        )

        from .feed import build_total_feed

        feed_elements = build_total_feed(self.state.base_feed, self.state.process)

        data = self.engine.surface_sweep(
            t_range=(400, 1600),
            param_name=param_name,
            param_range=param_range,
            n_t=25,
            n_param=20,
            pressure=self.state.pressure,
            feed=feed_elements,
        )
        self.state.last_surface = data
        self._plot_surface_results()

    def _plot_surface_results(self) -> None:
        sf = self.tabs[2]
        data: dict[str, Any] = self.state.last_surface  # type: ignore[assignment]

        sp_map = {
            "H\u2082": "H2",
            "CO": "CO",
            "CO\u2082": "CO2",
            "CH\u2084": "CH4",
            "H\u2082/CO": "h2_co",
        }
        sp_key = sp_map.get(sf["radio_species"].value_selected, "H2")

        temps = data["temperatures"] - 273.15
        params = data["param_values"]
        if data["param_name"] == "pressure":
            params = params / 101325.0

        t_grid, p_grid = np.meshgrid(temps, params, indexing="ij")

        if sp_key == "h2_co":
            z_data = data["h2_co_ratio"]
            z_label = "H\u2082/CO Ratio"
            cmap = "plasma"
        else:
            sp_idx = data["species"].index(sp_key)
            z_data = data["compositions"][:, :, sp_idx] * 100
            z_label = f'{SPECIES_DB[sp_key]["formula"]} [mol%]'
            cmap = "viridis"

        y_label = SURFACE_PARAM_LABELS.get(data["param_name"], data["param_name"])

        plots.plot_surface_3d(
            sf["ax_3d"],
            t_grid,
            p_grid,
            z_data,
            "Temp [\u00b0C]",
            y_label,
            z_label,
            cmap,
        )
        sf["cbar"] = plots.plot_contour(
            sf["ax_contour"],
            self.fig,
            t_grid,
            p_grid,
            z_data,
            "Temp [\u00b0C]",
            y_label,
            z_label,
            cmap,
            sf["cbar"],
        )
        self.fig.canvas.draw_idle()

    def _apply_feed(self) -> None:
        """Apply feed editor changes and update preview."""
        fd = self.tabs[3]

        # Update feed from sliders
        for elem, sl in fd["elem_sliders"].items():
            setattr(self.state.base_feed, elem, sl.val)

        # Plot feed composition
        plots.plot_feed_bars(fd["ax_comp"], self.state.base_feed.as_dict())

        # Preview equilibrium
        try:
            from .feed import build_total_feed

            feed_el = build_total_feed(self.state.base_feed, self.state.process)
            result = self.engine.solve(temperature=1000, feed=feed_el)
            plots.plot_equilibrium_preview(fd["ax_eq"], result)
        except Exception:
            pass

        self._run_single_point()
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main() -> None:
    app = GasificationApp()
    app.show()


if __name__ == "__main__":
    main()
