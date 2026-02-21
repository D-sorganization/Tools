"""Stateless plot rendering functions.

SRP: Takes data + axes, renders plots. No state, no callbacks, no layout.
Each function is small (<25 lines) and focused on one visualization.
"""

from typing import Any

import numpy as np

from .theme import COLORS, SPECIES_COLORS
from .thermo_data import SPECIES_DB


def plot_composition_bars(
    ax: Any, comp_dict: dict[str, float], species_keys: list[str]
) -> None:
    """Horizontal bar chart of gas-phase mole fractions."""
    ax.clear()
    ax.set_facecolor(COLORS["panel"])

    gas_keys = [k for k in species_keys if SPECIES_DB.get(k, {}).get("phase") == "gas"]
    vals = [comp_dict.get(k, 0) * 100 for k in gas_keys]
    colors = [SPECIES_COLORS.get(k, "#888") for k in gas_keys]
    labels = [SPECIES_DB[k]["formula"] for k in gas_keys]

    bars = ax.barh(
        range(len(gas_keys)),
        vals,
        color=colors,
        height=0.6,
        edgecolor="none",
        alpha=0.9,
    )
    ax.set_yticks(range(len(gas_keys)))
    ax.set_yticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_xlabel("Mole Fraction [%]", fontsize=10)
    ax.set_title(
        "Equilibrium Gas Composition",
        fontsize=12,
        fontweight="bold",
        color=COLORS["accent"],
        pad=10,
    )
    ax.set_xlim(0, max(max(vals), 1) * 1.2 + 1)
    ax.invert_yaxis()

    for bar, val in zip(bars, vals, strict=True):
        if val > 0.5:
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%",
                va="center",
                ha="left",
                fontsize=8,
                color=COLORS["text"],
                fontweight="bold",
            )


def plot_pie_chart(
    ax: Any, comp_dict: dict[str, float], threshold: float = 0.005
) -> None:
    """Pie chart of significant gas species."""
    ax.clear()
    ax.set_facecolor(COLORS["bg"])

    sig = [
        (k, v)
        for k, v in comp_dict.items()
        if v > threshold and SPECIES_DB.get(k, {}).get("phase") == "gas"
    ]
    if not sig:
        return

    labels = [SPECIES_DB[k]["formula"] for k, _ in sig]
    vals = [v for _, v in sig]
    colors = [SPECIES_COLORS.get(k, "#888") for k, _ in sig]

    wedges, texts, autotexts = ax.pie(
        vals,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        pctdistance=0.75,
        wedgeprops={"linewidth": 1.5, "edgecolor": COLORS["bg"], "alpha": 0.9},
        textprops={"fontsize": 9, "color": COLORS["text"]},
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_fontweight("bold")
        at.set_color(COLORS["bg"])
    ax.set_title(
        "Major Species Distribution",
        fontsize=11,
        fontweight="bold",
        color=COLORS["accent2"],
        pad=5,
    )


def plot_info_panel(ax: Any, result: Any) -> None:
    """Text info panel showing solver status and metrics."""
    ax.clear()
    ax.set_facecolor(COLORS["panel"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    status = "CONVERGED" if result.converged else "NOT CONVERGED"
    sc = COLORS["success"] if result.converged else COLORS["error"]

    lines = [
        (f"Status: {status}", sc, 12),
        (
            f"T = {result.temperature:.0f} K  ({result.temperature - 273.15:.0f} \u00b0C)",
            COLORS["accent"],
            10,
        ),
        (f"P = {result.pressure / 101325:.2f} atm", COLORS["accent2"], 10),
        (f"H\u2082/CO = {result.h2_co_ratio:.3f}", COLORS["text"], 10),
        (
            f"Carbon Conv. = {result.carbon_conversion * 100:.1f}%",
            COLORS["warning"],
            10,
        ),
        (f"CGE (HHV) = {result.cold_gas_efficiency * 100:.1f}%", COLORS["success"], 10),
        (f"Balance Error = {result.element_balance_error:.2e}", COLORS["text_dim"], 9),
        (f"Iterations = {result.iterations}", COLORS["text_dim"], 9),
    ]
    for i, (text, color, size) in enumerate(lines):
        ax.text(
            0.05,
            0.92 - i * 0.12,
            text,
            fontsize=size,
            color=color,
            fontweight="bold",
            transform=ax.transAxes,
        )


def plot_sweep_composition(
    ax: Any, results: list[Any], selected_species: list[str]
) -> None:
    """Temperature sweep composition curves with glow effect."""
    ax.clear()
    ax.set_facecolor(COLORS["panel"])

    temps_c = np.array([r.temperature - 273.15 for r in results])

    for sp_key in selected_species:
        if sp_key not in results[0].species:
            continue
        vals = np.array([r.composition_dict().get(sp_key, 0) * 100 for r in results])
        color = SPECIES_COLORS.get(sp_key, "#888")
        label = SPECIES_DB.get(sp_key, {}).get("formula", sp_key)
        ax.plot(temps_c, vals, color=color, label=label, linewidth=2.5, alpha=0.9)
        ax.plot(temps_c, vals, color=color, linewidth=5, alpha=0.15)

    ax.set_xlabel("Temperature [\u00b0C]", fontsize=10)
    ax.set_ylabel("Mole Fraction [%]", fontsize=10)
    ax.set_title(
        "Equilibrium Composition vs Temperature",
        fontsize=12,
        fontweight="bold",
        color=COLORS["accent"],
        pad=8,
    )
    ax.legend(
        loc="best",
        framealpha=0.7,
        facecolor=COLORS["panel"],
        edgecolor=COLORS["grid"],
        fontsize=8,
        ncol=2,
    )
    ax.set_xlim(temps_c[0], temps_c[-1])
    ax.set_ylim(bottom=0)


def plot_sweep_metrics(ax: Any, results: list[Any]) -> Any:
    """Dual-axis metrics plot: H2/CO + carbon conversion + CGE."""
    ax.clear()
    ax.set_facecolor(COLORS["panel"])

    temps_c = np.array([r.temperature - 273.15 for r in results])
    h2co = np.array([r.h2_co_ratio for r in results])
    c_conv = np.array([r.carbon_conversion * 100 for r in results])
    cge = np.array([r.cold_gas_efficiency * 100 for r in results])

    ln1 = ax.plot(
        temps_c, h2co, color=COLORS["accent"], linewidth=2.5, label="H\u2082/CO Ratio"
    )
    ax.plot(temps_c, h2co, color=COLORS["accent"], linewidth=5, alpha=0.15)
    ax.set_xlabel("Temperature [\u00b0C]", fontsize=10)
    ax.set_ylabel("H\u2082/CO Ratio", color=COLORS["accent"], fontsize=10)
    ax.tick_params(axis="y", labelcolor=COLORS["accent"])

    ax2 = ax.twinx()
    ln2 = ax2.plot(
        temps_c,
        c_conv,
        color=COLORS["warning"],
        linewidth=2,
        label="Carbon Conv. [%]",
        linestyle="--",
    )
    ln3 = ax2.plot(
        temps_c,
        cge,
        color=COLORS["success"],
        linewidth=2,
        label="CGE [%]",
        linestyle="-.",
    )
    ax2.set_ylabel("Conversion / Efficiency [%]", color=COLORS["text_dim"], fontsize=10)
    ax2.tick_params(axis="y", labelcolor=COLORS["text_dim"])

    lns = ln1 + ln2 + ln3
    ax.legend(
        lns,
        [line.get_label() for line in lns],
        loc="best",
        framealpha=0.7,
        facecolor=COLORS["panel"],
        edgecolor=COLORS["grid"],
        fontsize=8,
    )
    ax.set_title(
        "Process Metrics",
        fontsize=12,
        fontweight="bold",
        color=COLORS["accent2"],
        pad=8,
    )
    ax.set_xlim(temps_c[0], temps_c[-1])
    return ax2


def plot_surface_3d(
    ax: Any,
    t_grid: np.ndarray,
    p_grid: np.ndarray,
    z_data: np.ndarray,
    x_label: str,
    y_label: str,
    z_label: str,
    cmap: str = "viridis",
) -> None:
    """3D surface plot with wireframe overlay."""
    ax.clear()
    ax.set_facecolor(COLORS["bg"])

    ax.plot_surface(
        t_grid,
        p_grid,
        z_data,
        cmap=cmap,
        alpha=0.85,
        edgecolor="none",
        rstride=1,
        cstride=1,
        antialiased=True,
    )
    ax.plot_wireframe(
        t_grid,
        p_grid,
        z_data,
        color=COLORS["text_dim"],
        alpha=0.1,
        linewidth=0.3,
        rstride=3,
        cstride=3,
    )

    ax.set_xlabel(f"\n{x_label}", fontsize=9, color=COLORS["accent"])
    ax.set_ylabel(f"\n{y_label}", fontsize=9, color=COLORS["accent2"])
    ax.set_zlabel(f"\n{z_label}", fontsize=9, color=COLORS["text"])
    ax.set_title(
        f"{z_label} Surface",
        fontsize=12,
        fontweight="bold",
        color=COLORS["accent"],
        pad=15,
    )

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(COLORS["panel"])
        pane.set_alpha(0.8)


def plot_contour(
    ax: Any,
    fig: Any,
    t_grid: np.ndarray,
    p_grid: np.ndarray,
    z_data: np.ndarray,
    x_label: str,
    y_label: str,
    z_label: str,
    cmap: str = "viridis",
    existing_cbar: Any = None,
) -> Any:
    """Filled contour plot with labels and colorbar."""
    ax.clear()
    ax.set_facecolor(COLORS["panel"])

    cf = ax.contourf(t_grid, p_grid, z_data, levels=20, cmap=cmap, alpha=0.9)
    cs = ax.contour(
        t_grid,
        p_grid,
        z_data,
        levels=10,
        colors=COLORS["text_dim"],
        linewidths=0.5,
        alpha=0.5,
    )
    ax.clabel(cs, fontsize=7, colors=COLORS["text"], fmt="%.1f")

    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(
        f"{z_label} Contour",
        fontsize=11,
        fontweight="bold",
        color=COLORS["accent2"],
        pad=8,
    )

    if existing_cbar is not None:
        existing_cbar.remove()
    cbar = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(colors=COLORS["text_dim"], labelsize=8)
    return cbar


def plot_feed_bars(
    ax: Any,
    feed_dict: dict[str, float],
    title: str = "Feed Composition",
    ylabel: str = "Molar Amount",
) -> None:
    """Vertical bar chart of feed elements."""
    ax.clear()
    ax.set_facecolor(COLORS["panel"])
    if not feed_dict:
        return

    from .theme import ELEMENT_COLORS

    labels = list(feed_dict.keys())
    values = list(feed_dict.values())
    colors = [ELEMENT_COLORS.get(k, "#888") for k in labels]

    bars = ax.bar(labels, values, color=colors, edgecolor="none", alpha=0.9, width=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold", color=COLORS["accent"], pad=10)
    ax.set_ylabel(ylabel, fontsize=10)

    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=8,
            color=COLORS["text"],
            fontweight="bold",
        )


def plot_equilibrium_preview(ax: Any, result: Any) -> None:
    """Horizontal bar preview of significant equilibrium species."""
    ax.clear()
    ax.set_facecolor(COLORS["panel"])

    comp = result.composition_dict()
    sig = {k: v for k, v in comp.items() if v > 0.005}
    if not sig:
        ax.text(
            0.5,
            0.5,
            "No significant species",
            ha="center",
            va="center",
            fontsize=10,
            color=COLORS["text_dim"],
            transform=ax.transAxes,
        )
        return

    names = [SPECIES_DB[k]["formula"] for k in sig]
    vals = [v * 100 for v in sig.values()]
    clrs = [SPECIES_COLORS.get(k, "#888") for k in sig]

    bars = ax.barh(names, vals, color=clrs, height=0.5, edgecolor="none", alpha=0.9)
    ax.set_xlabel("Mole %", fontsize=10)
    ax.set_title(
        "Equilibrium at 1000 K",
        fontsize=12,
        fontweight="bold",
        color=COLORS["accent2"],
        pad=10,
    )
    for bar, val in zip(bars, vals, strict=True):
        if val > 1:
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=8,
                color=COLORS["text"],
                fontweight="bold",
            )
