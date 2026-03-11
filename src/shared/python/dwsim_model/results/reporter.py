"""
results/reporter.py
===================
Generates HTML (and optionally JSON / plain-text) reports from
FlowsheetResults and GasificationMetrics objects.

The HTML report is self-contained — all CSS and JavaScript (Chart.js via CDN)
is embedded or referenced, so you can open it without a web server.

Sections produced
-----------------
1. Header  — scenario name, date/time, model version, convergence status
2. KPI Summary  — traffic-light coloured cards for the headline metrics
3. Syngas Composition  — bar chart (Chart.js) + table
4. Stream Table  — all material streams: T, P, flow, key compositions
5. Energy Balance  — energy inputs, losses, and closure
6. Reaction Configuration  — summary from YAML configs
7. Warnings / Errors  — anything logged during the run
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def generate_html_report(
    results,  # FlowsheetResults  (from extractor.py)
    metrics,  # GasificationMetrics (from metrics.py)
    output_path: str | Path,
    scenario_name: str = "Baseline",
    model_version: str = "2.0",
    targets: Optional[dict] = None,
) -> Path:
    """
    Generate a self-contained HTML report and write it to *output_path*.

    Parameters
    ----------
    results:
        FlowsheetResults object from ResultsExtractor.extract().
    metrics:
        GasificationMetrics object from MetricsCalculator.calculate().
    output_path:
        Path for the output .html file.
    scenario_name:
        Label shown in the report header.
    model_version:
        Version string shown in the header.
    targets:
        Optional dict of KPI targets (from scenario YAML) for traffic-light
        colouring.  Keys match GasificationMetrics field names.

    Returns
    -------
    Path to the written HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = _build_html(
        results=results,
        metrics=metrics,
        scenario_name=scenario_name,
        model_version=model_version,
        targets=targets or {},
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report written to {output_path}")
    return output_path


def generate_json_report(
    results,
    metrics,
    output_path: str | Path,
    scenario_name: str = "Baseline",
) -> Path:
    """
    Write a machine-readable JSON report.

    Useful for programmatic consumption, parameter sweeps, and regression tests.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "scenario": scenario_name,
        "generated_at": datetime.now().isoformat(),
        "converged": results.converged,
        "metrics": metrics.to_dict(),
        "streams": {name: _stream_to_dict(s) for name, s in results.streams.items()},
        "energy_streams": results.energy_streams,
        "errors": results.errors,
    }

    output_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"JSON report written to {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _stream_to_dict(stream) -> dict:
    """Convert a StreamResult to a plain dict (handles missing attributes)."""
    d = {}
    for attr in (
        "temperature_C",
        "pressure_kPa",
        "mass_flow_kg_s",
        "specific_enthalpy_kJ_kg",
        "volumetric_flow_Nm3_h",
    ):
        d[attr] = getattr(stream, attr, None)
    d["mole_fractions"] = getattr(stream, "mole_fractions", {})
    return d


def _fmt(value, fmt=".3g", suffix="") -> str:
    """Format a numeric value, or return '—' for None/NaN."""
    if value is None:
        return "—"
    try:
        return f"{value:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _traffic_light_class(metric_key: str, value, targets: dict) -> str:
    """Return CSS class 'good', 'warn', or 'bad' based on target comparison."""
    if not targets or value is None:
        return ""
    target = targets.get(metric_key)
    if target is None:
        return ""

    # Metrics where higher is better
    higher_better = {
        "cold_gas_efficiency",
        "carbon_conversion_efficiency",
        "mass_balance_closure",
        "energy_balance_closure",
    }
    # Metrics where lower is better
    lower_better = {
        "tar_loading_mg_Nm3",
        "specific_energy_consumption_kWh_t",
    }

    if metric_key in higher_better:
        if value >= target * 0.98:
            return "good"
        elif value >= target * 0.90:
            return "warn"
        else:
            return "bad"
    elif metric_key in lower_better:
        if value <= target * 1.02:
            return "good"
        elif value <= target * 1.15:
            return "warn"
        else:
            return "bad"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_html(results, metrics, scenario_name, model_version, targets) -> str:
    """Assemble the complete HTML document string."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    converged_badge = (
        '<span class="badge-green">CONVERGED</span>'
        if results.converged
        else '<span class="badge-red">NOT CONVERGED</span>'
    )

    # ── Syngas composition data for Chart.js ────────────────────────────────
    # fix: was "Syngas_Out" - canonical stream name is "Final_Syngas" (gasification.py line 238)
    syngas_stream = results.streams.get("Final_Syngas") or results.streams.get(
        next(iter(results.streams), "")
    )
    chart_labels = []
    chart_data = []
    if syngas_stream and syngas_stream.mole_fractions:
        sorted_comps = sorted(syngas_stream.mole_fractions.items(), key=lambda x: -x[1])
        chart_labels = [c for c, _ in sorted_comps]
        chart_data = [round(v * 100, 2) for _, v in sorted_comps]

    chart_json_labels = json.dumps(chart_labels)
    chart_json_data = json.dumps(chart_data)

    # ── KPI Cards ───────────────────────────────────────────────────────────
    kpi_cards = _build_kpi_cards(metrics, targets)

    # ── Stream Table ────────────────────────────────────────────────────────
    stream_table = _build_stream_table(results)

    # ── Energy Balance Table ─────────────────────────────────────────────────
    energy_table = _build_energy_table(results)

    # ── Syngas Composition Table ─────────────────────────────────────────────
    syngas_table = _build_syngas_table(syngas_stream)

    # ── Warnings & Errors ───────────────────────────────────────────────────
    warnings_html = _build_warnings(results, metrics)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gasification Model Report — {scenario_name}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"
          crossorigin="anonymous"></script>
  <style>
    /* ── Reset & Base ── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #f0f2f5;
      color: #1a1a2e;
      line-height: 1.6;
    }}

    /* ── Layout ── */
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .section {{ background: #fff; border-radius: 12px; padding: 24px;
                margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
    h1 {{ font-size: 1.8rem; font-weight: 700; color: #16213e; }}
    h2 {{ font-size: 1.2rem; font-weight: 600; color: #16213e;
          margin-bottom: 16px; border-bottom: 2px solid #e2e8f0;
          padding-bottom: 8px; }}
    h3 {{ font-size: 1rem; font-weight: 600; color: #4a5568; margin-bottom: 8px; }}

    /* ── Header ── */
    .header {{
      background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
      color: #fff; border-radius: 12px; padding: 32px;
      margin-bottom: 24px; display: flex; justify-content: space-between;
      align-items: center; flex-wrap: wrap; gap: 16px;
    }}
    .header-meta {{ font-size: 0.85rem; opacity: 0.75; }}
    .badge-green {{
      background: #22c55e; color: #fff; border-radius: 6px;
      padding: 4px 12px; font-size: 0.8rem; font-weight: 700;
    }}
    .badge-red {{
      background: #ef4444; color: #fff; border-radius: 6px;
      padding: 4px 12px; font-size: 0.8rem; font-weight: 700;
    }}

    /* ── KPI Cards ── */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 16px;
    }}
    .kpi-card {{
      border: 1px solid #e2e8f0; border-radius: 10px;
      padding: 18px; text-align: center;
      transition: box-shadow 0.2s;
    }}
    .kpi-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,.12); }}
    .kpi-card .value {{
      font-size: 2rem; font-weight: 700; color: #2d3748;
    }}
    .kpi-card .label {{
      font-size: 0.8rem; color: #718096; margin-top: 4px;
    }}
    .kpi-card .target-label {{
      font-size: 0.75rem; color: #a0aec0; margin-top: 2px;
    }}
    .kpi-card.good  {{ border-left: 4px solid #22c55e; background: #f0fdf4; }}
    .kpi-card.warn  {{ border-left: 4px solid #f59e0b; background: #fffbeb; }}
    .kpi-card.bad   {{ border-left: 4px solid #ef4444; background: #fef2f2; }}

    /* ── Chart ── */
    .chart-container {{ position: relative; height: 300px; }}

    /* ── Tables ── */
    .table-scroll {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    th {{
      background: #edf2f7; color: #4a5568; font-weight: 600;
      padding: 10px 14px; text-align: left; white-space: nowrap;
    }}
    td {{ padding: 8px 14px; border-bottom: 1px solid #e2e8f0; }}
    tr:hover td {{ background: #f7fafc; }}
    .num {{ text-align: right; font-family: 'Courier New', monospace; }}

    /* ── Two-column layout for chart + syngas table ── */
    .two-col {{
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 24px; align-items: start;
    }}
    @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

    /* ── Warnings ── */
    .warning-list {{ list-style: none; }}
    .warning-list li {{
      background: #fffbeb; border-left: 4px solid #f59e0b;
      border-radius: 4px; padding: 10px 14px; margin-bottom: 8px;
      font-size: 0.875rem;
    }}
    .error-list li {{
      background: #fef2f2; border-left: 4px solid #ef4444;
    }}
    .no-issues {{ color: #22c55e; font-weight: 500; font-size: 0.9rem; }}

    /* ── Footer ── */
    .footer {{
      text-align: center; padding: 24px;
      color: #a0aec0; font-size: 0.8rem;
    }}
  </style>
</head>
<body>
<div class="container">

  <!-- ── Header ── -->
  <div class="header">
    <div>
      <h1>Gasification Model Report</h1>
      <div class="header-meta" style="margin-top:8px;">
        Scenario: <strong>{scenario_name}</strong> &nbsp;|&nbsp;
        Model v{model_version} &nbsp;|&nbsp;
        Generated: {timestamp}
      </div>
    </div>
    <div style="text-align:right;">
      {converged_badge}
    </div>
  </div>

  <!-- ── KPI Summary ── -->
  <div class="section">
    <h2>Key Performance Indicators</h2>
    <div class="kpi-grid">
      {kpi_cards}
    </div>
  </div>

  <!-- ── Syngas Composition ── -->
  <div class="section">
    <h2>Syngas Composition (Final_Syngas)</h2>
    <div class="two-col">
      <div class="chart-container">
        <canvas id="sysgasChart"></canvas>
      </div>
      <div class="table-scroll">
        {syngas_table}
      </div>
    </div>
  </div>

  <!-- ── Stream Summary ── -->
  <div class="section">
    <h2>Stream Summary</h2>
    <div class="table-scroll">
      {stream_table}
    </div>
  </div>

  <!-- ── Energy Balance ── -->
  <div class="section">
    <h2>Energy Streams</h2>
    <div class="table-scroll">
      {energy_table}
    </div>
  </div>

  <!-- ── Warnings ── -->
  <div class="section">
    <h2>Warnings &amp; Diagnostics</h2>
    {warnings_html}
  </div>

  <div class="footer">
    DWSIM Gasification Model v{model_version} &mdash;
    Report generated {timestamp} &mdash;
    This report is for engineering analysis only.
  </div>

</div>

<script>
(function() {{
  const ctx = document.getElementById('sysgasChart');
  if (!ctx) return;
  const labels = {chart_json_labels};
  const data   = {chart_json_data};

  // Colour palette: blue shades for main syngas species, greys for trace
  const palette = [
    '#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6',
    '#06b6d4','#84cc16','#f97316','#ec4899','#6b7280',
    '#94a3b8','#d1d5db',
  ];
  const colors = labels.map((_, i) => palette[i % palette.length]);

  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{
        label: 'Mole %',
        data: data,
        backgroundColor: colors,
        borderRadius: 4,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: ctx => ctx.parsed.y.toFixed(2) + ' mol%'
          }}
        }}
      }},
      scales: {{
        y: {{
          beginAtZero: true,
          title: {{ display: true, text: 'Mole %' }},
          ticks: {{ callback: v => v + '%' }}
        }},
        x: {{
          ticks: {{ maxRotation: 45 }}
        }}
      }}
    }}
  }});
}})();
</script>
</body>
</html>
"""


def _build_kpi_cards(metrics, targets: dict) -> str:
    """Build the HTML for KPI metric cards."""

    def card(key, value, label, fmt, suffix, target_key=None):
        css = _traffic_light_class(target_key or key, value, targets)
        formatted = _fmt(value, fmt, suffix)
        target_val = targets.get(target_key or key)
        target_html = (
            f'<div class="target-label">Target: {_fmt(target_val, fmt, suffix)}</div>'
            if target_val is not None
            else ""
        )
        return (
            f'<div class="kpi-card {css}">'
            f'<div class="value">{formatted}</div>'
            f'<div class="label">{label}</div>'
            f"{target_html}"
            f"</div>"
        )

    m = metrics
    cards = [
        card(
            "cold_gas_efficiency",
            getattr(m, "cold_gas_efficiency", None),
            "Cold Gas Efficiency",
            ".1%",
            "",
            "cold_gas_efficiency",
        ),
        card(
            "carbon_conversion_efficiency",
            getattr(m, "carbon_conversion_efficiency", None),
            "Carbon Conversion",
            ".1%",
            "",
            "carbon_conversion_efficiency",
        ),
        card(
            "h2_co_ratio",
            getattr(m, "h2_co_ratio", None),
            "H₂/CO Ratio",
            ".2f",
            "",
            "h2_co_ratio_target",
        ),
        card(
            "specific_energy_consumption_kWh_t",
            getattr(m, "specific_energy_consumption_kWh_t", None),
            "Specific Energy (kWh/t)",
            ".0f",
            " kWh/t",
            "specific_energy_consumption_kWh_t",
        ),
        card(
            "tar_loading_mg_Nm3",
            getattr(m, "tar_loading_mg_Nm3", None),
            "Tar Loading",
            ".1f",
            " mg/Nm³",
            "tar_loading_mg_Nm3_max",
        ),
        card(
            "mass_balance_closure",
            getattr(m, "mass_balance_closure", None),
            "Mass Balance Closure",
            ".3f",
            "",
            "mass_balance_closure",
        ),
        card(
            "energy_balance_closure",
            getattr(m, "energy_balance_closure", None),
            "Energy Balance Closure",
            ".3f",
            "",
            "energy_balance_closure",
        ),
        card(
            "syngas_lhv_mj_nm3",
            getattr(m, "syngas_lhv_mj_nm3", None),
            "Syngas LHV",
            ".2f",
            " MJ/Nm³",
            None,
        ),
    ]
    return "\n".join(cards)


def _build_stream_table(results) -> str:
    """Build the HTML stream summary table."""
    key_comps = [
        "Carbon monoxide",
        "Hydrogen",
        "Carbon dioxide",
        "Methane",
        "Water",
        "Nitrogen",
    ]

    header_comps = "".join(f"<th>{c[:4]}.</th>" for c in key_comps)
    header = (
        "<table><thead><tr>"
        "<th>Stream</th><th>T (°C)</th><th>P (kPa)</th>"
        "<th>Flow (kg/s)</th><th>Vol (Nm³/h)</th>"
        f"{header_comps}"
        "</tr></thead><tbody>"
    )

    rows = []
    for name, s in sorted(results.streams.items()):
        mf = getattr(s, "mole_fractions", {}) or {}
        comp_cells = "".join(
            f'<td class="num">{_fmt(mf.get(c), ".3f")}</td>' for c in key_comps
        )
        rows.append(
            f"<tr>"
            f"<td><strong>{name}</strong></td>"
            f'<td class="num">{_fmt(getattr(s, "temperature_C", None), ".1f")}</td>'
            f'<td class="num">{_fmt(getattr(s, "pressure_kPa", None), ".1f")}</td>'
            f'<td class="num">{_fmt(getattr(s, "mass_flow_kg_s", None), ".3f")}</td>'
            f'<td class="num">{_fmt(getattr(s, "volumetric_flow_Nm3_h", None), ".1f")}</td>'
            f"{comp_cells}"
            f"</tr>"
        )

    if not rows:
        rows = [
            "<tr><td colspan='10' style='text-align:center;color:#a0aec0;'>"
            "No stream data extracted from simulation.</td></tr>"
        ]

    return header + "\n".join(rows) + "</tbody></table>"


def _build_energy_table(results) -> str:
    """Build the HTML energy streams table."""
    header = (
        "<table><thead><tr>"
        "<th>Energy Stream</th><th class='num'>Power (kW)</th>"
        "<th>Description</th>"
        "</tr></thead><tbody>"
    )

    # Descriptions for known energy streams
    descriptions = {
        "E_Gasifier_HeatLoss": "Gasifier thermal losses",
        "E_PEM_AC_Power": "PEM plasma AC input",
        "E_PEM_DC_Power": "PEM plasma DC power",
        "E_PEM_HeatLoss": "PEM thermal losses",
        "E_TRC_HeatLoss": "TRC thermal losses",
        "E_Blower": "Gas cleanup blower",
        "E_Gasifier_PreHeat": "Gasifier pre-heat duty",
        "E_PEM_PreHeat": "PEM pre-heat duty",
        "E_TRC_PreHeat": "TRC pre-heat duty",
    }

    rows = []
    for name, value in sorted(results.energy_streams.items()):
        kw = value / 1000.0 if value is not None else None
        desc = descriptions.get(name, "—")
        color = ""
        if kw is not None:
            color = "color:#ef4444;" if kw < 0 else "color:#22c55e;"
        rows.append(
            f"<tr>"
            f"<td><strong>{name}</strong></td>"
            f'<td class="num" style="{color}">{_fmt(kw, ".1f")} kW</td>'
            f"<td>{desc}</td>"
            f"</tr>"
        )

    if not rows:
        rows = [
            "<tr><td colspan='3' style='text-align:center;color:#a0aec0;'>"
            "No energy stream data available.</td></tr>"
        ]

    # Total
    total_kw = sum(v / 1000.0 for v in results.energy_streams.values() if v is not None)
    rows.append(
        f"<tr style='font-weight:700;background:#edf2f7;'>"
        f"<td>TOTAL</td>"
        f'<td class="num">{_fmt(total_kw, ".1f")} kW</td>'
        f"<td></td></tr>"
    )

    return header + "\n".join(rows) + "</tbody></table>"


def _build_syngas_table(syngas_stream) -> str:
    """Build the HTML syngas composition table."""
    header = (
        "<table><thead><tr>"
        "<th>Component</th><th class='num'>Mole %</th>"
        "</tr></thead><tbody>"
    )

    if syngas_stream is None or not getattr(syngas_stream, "mole_fractions", {}):
        return (
            header + "<tr><td colspan='2' style='text-align:center;color:#a0aec0;'>"
            "No syngas stream data.</td></tr></tbody></table>"
        )

    rows = []
    mf = syngas_stream.mole_fractions
    for comp, val in sorted(mf.items(), key=lambda x: -x[1]):
        rows.append(
            f'<tr><td>{comp}</td><td class="num">{_fmt(val * 100, ".2f")}%</td></tr>'
        )

    return header + "\n".join(rows) + "</tbody></table>"


def _build_warnings(results, metrics) -> str:
    """Build the warnings / diagnostics section HTML."""
    errors = getattr(results, "errors", [])
    warnings = getattr(metrics, "warnings", [])

    if not errors and not warnings:
        return '<p class="no-issues">✓ No warnings or errors recorded.</p>'

    html_parts = []

    if errors:
        items = "\n".join(f"<li>{e}</li>" for e in errors)
        html_parts.append(
            f"<h3>Simulation Errors ({len(errors)})</h3>"
            f'<ul class="warning-list error-list">{items}</ul>'
        )

    if warnings:
        items = "\n".join(f"<li>{w}</li>" for w in warnings)
        html_parts.append(
            f"<h3>Diagnostic Warnings ({len(warnings)})</h3>"
            f'<ul class="warning-list">{items}</ul>'
        )

    return "\n".join(html_parts)
