import React from "react";

import {
  AGG_METHODS,
  FILTER_TYPES,
  type AggMethod,
  type FilterSpec,
  type FilterType,
} from "../../api/explorerSchemas";
import type { Pipeline } from "./explorerState";
import { Btn, Check, Field, NumInput, Row, Select, TextInput } from "./ui";

/**
 * Build the processing pipeline applied when (re)building a dataset: optional
 * resample, an ordered list of per-column filters/transforms, derived columns
 * from safe expressions, and an optional index trim. Controlled component.
 */

export interface PipelinePanelProps {
  columns: string[];
  pipeline: Pipeline;
  onChange: (p: Pipeline) => void;
}

/** Numeric parameters each filter type exposes, with sane defaults. */
const PARAM_SPECS: Record<
  FilterType,
  { key: string; label: string; def: number; step?: number }[]
> = {
  moving_average: [{ key: "window", label: "window", def: 5, step: 1 }],
  exponential: [{ key: "alpha", label: "alpha", def: 0.2, step: 0.05 }],
  median: [{ key: "window", label: "window", def: 5, step: 2 }],
  gaussian: [{ key: "sigma", label: "sigma", def: 2, step: 0.5 }],
  savgol: [
    { key: "window", label: "window", def: 7, step: 2 },
    { key: "polyorder", label: "polyorder", def: 2, step: 1 },
  ],
  hampel: [
    { key: "window", label: "window", def: 7, step: 2 },
    { key: "n_sigma", label: "n_sigma", def: 3, step: 0.5 },
  ],
  zscore: [{ key: "threshold", label: "threshold", def: 3, step: 0.5 }],
  fft_lowpass: [{ key: "high", label: "cutoff Hz", def: 1, step: 0.1 }],
  fft_highpass: [{ key: "low", label: "cutoff Hz", def: 0.1, step: 0.1 }],
  fft_bandpass: [
    { key: "low", label: "low Hz", def: 0.1, step: 0.1 },
    { key: "high", label: "high Hz", def: 1, step: 0.1 },
  ],
  integrate: [],
  differentiate: [],
};

function defaultParams(type: FilterType): Record<string, number> {
  const out: Record<string, number> = {};
  for (const p of PARAM_SPECS[type]) out[p.key] = p.def;
  return out;
}

export const PipelinePanel: React.FC<PipelinePanelProps> = ({
  columns,
  pipeline,
  onChange,
}) => {
  const firstCol = columns[0] ?? "";

  const setFilters = (filters: FilterSpec[]) =>
    onChange({ ...pipeline, filters });
  const updateFilter = (i: number, patch: Partial<FilterSpec>) =>
    setFilters(pipeline.filters.map((f, k) => (k === i ? { ...f, ...patch } : f)));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.9rem" }}>
      {/* Resample */}
      <div>
        <Check
          label="Resample onto a uniform grid"
          checked={pipeline.resample !== null}
          onChange={(on) =>
            onChange({
              ...pipeline,
              resample: on
                ? { interval_s: 1, agg: "mean", interpolate: false }
                : null,
            })
          }
        />
        {pipeline.resample && (
          <Row>
            <Field label="interval (s)">
              <NumInput
                min={0.001}
                step={0.5}
                value={pipeline.resample.interval_s}
                onChange={(e) =>
                  onChange({
                    ...pipeline,
                    resample: {
                      ...pipeline.resample!,
                      interval_s: Number(e.target.value) || 1,
                    },
                  })
                }
              />
            </Field>
            <Field label="aggregate">
              <Select
                value={pipeline.resample.agg}
                onChange={(e) =>
                  onChange({
                    ...pipeline,
                    resample: {
                      ...pipeline.resample!,
                      agg: e.target.value as AggMethod,
                    },
                  })
                }
              >
                {AGG_METHODS.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </Select>
            </Field>
            <Check
              label="interpolate gaps"
              checked={pipeline.resample.interpolate}
              onChange={(v) =>
                onChange({
                  ...pipeline,
                  resample: { ...pipeline.resample!, interpolate: v },
                })
              }
            />
          </Row>
        )}
      </div>

      {/* Filters */}
      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span style={{ fontSize: "0.74rem", color: "var(--text-secondary)" }}>
            Filters &amp; transforms ({pipeline.filters.length})
          </span>
          <Btn
            onClick={() =>
              setFilters([
                ...pipeline.filters,
                {
                  target: firstCol,
                  type: "moving_average",
                  params: defaultParams("moving_average"),
                  output: null,
                },
              ])
            }
            disabled={columns.length === 0}
          >
            + filter
          </Btn>
        </div>
        <div
          style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}
        >
          {pipeline.filters.map((f, i) => (
            <Row key={i} gap="0.45rem" wrap>
              <Select
                value={f.target}
                onChange={(e) => updateFilter(i, { target: e.target.value })}
                style={{ minWidth: "7rem" }}
              >
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </Select>
              <Select
                value={f.type}
                onChange={(e) => {
                  const type = e.target.value as FilterType;
                  updateFilter(i, { type, params: defaultParams(type) });
                }}
              >
                {FILTER_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </Select>
              {PARAM_SPECS[f.type].map((p) => (
                <NumInput
                  key={p.key}
                  title={p.label}
                  step={p.step ?? 1}
                  value={f.params[p.key] ?? p.def}
                  style={{ width: "5rem" }}
                  onChange={(e) =>
                    updateFilter(i, {
                      params: {
                        ...f.params,
                        [p.key]: Number(e.target.value),
                      },
                    })
                  }
                />
              ))}
              <TextInput
                placeholder="→ new col (optional)"
                value={f.output ?? ""}
                onChange={(e) =>
                  updateFilter(i, { output: e.target.value || null })
                }
                style={{ width: "9rem" }}
              />
              <Btn
                variant="danger"
                onClick={() =>
                  setFilters(pipeline.filters.filter((_, k) => k !== i))
                }
              >
                ✕
              </Btn>
            </Row>
          ))}
        </div>
      </div>

      {/* Derived columns */}
      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span style={{ fontSize: "0.74rem", color: "var(--text-secondary)" }}>
            Derived columns ({pipeline.derived.length})
          </span>
          <Btn
            onClick={() =>
              onChange({
                ...pipeline,
                derived: [
                  ...pipeline.derived,
                  { name: `derived_${pipeline.derived.length + 1}`, expression: "" },
                ],
              })
            }
          >
            + derived
          </Btn>
        </div>
        <div
          style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}
        >
          {pipeline.derived.map((d, i) => (
            <Row key={i} gap="0.45rem" wrap={false}>
              <TextInput
                value={d.name}
                placeholder="name"
                onChange={(e) =>
                  onChange({
                    ...pipeline,
                    derived: pipeline.derived.map((x, k) =>
                      k === i ? { ...x, name: e.target.value } : x,
                    ),
                  })
                }
                style={{ width: "9rem" }}
              />
              <TextInput
                value={d.expression}
                placeholder="e.g. (a + b) / 2"
                onChange={(e) =>
                  onChange({
                    ...pipeline,
                    derived: pipeline.derived.map((x, k) =>
                      k === i ? { ...x, expression: e.target.value } : x,
                    ),
                  })
                }
                style={{ flex: 1, minWidth: "10rem" }}
              />
              <Btn
                variant="danger"
                onClick={() =>
                  onChange({
                    ...pipeline,
                    derived: pipeline.derived.filter((_, k) => k !== i),
                  })
                }
              >
                ✕
              </Btn>
            </Row>
          ))}
          {pipeline.derived.length > 0 && (
            <span style={{ fontSize: "0.64rem", color: "var(--text-muted)" }}>
              Expressions may use column names and sin cos sqrt abs log log10 exp
              min max mean clip, plus pi and e.
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
