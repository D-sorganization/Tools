import { useMemo, useState } from "react";

import type { LaunchMonitorRow } from "../model/launchMonitorAnalysisTypes";
import {
  PARAMETER_GROUPS,
  parameterGroupLabel,
  type ParameterGroup,
} from "../model/launchMonitorConventions";
import {
  buildComparisonRows,
  exportComparisonCsv,
  exportComparisonJson,
  filterComparisonRows,
} from "../model/launchMonitorComparisonWorkspace";

interface Props {
  rows?: LaunchMonitorRow[];
  sourceName: string;
}

const card = "rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 shadow-lg shadow-black/20";
const field = "rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-100 focus:border-sky-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500";
const button = "rounded border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500";

function downloadFile(name: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function LaunchMonitorComparisonWorkspace({ rows, sourceName }: Props) {
  const [selectedGroup, setSelectedGroup] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");

  const values = useMemo(() => {
    const tm: Record<string, number> = {};
    const fs: Record<string, number> = {};
    if (!rows || rows.length === 0) return { tm, fs };

    const tmRows = rows.filter((r) => String(r.monitor_vendor ?? "").toLowerCase().includes("trackman"));
    const fsRows = rows.filter((r) => String(r.monitor_vendor ?? "").toLowerCase().includes("foresight"));

    const keys = [...new Set(rows.flatMap(Object.keys))];
    for (const key of keys) {
      if (tmRows.length > 0) {
        const nums = tmRows.map((r) => Number(r[key])).filter((v) => Number.isFinite(v));
        if (nums.length > 0) tm[key] = nums.reduce((a, b) => a + b, 0) / nums.length;
      }
      if (fsRows.length > 0) {
        const nums = fsRows.map((r) => Number(r[key])).filter((v) => Number.isFinite(v));
        if (nums.length > 0) fs[key] = nums.reduce((a, b) => a + b, 0) / nums.length;
      }
      if (tmRows.length === 0 && fsRows.length === 0) {
        const nums = rows.map((r) => Number(r[key])).filter((v) => Number.isFinite(v));
        if (nums.length > 0) {
          const avg = nums.reduce((a, b) => a + b, 0) / nums.length;
          tm[key] = avg;
          fs[key] = avg;
        }
      }
    }
    return { tm, fs };
  }, [rows]);

  const allRows = useMemo(() => buildComparisonRows(values.tm, values.fs), [values]);
  const filteredRows = useMemo(
    () => filterComparisonRows(allRows, selectedGroup, searchQuery),
    [allRows, selectedGroup, searchQuery],
  );

  const handleExportJson = () => {
    const json = exportComparisonJson(filteredRows, sourceName);
    downloadFile("launch_monitor_comparison.json", json, "application/json");
  };

  const handleExportCsv = () => {
    const csv = exportComparisonCsv(filteredRows);
    downloadFile("launch_monitor_comparison.csv", csv, "text/csv;charset=utf-8;");
  };

  return (
    <section aria-label="Launch-Monitor Comparison Workspace" className={`${card} space-y-4`}>
      <div>
        <h3 className="text-base font-semibold text-slate-100">
          Launch-Monitor Convention Comparison (TrackMan vs Foresight)
        </h3>
        <p className="mt-1 text-xs text-slate-400">
          Direct subtraction is restricted to identical calculation contracts. Non-equivalent
          quantities report typed comparability reasons instead of fabricated differences.
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <span>Group:</span>
            <select
              value={selectedGroup}
              aria-label="Filter Parameter Group"
              title="Filter parameters by logical group"
              onChange={(e) => setSelectedGroup(e.target.value)}
              className={field}
            >
              <option value="all">All Groups</option>
              {PARAMETER_GROUPS.map((grp: ParameterGroup) => (
                <option key={grp} value={grp}>{parameterGroupLabel(grp)}</option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2 text-sm text-slate-300">
            <span>Search:</span>
            <input
              type="search"
              value={searchQuery}
              aria-label="Search Comparison Parameters"
              placeholder="Search parameters..."
              title="Search by parameter name, ID, or definition"
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`${field} w-48 sm:w-64`}
            />
          </label>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleExportJson}
            aria-label="Export Comparison as JSON"
            title="Download comparison records as JSON"
            className={button}
          >
            Export JSON
          </button>
          <button
            type="button"
            onClick={handleExportCsv}
            aria-label="Export Comparison as CSV"
            title="Download comparison records as CSV"
            className={button}
          >
            Export CSV
          </button>
        </div>
      </div>

      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table
          className="w-full text-left text-xs text-slate-200"
          aria-label="Launch Monitor Side-by-Side Comparison Table"
        >
          <thead className="border-b border-slate-800 bg-slate-950/80 text-[11px] uppercase tracking-wider text-slate-400">
            <tr>
              <th scope="col" className="p-2.5">Group</th>
              <th scope="col" className="p-2.5">Parameter</th>
              <th scope="col" className="p-2.5 text-right">TrackMan</th>
              <th scope="col" className="p-2.5 text-right">Foresight</th>
              <th scope="col" className="p-2.5 text-right">Signed Diff (TM - FS)</th>
              <th scope="col" className="p-2.5">Unit</th>
              <th scope="col" className="p-2.5">TM Ref</th>
              <th scope="col" className="p-2.5">FS Ref</th>
              <th scope="col" className="p-2.5">TM Event</th>
              <th scope="col" className="p-2.5">FS Event</th>
              <th scope="col" className="p-2.5">Comparability</th>
              <th scope="col" className="p-2.5">Definition</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 bg-slate-900/30">
            {filteredRows.map((row) => (
              <tr key={row.parameterId} className="hover:bg-slate-800/40">
                <td className="p-2.5 text-slate-400">{row.groupLabel}</td>
                <td className="p-2.5 font-medium text-slate-100">{row.label}</td>
                <td className="p-2.5 text-right font-mono">
                  {row.trackmanValue !== null ? row.trackmanValue.toFixed(2) : "—"}
                </td>
                <td className="p-2.5 text-right font-mono">
                  {row.foresightValue !== null ? row.foresightValue.toFixed(2) : "—"}
                </td>
                <td className={`p-2.5 text-right font-mono ${row.isComparable ? "text-emerald-400" : "italic text-amber-300"}`}>
                  {row.differenceText}
                </td>
                <td className="p-2.5 text-slate-400">{row.unit}</td>
                <td className="p-2.5 text-slate-400">{row.trackmanRef}</td>
                <td className="p-2.5 text-slate-400">{row.foresightRef}</td>
                <td className="p-2.5 text-slate-400">{row.trackmanTime}</td>
                <td className="p-2.5 text-slate-400">{row.foresightTime}</td>
                <td className="p-2.5">
                  {row.isComparable ? (
                    <span className="rounded bg-emerald-950/60 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-300 border border-emerald-800/50">
                      Comparable
                    </span>
                  ) : (
                    <span
                      title={`Incompatible reasons: ${row.reasonsText}`}
                      className="rounded bg-amber-950/60 px-1.5 py-0.5 text-[10px] font-semibold text-amber-300 border border-amber-800/50 cursor-help"
                    >
                      {row.reasonsText}
                    </span>
                  )}
                </td>
                <td className="max-w-xs truncate p-2.5 text-slate-400" title={row.definition}>
                  {row.definition}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
