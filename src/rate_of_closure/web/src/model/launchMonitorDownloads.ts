function downloadBlob(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function downloadJson(name: string, payload: unknown) {
  downloadBlob(name, new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  }));
}

export function downloadCsv<T extends object>(name: string, rows: T[]) {
  const headers = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const quote = (value: unknown) => `"${String(value ?? "").replace(/"/g, '""')}"`;

  // ⚡ Bolt Optimization: Replace chained array .map().join() with a single-pass loop
  // to eliminate intermediate array allocations and reduce GC pressure for large dataset exports.
  let csv = headers.map(quote).join(",");
  for (let i = 0; i < rows.length; i++) {
    csv += "\n";
    const row = rows[i] as Record<string, unknown>;
    for (let j = 0; j < headers.length; j++) {
      if (j > 0) csv += ",";
      csv += quote(row[headers[j]]);
    }
  }

  downloadBlob(name, new Blob([csv], { type: "text/csv;charset=utf-8" }));
}

export function downloadSvg(name: string, id: string) {
  const node = document.getElementById(id);
  if (!(node instanceof SVGElement)) return;
  downloadBlob(name, new Blob([
    new XMLSerializer().serializeToString(node),
  ], { type: "image/svg+xml;charset=utf-8" }));
}
