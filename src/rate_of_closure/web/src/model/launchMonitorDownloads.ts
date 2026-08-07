export function downloadJson(name: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  downloadBlob(name, blob);
}

export function downloadCsv<T extends object>(name: string, rows: T[]) {
  const headers = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const quote = (value: unknown) => `"${String(value ?? "").replace(/"/g, '""')}"`;
  const csv = [headers.map(quote), ...rows.map((row) => headers.map(
    (key) => quote((row as Record<string, unknown>)[key]),
  ))]
    .map((line) => line.join(",")).join("\n");
  downloadBlob(name, new Blob([csv], { type: "text/csv;charset=utf-8" }));
}

export function downloadSvg(name: string, id: string) {
  const node = document.getElementById(id);
  if (!(node instanceof SVGElement)) return;
  const payload = new XMLSerializer().serializeToString(node);
  downloadBlob(name, new Blob([payload], { type: "image/svg+xml;charset=utf-8" }));
}

function downloadBlob(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

