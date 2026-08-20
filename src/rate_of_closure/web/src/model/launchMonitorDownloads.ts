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
  const lines = [headers, ...rows.map((row) => headers.map(
    (key) => (row as Record<string, unknown>)[key],
  ))];
  downloadBlob(name, new Blob([
    lines.map((line) => line.map(quote).join(",")).join("\n"),
  ], { type: "text/csv;charset=utf-8" }));
}

export function downloadSvg(name: string, id: string) {
  const node = document.getElementById(id);
  if (!(node instanceof SVGElement)) return;
  downloadBlob(name, new Blob([
    new XMLSerializer().serializeToString(node),
  ], { type: "image/svg+xml;charset=utf-8" }));
}
