/** Read a browser File with a FileReader fallback for compatible test shells. */
export function readBrowserFileText(file: File): Promise<string> {
  if (typeof file.text === "function") return file.text();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(String(reader.result ?? "")));
    reader.addEventListener("error", () =>
      reject(reader.error ?? new Error("Could not read file.")),
    );
    reader.readAsText(file, "utf-8");
  });
}
