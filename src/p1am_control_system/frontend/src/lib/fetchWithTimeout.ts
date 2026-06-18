/**
 * fetch() that always settles within `timeoutMs`.
 *
 * Control screens flip a `busy` flag around a request and clear it in `finally`.
 * A plain fetch() against a backend that accepts the socket but never responds
 * (e.g. mid-restart) hangs forever, so `busy` sticks true and every control
 * silently locks with no way to recover but a page reload. Aborting on a timeout
 * guarantees the promise rejects, the `finally` runs, and `busy` clears.
 */
export async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs = 8000,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs / 1000}s`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}
