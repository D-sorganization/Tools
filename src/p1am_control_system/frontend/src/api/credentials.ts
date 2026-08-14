/**
 * Per-browser credential storage for the P1AM HMI (issue #4007).
 *
 * The HMI shipped with **no credential handling at all**: `apiFetch` built its
 * headers from caller options only and the WebSocket was opened bare. So an
 * operator who set `P1AM_API_KEY` on the backend got a socket closed with 1008
 * and a 401/403 on every control action — which made `P1AM_DEV_NO_AUTH=1` the
 * only way to make the shipped product work, and that is exactly why the
 * production installer hardcoded it.
 *
 * The key is held in `localStorage`, so it is scoped to one browser profile on
 * one machine and never travels in a URL the server can log. Three ways in:
 *
 * 1. `deploy/launch-hmi.sh` opens the kiosk at `#apikey=...`, read from the
 *    root-owned `/etc/p1am/backend.env`. The fragment is stripped from the
 *    address bar immediately and is never sent to any server (fragments are
 *    client-side only), so the kiosk is provisioned with no operator action.
 * 2. {@link promptForApiKey} when the backend rejects us.
 * 3. {@link setApiKey} from anywhere that wants to build a settings UI.
 */

/** localStorage key holding the operator/admin credential. */
const STORAGE_KEY = "p1am.apiKey";

/** URL-fragment parameter the kiosk launcher uses to seed the key once. */
const BOOTSTRAP_FRAGMENT_KEY = "apikey";

/** Credential header the backend's `auth_config` reads. */
export const CREDENTIAL_HEADER = "X-API-Key";

/**
 * Marker header sent on every request. It is deliberately *custom*: a
 * CORS-simple request cannot set one, so its presence forces the browser into a
 * preflight, which is what stops a foreign page from issuing a bodyless control
 * POST (`RequestGuardMiddleware`, issue #4037).
 */
export const CSRF_HEADER = "X-Requested-With";
export const CSRF_HEADER_VALUE = "p1am-hmi";

/** Minimal slice of the `Storage` API this module needs. */
interface KeyStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/**
 * Session-lifetime fallback used when `localStorage` is unavailable or
 * non-conforming (private browsing, a locked-down kiosk profile, a test
 * environment whose global `localStorage` is a stub). The credential then
 * survives navigation within the page but not a reload — degraded, not broken.
 */
const memoryStore = new Map<string, string>();
const inMemory: KeyStore = {
  getItem: (key) => memoryStore.get(key) ?? null,
  setItem: (key, value) => void memoryStore.set(key, value),
  removeItem: (key) => void memoryStore.delete(key),
};

function storage(): KeyStore {
  try {
    if (typeof window === "undefined") return inMemory;
    const candidate = window.localStorage as unknown as KeyStore | undefined;
    if (
      candidate &&
      typeof candidate.getItem === "function" &&
      typeof candidate.setItem === "function" &&
      typeof candidate.removeItem === "function"
    ) {
      return candidate;
    }
  } catch {
    // Accessing localStorage throws outright in some privacy modes.
  }
  return inMemory;
}

/**
 * Consume a `#apikey=...` fragment, persist it, and scrub it from the URL.
 *
 * Runs at most once per page load. Stripping via `replaceState` keeps the key
 * out of the address bar, the back/forward history and any `Referer`.
 */
function consumeBootstrapFragment(): void {
  if (typeof window === "undefined" || !window.location.hash) return;
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const seeded = params.get(BOOTSTRAP_FRAGMENT_KEY);
  if (!seeded) return;

  setApiKey(seeded);
  params.delete(BOOTSTRAP_FRAGMENT_KEY);
  const rest = params.toString();
  window.history.replaceState(
    null,
    "",
    `${window.location.pathname}${window.location.search}${rest ? `#${rest}` : ""}`,
  );
}

let bootstrapped = false;

/**
 * The stored credential, or null when none has been provisioned.
 *
 * @returns the operator/admin API key for this browser profile.
 */
export function getApiKey(): string | null {
  if (!bootstrapped) {
    bootstrapped = true;
    consumeBootstrapFragment();
  }
  return storage().getItem(STORAGE_KEY);
}

/**
 * Persist a credential for this browser profile.
 *
 * @param key - the API key; blank input clears the stored value instead.
 */
export function setApiKey(key: string): void {
  const trimmed = key.trim();
  if (!trimmed) {
    clearApiKey();
    return;
  }
  storage().setItem(STORAGE_KEY, trimmed);
}

/** Forget the stored credential (e.g. after the backend rejects it). */
export function clearApiKey(): void {
  storage().removeItem(STORAGE_KEY);
}

/**
 * Headers every backend call carries: the credential when we have one, and the
 * preflight-forcing marker always.
 *
 * @returns a plain header map safe to spread into `fetch` options.
 */
export function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = { [CSRF_HEADER]: CSRF_HEADER_VALUE };
  const key = getApiKey();
  if (key) headers[CREDENTIAL_HEADER] = key;
  return headers;
}

/**
 * Ask the operator for the credential and store it.
 *
 * Deliberately a `window.prompt`: the Pi runs a single-purpose kiosk where this
 * is a once-per-browser bootstrap, not a login flow. Returns null when the
 * operator dismisses it or no prompt is available.
 */
export function promptForApiKey(): string | null {
  if (typeof window === "undefined" || typeof window.prompt !== "function") {
    return null;
  }
  const entered = window.prompt(
    "This control system requires an API key.\n\n" +
      "On the Pi, run:  sudo grep '^P1AM_API_KEY=' /etc/p1am/backend.env",
  );
  if (entered === null) return null;
  setApiKey(entered);
  return getApiKey();
}
