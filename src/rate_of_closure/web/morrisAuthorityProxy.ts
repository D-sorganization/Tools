/** Server-only Vite proxy for the private loopback Morris authority. */

import type { ProxyOptions } from "vite";

export type AuthorityProxy = Record<string, string | undefined>;

const TARGET = "ROC_MORRIS_AUTHORITY_URL";
const TOKEN = "ROC_MORRIS_AUTHORITY_TOKEN";

const loopbackTarget = (value: string): string => {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new RangeError("Morris authority target must be a valid URL");
  }
  const exactOrigin = parsed.protocol === "http:"
    && parsed.hostname === "127.0.0.1"
    && parsed.port !== ""
    && Number.isInteger(Number(parsed.port))
    && Number(parsed.port) >= 1
    && Number(parsed.port) <= 65_535
    && parsed.username === ""
    && parsed.password === ""
    && parsed.pathname === "/"
    && parsed.search === ""
    && parsed.hash === "";
  if (!exactOrigin) throw new RangeError("Morris authority target must be an IPv4 loopback origin");
  return parsed.origin;
};

const privateToken = (value: string): string => {
  const visibleAscii = [...value].every((character) => {
    const code = character.charCodeAt(0);
    return code >= 33 && code <= 126;
  });
  if (value.length < 8 || value !== value.trim() || !visibleAscii) {
    throw new RangeError("Morris authority token is invalid");
  }
  return value;
};

export function morrisAuthorityProxy(environment: AuthorityProxy): Record<string, ProxyOptions> {
  const target = environment[TARGET];
  const token = environment[TOKEN];
  if (target === undefined && token === undefined) return {};
  if (target === undefined || token === undefined) {
    throw new RangeError("Morris authority proxy requires target and token");
  }
  return {
    "/api/rate-of-closure": {
      target: loopbackTarget(target),
      changeOrigin: false,
      headers: { Authorization: `Bearer ${privateToken(token)}` },
    },
  };
}
