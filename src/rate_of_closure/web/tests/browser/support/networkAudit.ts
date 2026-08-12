import { expect, type Page, type Request } from "@playwright/test";

const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

export interface NetworkAudit {
  assertBoundaryClean(): void;
  assertClean(): void;
}

interface AuditOptions {
  readonly allowedRuntimeError?: RegExp;
  readonly forbidApi?: boolean;
  readonly maxRuntimeErrors?: number;
  readonly maxTransportFailures?: number;
}

export const summarizeRuntimeError = (kind: string, detail: string): string => {
  const redacted = detail
    .replace(/\b(?:https?|file|blob|data):[^\s]+/giu, "[url]")
    .replace(/\b[A-Za-z0-9_-]{24,}\b/gu, "[redacted]")
    .replace(/\s+/gu, " ")
    .trim();
  return `${kind}: ${redacted.slice(0, 160) || "no detail"}`;
};

const requestViolation = (request: Request, expectedOrigin: string): string | null => {
  const target = new URL(request.url());
  if (!LOCAL_HOSTS.has(target.hostname)) return "external-origin request";
  if (target.origin !== expectedOrigin) return "wrong-loopback-origin request";
  const headers = request.headers();
  if ("authorization" in headers || "cookie" in headers) return "browser credential request";
  return null;
};

/** Audit one page without retaining or printing request URLs or secret values. */
export function auditSameOriginNetwork(page: Page, expectedOrigin: string,
  options: AuditOptions = {}): NetworkAudit {
  const violations: string[] = [];
  const failures: string[] = [];
  const runtimeErrors: string[] = [];
  page.on("request", (request) => {
    const violation = requestViolation(request, expectedOrigin);
    if (violation !== null) violations.push(violation);
    else if (options.forbidApi === true && new URL(request.url()).pathname.startsWith("/api/")) {
      violations.push("static inspection authority request");
    }
  });
  page.on("requestfailed", () => failures.push("browser request failed"));
  page.on("pageerror", (error) => {
    runtimeErrors.push(summarizeRuntimeError("page error", error.message));
  });
  page.on("console", (message) => {
    if (message.type() === "error") {
      runtimeErrors.push(summarizeRuntimeError("console error", message.text()));
    }
  });
  return {
    assertBoundaryClean(): void {
      expect(violations, "browser network boundary violations").toEqual([]);
    },
    assertClean(): void {
      expect(violations, "browser network boundary violations").toEqual([]);
      expect(failures.length, "browser network transport failures")
        .toBeLessThanOrEqual(options.maxTransportFailures ?? 0);
      expect(runtimeErrors.length, "browser runtime errors")
        .toBeLessThanOrEqual(options.maxRuntimeErrors ?? 0);
      const allowedRuntimeError = options.allowedRuntimeError;
      if (allowedRuntimeError !== undefined) {
        runtimeErrors.forEach((error) => expect(error).toMatch(allowedRuntimeError));
      }
    },
  };
}
