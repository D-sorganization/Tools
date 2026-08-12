import { expect, type Page, type Request } from "@playwright/test";

const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

export interface NetworkAudit {
  assertClean(): void;
}

interface AuditOptions { readonly forbidApi?: boolean }

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
  page.on("pageerror", () => runtimeErrors.push("page error"));
  page.on("console", (message) => {
    if (message.type() === "error") runtimeErrors.push("console error");
  });
  return {
    assertClean(): void {
      expect(violations, "browser network boundary violations").toEqual([]);
      expect(failures, "browser network transport failures").toEqual([]);
      expect(runtimeErrors, "browser runtime errors").toEqual([]);
    },
  };
}
