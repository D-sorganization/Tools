/** Vite-only proxy configuration that keeps the ephemeral token server-side. */

const LOOPBACK_AUTHORITY = /^http:\/\/127\.0\.0\.1:([1-9]\d{0,4})$/;

export interface AuthorityProxyEnvironment {
  readonly ROC_AUTHORITY_URL?: string;
  readonly ROC_AUTHORITY_TOKEN?: string;
}

export interface AuthorityProxyConfig {
  readonly target: string;
  readonly changeOrigin: false;
  readonly headers: Readonly<{ Authorization: string }>;
}

/** Build a strict loopback proxy or disable it when no authority was launched. */
export const buildAuthorityProxyConfig = (
  environment: AuthorityProxyEnvironment,
): AuthorityProxyConfig | undefined => {
  const target = environment.ROC_AUTHORITY_URL;
  const token = environment.ROC_AUTHORITY_TOKEN;
  if (target === undefined && token === undefined) return undefined;
  if (target === undefined || token === undefined) {
    throw new Error("authority URL and token must be configured together");
  }
  const match = LOOPBACK_AUTHORITY.exec(target);
  const port = Number(match?.[1] ?? 0);
  if (match === null || port > 65_535) {
    throw new Error("authority URL must use an explicit IPv4 loopback port");
  }
  if (!token || token !== token.trim()) {
    throw new Error("authority token must be nonempty and trimmed");
  }
  return Object.freeze({
    target,
    changeOrigin: false,
    headers: Object.freeze({ Authorization: `Bearer ${token}` }),
  });
};
