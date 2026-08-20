import { execFileSync } from "node:child_process";
import { rmSync } from "node:fs";
import path from "node:path";
import process from "node:process";

const COMMIT_PATTERN = /^[0-9a-f]{40}$/;

export default function exactReleaseBuild(): void {
  const revision = execFileSync("git", ["rev-parse", "HEAD"], {
    encoding: "utf8",
  }).trim();
  if (!COMMIT_PATTERN.test(revision)) {
    throw new Error("browser qualification requires one exact Git revision");
  }
  rmSync(path.resolve(import.meta.dirname, "../../../dist"), {
    force: true,
    recursive: true,
  });
  const npmEntrypoint = process.env.npm_execpath;
  if (npmEntrypoint === undefined || path.basename(npmEntrypoint) !== "npm-cli.js") {
    throw new Error("browser qualification must be launched through npm");
  }
  execFileSync(process.execPath, [npmEntrypoint, "run", "build:release"], {
    env: { ...process.env, ROC_RELEASE_REVISION: revision },
    stdio: "inherit",
  });
}
