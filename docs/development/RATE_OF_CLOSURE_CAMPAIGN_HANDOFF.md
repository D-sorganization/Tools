# Rate of Closure Ball-Flight Campaign Handoff

Status verified 2026-08-06. This is an isolated, local integration; it has not
been pushed and no source PR branch was rewritten.

## Integration checkout

- Worktree: `C:\Users\diete\Repositories\Tools-worktrees\ballflight-campaign-integration`
- Branch: `codex/ballflight-campaign-integration`
- Integration base: `626cfb64b0eddaa598a2a24dc2a050a420be25be`
- Implementation head before this handoff-only commit:
  `6aa5d5d4586118058c11b7f72461ed4f6ef63bea`

## Included PR stack

The source heads were merged in dependency order. A later source head includes
the earlier commits from that PR.

| PR | Capability | Exact included source head |
| --- | --- | --- |
| #4203 | Launch-monitor convention registry and fail-closed unknown signs | `3d899c8e95bc6808b07a1b230a21021d845c14ad` |
| #4209 | Launch Direction convention integration and visible unavailable Foresight option | `332fabdb41443119ae5f1f29ef63a8f9d7916144` |
| #4210 | Canonical flight-result metric catalog | `e6524dbb852e9356ae666dda5307cf0fd7e36960` |
| #4211 | Desired-flight inverse solver | `24d891cf78f5de125bb1fda602a7a9136b91f138` |
| #4215 | Impact solution families | `8e3af21672b105bcbc6f821644e013896d8293ba` |
| #4216 | Capability optimizer, including variability and downside/CVaR objectives | `4e11182d7d72abe66fd1066ca2086c2a87df5323` |
| #4207 | Paired wind physics and responsive locked-aspect canvases | `d668de1f1f808f7d5c8a4c5314a3ca940d71a4b9` |
| #4213 | Wind-estimate uncertainty analysis | `cb68d876591765428af3bf9ec17d9be27bf5c7df` |
| #4214 | Interactive 3D playback, correct Launch/Apex/Landing events, responsive canvas | `a7d337155cbd74c8198d9ef7f21add1b5d52b013` |
| #4208 | Versioned 3D spatial-target contract | `9aec34d89f91c08bf0882c556b66242d00cf3ba6` |
| #4212 | PyQt/React Launch Monitor Analytics and split statistics modules | `4b22e79cf829bac12217e60634ffbfbea5c40d6b` |

Integration-only reconciliation commits are
`16395378ec81c6b4c623804fc65ed886ea1bde7a` (formatting),
`107d8e43246d1ca545be1cb8980622f7a208a895` (Flight Explorer split),
`91a0bba09f5fba560744d9be840787dad500b2cf` (strict typing), and
`18fe8768fe27cc21d2d987a426e1a01fda3f5303` (spec reconciliation).

## Launch and registration

Run both commands from the worktree root in separate PowerShell terminals:

```powershell
python src/rate_of_closure/launch_pyqt6.py
cd src/rate_of_closure/web
npm run dev -- --host 127.0.0.1 --port 5270 --strictPort
```

The web app is then at `http://127.0.0.1:5270/`. Its authoritative Vite
package is `src/rate_of_closure/web`. The React navigation ID
`launch-monitor-analytics` is declared in
`web/src/model/viewPreferences.ts`, rendered by `web/src/App.tsx`, and backed
by `web/src/components/LaunchMonitorAnalyticsPanel.tsx`. The PyQt stable tab ID
`launch_monitor_analytics` is registered in `ui/pyqt6/main_window.py` and
backed by `ui/pyqt6/launch_monitor_analytics_tab.py`.

## Verification evidence

- Python campaign suite: `740 passed, 4 skipped, 15 warnings`.
- React/Vitest suite: `70` files and `439` tests passed.
- React production build: `tsc && vite build` passed (147 modules).
- React `type-check` and ESLint passed.
- Production Python mypy: no issues in 60 changed source files.
- Ruff and Black: 79 changed Python files passed.
- Module-size budget and `git diff --check` passed. The Flight Explorer and
  launch-monitor analytics production modules are each below 400 lines.
- The four skips are Rust parity cases because a compatible `tools_core` wheel
  is not installed. Other warnings are the existing Hypothesis pytest-plugin,
  Matplotlib legend, and Node local-storage-path warnings.
- A repository-root `npm run build` is not a valid campaign gate in this
  checkout: unrelated workspaces lack `turbo`, `next`, and other dependencies.
  The authoritative Rate of Closure package build above passes.

## Open release blockers

GitHub issue #4201 remains open. Its 2026-08-06 release checkpoint still
requires all of the following before any production-ready or merge claim:

- protected CI and required reviews for the combined stack;
- complete PyQt/React end-user workflows for spatial targets, desired-flight
  solving, solution families, capability profiles, and wind uncertainty;
- off-main-thread execution with progress and cancellation;
- complete save/load/export integration;
- Rust/WASM trajectory parity and installed-package/UpstreamDrift pin checks;
- scientific validation, convergence, performance, and benchmark evidence;
- browser resize, high-DPI, keyboard, accessibility, reduced-motion, and visual
  regression coverage.

The metric catalog, inverse solver, solution families, capability optimizer,
spatial target, and wind-uncertainty work must therefore be described as tested
contracts/cores unless and until their missing UI workflows are delivered.

## Next safe steps

1. Rebase or reconstruct the stack only after each parent PR is reviewed; do
   not retarget, force-push, admin-merge, or bypass protected checks.
2. Add the missing UI workflows against the canonical shared Python/TypeScript
   contracts, with one visible-control-to-state integration test per control.
3. Add cancellation/progress, persistence/export migrations, Rust/WASM golden
   parity, performance budgets, and Playwright visual/accessibility coverage.
4. Verify a clean installed package and the exact UpstreamDrift dependency pin.
5. Rerun every recorded gate, inspect protected GitHub checks/reviews, and keep
   #4201 open until every acceptance criterion has current evidence.
