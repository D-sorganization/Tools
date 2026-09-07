# Changelog

## [1.16.2] - 2026-09-06 (patch bump)

### Changes

#### Features

- feat(wind): re-land wind strategy panel, worker, and responsive flight explorer integration (#4960) (#5004)
- feat(launcher): one tool registry, one launcher — generated tools.json/contract/README, tile launcher retired (#4916) (#4935)
- feat(spec): key SPEC.md change-log rows by PR instead of a serial version (#1520) (#4949)
- feat(swing_sim): re-land impact-interval club dynamics from PR #4133 (#4130) (#4945)
- feat(governance): divergence ledger + paired-PR gate; package-sharded module inventory (#4915, #4818) (#4934)

#### Fixes

- fix(tests, #4933): unquarantine god class guard and cross repo contracts (#5033)
- fix(tests, #4933): unquarantine bootstrap and calculators_expanded tests (#4933) (#5027)
- fix(tests, #4933): unquarantine shared package api stability and logging consistency tests (#5029)
- fix(tests, #4933): unquarantine dry compliance tests (#5031)
- fix(tests, #4933): unquarantine wave solver and math primitives bindings tests (#5032)
- fix(tests, #4933): unquarantine model generation api adapters, fix Flask endpoint collisions, and streamline rest_api shim (#4933) (#5026)
- fix(tests, #4933): unquarantine project packer, backup copy, and phase 1 quick wins tests (#5025)
- fix(tests, #4933): unquarantine vessel drafter contracts fallback and python version contract tests (#5023)
- fix(pressure-drop): rename laundered sibling privates public at home (#3991) (#5011)
- fix(tests): unquarantine pendulum and interaction tests with DbC and fixture fixes (#4933) (#5010)
- fix(p1am-firmware): remove blanket [0,100] tag clamp; enforce interlock limit domain at the boundary (#5003)
- fix(steam-engine-calculator): make SteamRequest mode-aware and expose engine selection (#5000)
- fix(c3d-viewer): show error state on reader failure, annotate demo fallback (#4999)
- fix(safety): withdraw uninstantiable neural PLC driver branch from PLCFactory (#4950) (#4990)
- fix(rate-web): preserve scroll position in visual capture for variation E2E (#4977) (#4987)
- fix(rate-of-closure): support explicit exemption marker in visual evidence gate (#4858) (#4984)
- fix(registry): report tools dropped for a missing launcher instead of hiding them (#4916) (#4981)
- fix(sidekick): replace token keyword in theme logger to prevent Semgrep credential disclosure false positive (#4978)
- fix(rate): raise pyqt resize budget and ensure visual state viewport visibility (#4968) (#4973)
- fix(ai): initialize message controller before loading session history (#4966) (#4970)
- fix(ci): allow ControlTower and Oglaptop font stack versions in Rate PyQt check (#4930) (#4965)
- fix(ai): honest chat placeholders and live Sidekick analytics registration (UD #9474) (#4959)
- fix(rate-web): close the second 6 px putting overflow at 390x844 — the green-import row (RM #1507) (#4936)
- fix(hooks): actually register the spec-rows merge driver (#1520) (#4956)
- fix(spec): register the spec-rows merge driver by a worktree-relative path (#1520) (#4953)
- fix(p1am): wire the PLC interlock reset path and stop backend defaults from tripping unmapped tags (#4911) (#4928)
- fix(ci): main-green — rate web narrow-viewport overflow, ci-standard concurrency (RM #1507) (#4927)

#### Other

- test(rate-of-closure, #5021): re-approve pyqt visual baselines from trusted push run 34045862045 (#5021) (#5022)
- ⚡ Bolt: Replace map and array spread in PuttingVisuals with single-pass loop (#5015)
- ⚡ Bolt: [performance improvement] Eliminate intermediate array allocations for chart bounds calculation (#5016)
- 🎨 Palette: Add focus rings to PlotsPanel buttons (#5017)
- test(rate-of-closure, #5018): re-approve pyqt visual baselines from trusted push run 34024927028 (#5018) (#5020)
- test(rate-of-closure, #5008): re-approve pyqt flight_explorer visual baseline (#5008) (#5013)
- refactor(folder-tools): delete vendored folder_tools leftover and fix dead tests (#5002)
- refactor(optimizer_gui): remove dead vendored GUI copy, keep canonical shim (#5001)
- docs(audit): reconcile golf app gap audit and rate of closure campaign ledger (#4921) (#4998)
- chore(deps): bump trimesh from 4.12.2 to 5.1.0 (#4996)
- chore(deps): bump ruff from 0.16.4 to 0.16.5 (#4995)
- ⚡ Bolt: Optimize swing scene bounds calculation (#4993)
- ⚡ Bolt: [performance improvement] Replace array spreads with single-pass validation bounds in golfLikeImpactIndex (#4991)
- ⚡ Bolt: [performance improvement] Replace spread operators with single-pass loops in launch monitor analysis (#4992)
- ci(tests): run the whole test tree as CI shards, one coverage floor, one mypy pin (#4913) (#4938)
- chore(inventory): make module-inventory index derivable from shards (#4957) (#4989)
- ⚡ Bolt: Optimize charting domain bounds calculation (#4969)
- ⚡ Bolt: Optimize launch monitor column inspection and grouping (#4971)
- test(ci): guard against required checks that can never report (#4983)
- ci: drop the remaining pull_request path filter (#4976)
- chore(security): ignore agent-local permission state (#4974)
- docs(scada): F-matrix as the tracker of record + three independent defect fixes (#4912) (#4947)
- chore(ci): retire 25 unowned Jules-* workflows, keep 3 (#1483) (#4948)
- 🎨 Palette: Add explicit label-input associations for accessibility (#4940)
- test(rate-of-closure, #4844): name every visual-drift offender; verify the system font stack (#4963)
- test(rate-of-closure, #4844): re-approve the nine drifted PyQt baselines under the recorded new font stack (#4964)
- test(rate-of-closure): re-approve the stale react/putting visual baseline (RM #1507 main-green) (#4958)
- ⚡ Bolt: [Performance] Replace Math.min/max spread with single-pass loop in VariationScatter (#4942)
- ⚡ Bolt: Optimize 3D projection bounds calculation (#4943)
- docs(spec): correct the merge-abort claim left in driver_command (#1520) (#4955)
- docs(spec-merge-driver): correct the merge-abort claim shipped in #4949/#4953 (#4954)
- refactor(rate-of-closure): launch-monitor tab consumes the canonical layer where twins permit (ADR-0046 Stage 2) (#4944)
- ci(security): re-enable CodeQL, same-repo guard on workflow_run jobs, drop dead pip-audit ignore, Jules inventory (#4923) (#4937)
- ci(contracts): fail without a downstream suite, API baselines for every vendored package, wheel + SBOM per release (#4920) (#4939)
- docs(release): closed-stack gap-audit decisions and campaign states (#4921 Phase 1) (#4932)
- docs(adr): fleet ADR home — mirror ADR-0016/0022/0031/0045-0048, add ADR-0049 + reference gate, fix ADR-007 duplicate (#4914) (#4931)

## [1.16.1] - 2026-09-06 (patch bump)

### Changes

#### Features

- feat(wind): re-land wind strategy panel, worker, and responsive flight explorer integration (#4960) (#5004)
- feat(launcher): one tool registry, one launcher — generated tools.json/contract/README, tile launcher retired (#4916) (#4935)
- feat(spec): key SPEC.md change-log rows by PR instead of a serial version (#1520) (#4949)
- feat(swing_sim): re-land impact-interval club dynamics from PR #4133 (#4130) (#4945)
- feat(governance): divergence ledger + paired-PR gate; package-sharded module inventory (#4915, #4818) (#4934)

#### Fixes

- fix(pressure-drop): rename laundered sibling privates public at home (#3991) (#5011)
- fix(tests): unquarantine pendulum and interaction tests with DbC and fixture fixes (#4933) (#5010)
- fix(p1am-firmware): remove blanket [0,100] tag clamp; enforce interlock limit domain at the boundary (#5003)
- fix(steam-engine-calculator): make SteamRequest mode-aware and expose engine selection (#5000)
- fix(c3d-viewer): show error state on reader failure, annotate demo fallback (#4999)
- fix(safety): withdraw uninstantiable neural PLC driver branch from PLCFactory (#4950) (#4990)
- fix(rate-web): preserve scroll position in visual capture for variation E2E (#4977) (#4987)
- fix(rate-of-closure): support explicit exemption marker in visual evidence gate (#4858) (#4984)
- fix(registry): report tools dropped for a missing launcher instead of hiding them (#4916) (#4981)
- fix(sidekick): replace token keyword in theme logger to prevent Semgrep credential disclosure false positive (#4978)
- fix(rate): raise pyqt resize budget and ensure visual state viewport visibility (#4968) (#4973)
- fix(ai): initialize message controller before loading session history (#4966) (#4970)
- fix(ci): allow ControlTower and Oglaptop font stack versions in Rate PyQt check (#4930) (#4965)
- fix(ai): honest chat placeholders and live Sidekick analytics registration (UD #9474) (#4959)
- fix(rate-web): close the second 6 px putting overflow at 390x844 — the green-import row (RM #1507) (#4936)
- fix(hooks): actually register the spec-rows merge driver (#1520) (#4956)
- fix(spec): register the spec-rows merge driver by a worktree-relative path (#1520) (#4953)
- fix(p1am): wire the PLC interlock reset path and stop backend defaults from tripping unmapped tags (#4911) (#4928)
- fix(ci): main-green — rate web narrow-viewport overflow, ci-standard concurrency (RM #1507) (#4927)

#### Other

- test(rate-of-closure, #5008): re-approve pyqt flight_explorer visual baseline (#5008) (#5013)
- refactor(folder-tools): delete vendored folder_tools leftover and fix dead tests (#5002)
- refactor(optimizer_gui): remove dead vendored GUI copy, keep canonical shim (#5001)
- docs(audit): reconcile golf app gap audit and rate of closure campaign ledger (#4921) (#4998)
- chore(deps): bump trimesh from 4.12.2 to 5.1.0 (#4996)
- chore(deps): bump ruff from 0.16.4 to 0.16.5 (#4995)
- ⚡ Bolt: Optimize swing scene bounds calculation (#4993)
- ⚡ Bolt: [performance improvement] Replace array spreads with single-pass validation bounds in golfLikeImpactIndex (#4991)
- ⚡ Bolt: [performance improvement] Replace spread operators with single-pass loops in launch monitor analysis (#4992)
- ci(tests): run the whole test tree as CI shards, one coverage floor, one mypy pin (#4913) (#4938)
- chore(inventory): make module-inventory index derivable from shards (#4957) (#4989)
- ⚡ Bolt: Optimize charting domain bounds calculation (#4969)
- ⚡ Bolt: Optimize launch monitor column inspection and grouping (#4971)
- test(ci): guard against required checks that can never report (#4983)
- ci: drop the remaining pull_request path filter (#4976)
- chore(security): ignore agent-local permission state (#4974)
- docs(scada): F-matrix as the tracker of record + three independent defect fixes (#4912) (#4947)
- chore(ci): retire 25 unowned Jules-* workflows, keep 3 (#1483) (#4948)
- 🎨 Palette: Add explicit label-input associations for accessibility (#4940)
- test(rate-of-closure, #4844): name every visual-drift offender; verify the system font stack (#4963)
- test(rate-of-closure, #4844): re-approve the nine drifted PyQt baselines under the recorded new font stack (#4964)
- test(rate-of-closure): re-approve the stale react/putting visual baseline (RM #1507 main-green) (#4958)
- ⚡ Bolt: [Performance] Replace Math.min/max spread with single-pass loop in VariationScatter (#4942)
- ⚡ Bolt: Optimize 3D projection bounds calculation (#4943)
- docs(spec): correct the merge-abort claim left in driver_command (#1520) (#4955)
- docs(spec-merge-driver): correct the merge-abort claim shipped in #4949/#4953 (#4954)
- refactor(rate-of-closure): launch-monitor tab consumes the canonical layer where twins permit (ADR-0046 Stage 2) (#4944)
- ci(security): re-enable CodeQL, same-repo guard on workflow_run jobs, drop dead pip-audit ignore, Jules inventory (#4923) (#4937)
- ci(contracts): fail without a downstream suite, API baselines for every vendored package, wheel + SBOM per release (#4920) (#4939)
- docs(release): closed-stack gap-audit decisions and campaign states (#4921 Phase 1) (#4932)
- docs(adr): fleet ADR home — mirror ADR-0016/0022/0031/0045-0048, add ADR-0049 + reference gate, fix ADR-007 duplicate (#4914) (#4931)

## [1.16.0] - 2026-09-06 (minor bump)

### Changes

#### Features

- feat(wind): re-land wind strategy panel, worker, and responsive flight explorer integration (#4960) (#5004)
- feat(launcher): one tool registry, one launcher — generated tools.json/contract/README, tile launcher retired (#4916) (#4935)
- feat(spec): key SPEC.md change-log rows by PR instead of a serial version (#1520) (#4949)
- feat(swing_sim): re-land impact-interval club dynamics from PR #4133 (#4130) (#4945)
- feat(governance): divergence ledger + paired-PR gate; package-sharded module inventory (#4915, #4818) (#4934)

#### Fixes

- fix(p1am-firmware): remove blanket [0,100] tag clamp; enforce interlock limit domain at the boundary (#5003)
- fix(steam-engine-calculator): make SteamRequest mode-aware and expose engine selection (#5000)
- fix(c3d-viewer): show error state on reader failure, annotate demo fallback (#4999)
- fix(safety): withdraw uninstantiable neural PLC driver branch from PLCFactory (#4950) (#4990)
- fix(rate-web): preserve scroll position in visual capture for variation E2E (#4977) (#4987)
- fix(rate-of-closure): support explicit exemption marker in visual evidence gate (#4858) (#4984)
- fix(registry): report tools dropped for a missing launcher instead of hiding them (#4916) (#4981)
- fix(sidekick): replace token keyword in theme logger to prevent Semgrep credential disclosure false positive (#4978)
- fix(rate): raise pyqt resize budget and ensure visual state viewport visibility (#4968) (#4973)
- fix(ai): initialize message controller before loading session history (#4966) (#4970)
- fix(ci): allow ControlTower and Oglaptop font stack versions in Rate PyQt check (#4930) (#4965)
- fix(ai): honest chat placeholders and live Sidekick analytics registration (UD #9474) (#4959)
- fix(rate-web): close the second 6 px putting overflow at 390x844 — the green-import row (RM #1507) (#4936)
- fix(hooks): actually register the spec-rows merge driver (#1520) (#4956)
- fix(spec): register the spec-rows merge driver by a worktree-relative path (#1520) (#4953)
- fix(p1am): wire the PLC interlock reset path and stop backend defaults from tripping unmapped tags (#4911) (#4928)
- fix(ci): main-green — rate web narrow-viewport overflow, ci-standard concurrency (RM #1507) (#4927)

#### Other

- refactor(folder-tools): delete vendored folder_tools leftover and fix dead tests (#5002)
- refactor(optimizer_gui): remove dead vendored GUI copy, keep canonical shim (#5001)
- docs(audit): reconcile golf app gap audit and rate of closure campaign ledger (#4921) (#4998)
- chore(deps): bump trimesh from 4.12.2 to 5.1.0 (#4996)
- chore(deps): bump ruff from 0.16.4 to 0.16.5 (#4995)
- ⚡ Bolt: Optimize swing scene bounds calculation (#4993)
- ⚡ Bolt: [performance improvement] Replace array spreads with single-pass validation bounds in golfLikeImpactIndex (#4991)
- ⚡ Bolt: [performance improvement] Replace spread operators with single-pass loops in launch monitor analysis (#4992)
- ci(tests): run the whole test tree as CI shards, one coverage floor, one mypy pin (#4913) (#4938)
- chore(inventory): make module-inventory index derivable from shards (#4957) (#4989)
- ⚡ Bolt: Optimize charting domain bounds calculation (#4969)
- ⚡ Bolt: Optimize launch monitor column inspection and grouping (#4971)
- test(ci): guard against required checks that can never report (#4983)
- ci: drop the remaining pull_request path filter (#4976)
- chore(security): ignore agent-local permission state (#4974)
- docs(scada): F-matrix as the tracker of record + three independent defect fixes (#4912) (#4947)
- chore(ci): retire 25 unowned Jules-* workflows, keep 3 (#1483) (#4948)
- 🎨 Palette: Add explicit label-input associations for accessibility (#4940)
- test(rate-of-closure, #4844): name every visual-drift offender; verify the system font stack (#4963)
- test(rate-of-closure, #4844): re-approve the nine drifted PyQt baselines under the recorded new font stack (#4964)
- test(rate-of-closure): re-approve the stale react/putting visual baseline (RM #1507 main-green) (#4958)
- ⚡ Bolt: [Performance] Replace Math.min/max spread with single-pass loop in VariationScatter (#4942)
- ⚡ Bolt: Optimize 3D projection bounds calculation (#4943)
- docs(spec): correct the merge-abort claim left in driver_command (#1520) (#4955)
- docs(spec-merge-driver): correct the merge-abort claim shipped in #4949/#4953 (#4954)
- refactor(rate-of-closure): launch-monitor tab consumes the canonical layer where twins permit (ADR-0046 Stage 2) (#4944)
- ci(security): re-enable CodeQL, same-repo guard on workflow_run jobs, drop dead pip-audit ignore, Jules inventory (#4923) (#4937)
- ci(contracts): fail without a downstream suite, API baselines for every vendored package, wheel + SBOM per release (#4920) (#4939)
- docs(release): closed-stack gap-audit decisions and campaign states (#4921 Phase 1) (#4932)
- docs(adr): fleet ADR home — mirror ADR-0016/0022/0031/0045-0048, add ADR-0049 + reference gate, fix ADR-007 duplicate (#4914) (#4931)

## [1.15.4] - 2026-09-05 (patch bump)

### Changes

#### Features

- feat(launcher): one tool registry, one launcher — generated tools.json/contract/README, tile launcher retired (#4916) (#4935)
- feat(spec): key SPEC.md change-log rows by PR instead of a serial version (#1520) (#4949)
- feat(swing_sim): re-land impact-interval club dynamics from PR #4133 (#4130) (#4945)
- feat(governance): divergence ledger + paired-PR gate; package-sharded module inventory (#4915, #4818) (#4934)

#### Fixes

- fix(safety): withdraw uninstantiable neural PLC driver branch from PLCFactory (#4950) (#4990)
- fix(rate-web): preserve scroll position in visual capture for variation E2E (#4977) (#4987)
- fix(rate-of-closure): support explicit exemption marker in visual evidence gate (#4858) (#4984)
- fix(registry): report tools dropped for a missing launcher instead of hiding them (#4916) (#4981)
- fix(sidekick): replace token keyword in theme logger to prevent Semgrep credential disclosure false positive (#4978)
- fix(rate): raise pyqt resize budget and ensure visual state viewport visibility (#4968) (#4973)
- fix(ai): initialize message controller before loading session history (#4966) (#4970)
- fix(ci): allow ControlTower and Oglaptop font stack versions in Rate PyQt check (#4930) (#4965)
- fix(ai): honest chat placeholders and live Sidekick analytics registration (UD #9474) (#4959)
- fix(rate-web): close the second 6 px putting overflow at 390x844 — the green-import row (RM #1507) (#4936)
- fix(hooks): actually register the spec-rows merge driver (#1520) (#4956)
- fix(spec): register the spec-rows merge driver by a worktree-relative path (#1520) (#4953)
- fix(p1am): wire the PLC interlock reset path and stop backend defaults from tripping unmapped tags (#4911) (#4928)
- fix(ci): main-green — rate web narrow-viewport overflow, ci-standard concurrency (RM #1507) (#4927)

#### Other

- ⚡ Bolt: [performance improvement] Replace array spreads with single-pass validation bounds in golfLikeImpactIndex (#4991)
- ⚡ Bolt: [performance improvement] Replace spread operators with single-pass loops in launch monitor analysis (#4992)
- ci(tests): run the whole test tree as CI shards, one coverage floor, one mypy pin (#4913) (#4938)
- chore(inventory): make module-inventory index derivable from shards (#4957) (#4989)
- ⚡ Bolt: Optimize charting domain bounds calculation (#4969)
- ⚡ Bolt: Optimize launch monitor column inspection and grouping (#4971)
- test(ci): guard against required checks that can never report (#4983)
- ci: drop the remaining pull_request path filter (#4976)
- chore(security): ignore agent-local permission state (#4974)
- docs(scada): F-matrix as the tracker of record + three independent defect fixes (#4912) (#4947)
- chore(ci): retire 25 unowned Jules-* workflows, keep 3 (#1483) (#4948)
- 🎨 Palette: Add explicit label-input associations for accessibility (#4940)
- test(rate-of-closure, #4844): name every visual-drift offender; verify the system font stack (#4963)
- test(rate-of-closure, #4844): re-approve the nine drifted PyQt baselines under the recorded new font stack (#4964)
- test(rate-of-closure): re-approve the stale react/putting visual baseline (RM #1507 main-green) (#4958)
- ⚡ Bolt: [Performance] Replace Math.min/max spread with single-pass loop in VariationScatter (#4942)
- ⚡ Bolt: Optimize 3D projection bounds calculation (#4943)
- docs(spec): correct the merge-abort claim left in driver_command (#1520) (#4955)
- docs(spec-merge-driver): correct the merge-abort claim shipped in #4949/#4953 (#4954)
- refactor(rate-of-closure): launch-monitor tab consumes the canonical layer where twins permit (ADR-0046 Stage 2) (#4944)
- ci(security): re-enable CodeQL, same-repo guard on workflow_run jobs, drop dead pip-audit ignore, Jules inventory (#4923) (#4937)
- ci(contracts): fail without a downstream suite, API baselines for every vendored package, wheel + SBOM per release (#4920) (#4939)
- docs(release): closed-stack gap-audit decisions and campaign states (#4921 Phase 1) (#4932)
- docs(adr): fleet ADR home — mirror ADR-0016/0022/0031/0045-0048, add ADR-0049 + reference gate, fix ADR-007 duplicate (#4914) (#4931)

## [1.15.3] - 2026-09-04 (patch bump)

### Changes

#### Features

- feat(spec): key SPEC.md change-log rows by PR instead of a serial version (#1520) (#4949)
- feat(swing_sim): re-land impact-interval club dynamics from PR #4133 (#4130) (#4945)
- feat(governance): divergence ledger + paired-PR gate; package-sharded module inventory (#4915, #4818) (#4934)

#### Fixes

- fix(rate-web): preserve scroll position in visual capture for variation E2E (#4977) (#4987)
- fix(rate-of-closure): support explicit exemption marker in visual evidence gate (#4858) (#4984)
- fix(registry): report tools dropped for a missing launcher instead of hiding them (#4916) (#4981)
- fix(sidekick): replace token keyword in theme logger to prevent Semgrep credential disclosure false positive (#4978)
- fix(rate): raise pyqt resize budget and ensure visual state viewport visibility (#4968) (#4973)
- fix(ai): initialize message controller before loading session history (#4966) (#4970)
- fix(ci): allow ControlTower and Oglaptop font stack versions in Rate PyQt check (#4930) (#4965)
- fix(ai): honest chat placeholders and live Sidekick analytics registration (UD #9474) (#4959)
- fix(rate-web): close the second 6 px putting overflow at 390x844 — the green-import row (RM #1507) (#4936)
- fix(hooks): actually register the spec-rows merge driver (#1520) (#4956)
- fix(spec): register the spec-rows merge driver by a worktree-relative path (#1520) (#4953)
- fix(p1am): wire the PLC interlock reset path and stop backend defaults from tripping unmapped tags (#4911) (#4928)
- fix(ci): main-green — rate web narrow-viewport overflow, ci-standard concurrency (RM #1507) (#4927)

#### Other

- test(ci): guard against required checks that can never report (#4983)
- ci: drop the remaining pull_request path filter (#4976)
- chore(security): ignore agent-local permission state (#4974)
- docs(scada): F-matrix as the tracker of record + three independent defect fixes (#4912) (#4947)
- chore(ci): retire 25 unowned Jules-* workflows, keep 3 (#1483) (#4948)
- 🎨 Palette: Add explicit label-input associations for accessibility (#4940)
- test(rate-of-closure, #4844): name every visual-drift offender; verify the system font stack (#4963)
- test(rate-of-closure, #4844): re-approve the nine drifted PyQt baselines under the recorded new font stack (#4964)
- test(rate-of-closure): re-approve the stale react/putting visual baseline (RM #1507 main-green) (#4958)
- ⚡ Bolt: [Performance] Replace Math.min/max spread with single-pass loop in VariationScatter (#4942)
- ⚡ Bolt: Optimize 3D projection bounds calculation (#4943)
- docs(spec): correct the merge-abort claim left in driver_command (#1520) (#4955)
- docs(spec-merge-driver): correct the merge-abort claim shipped in #4949/#4953 (#4954)
- refactor(rate-of-closure): launch-monitor tab consumes the canonical layer where twins permit (ADR-0046 Stage 2) (#4944)
- ci(security): re-enable CodeQL, same-repo guard on workflow_run jobs, drop dead pip-audit ignore, Jules inventory (#4923) (#4937)
- ci(contracts): fail without a downstream suite, API baselines for every vendored package, wheel + SBOM per release (#4920) (#4939)
- docs(release): closed-stack gap-audit decisions and campaign states (#4921 Phase 1) (#4932)
- docs(adr): fleet ADR home — mirror ADR-0016/0022/0031/0045-0048, add ADR-0049 + reference gate, fix ADR-007 duplicate (#4914) (#4931)

## [1.15.2] - 2026-09-04 (patch bump)

### Changes

#### Features

- feat(spec): key SPEC.md change-log rows by PR instead of a serial version (#1520) (#4949)
- feat(swing_sim): re-land impact-interval club dynamics from PR #4133 (#4130) (#4945)
- feat(governance): divergence ledger + paired-PR gate; package-sharded module inventory (#4915, #4818) (#4934)

#### Fixes

- fix(registry): report tools dropped for a missing launcher instead of hiding them (#4916) (#4981)
- fix(sidekick): replace token keyword in theme logger to prevent Semgrep credential disclosure false positive (#4978)
- fix(rate): raise pyqt resize budget and ensure visual state viewport visibility (#4968) (#4973)
- fix(ai): initialize message controller before loading session history (#4966) (#4970)
- fix(ci): allow ControlTower and Oglaptop font stack versions in Rate PyQt check (#4930) (#4965)
- fix(ai): honest chat placeholders and live Sidekick analytics registration (UD #9474) (#4959)
- fix(rate-web): close the second 6 px putting overflow at 390x844 — the green-import row (RM #1507) (#4936)
- fix(hooks): actually register the spec-rows merge driver (#1520) (#4956)
- fix(spec): register the spec-rows merge driver by a worktree-relative path (#1520) (#4953)
- fix(p1am): wire the PLC interlock reset path and stop backend defaults from tripping unmapped tags (#4911) (#4928)
- fix(ci): main-green — rate web narrow-viewport overflow, ci-standard concurrency (RM #1507) (#4927)

#### Other

- ci: drop the remaining pull_request path filter (#4976)
- chore(security): ignore agent-local permission state (#4974)
- docs(scada): F-matrix as the tracker of record + three independent defect fixes (#4912) (#4947)
- chore(ci): retire 25 unowned Jules-\* workflows, keep 3 (#1483) (#4948)
- 🎨 Palette: Add explicit label-input associations for accessibility (#4940)
- test(rate-of-closure, #4844): name every visual-drift offender; verify the system font stack (#4963)
- test(rate-of-closure, #4844): re-approve the nine drifted PyQt baselines under the recorded new font stack (#4964)
- test(rate-of-closure): re-approve the stale react/putting visual baseline (RM #1507 main-green) (#4958)
- ⚡ Bolt: [Performance] Replace Math.min/max spread with single-pass loop in VariationScatter (#4942)
- ⚡ Bolt: Optimize 3D projection bounds calculation (#4943)
- docs(spec): correct the merge-abort claim left in driver_command (#1520) (#4955)
- docs(spec-merge-driver): correct the merge-abort claim shipped in #4949/#4953 (#4954)
- refactor(rate-of-closure): launch-monitor tab consumes the canonical layer where twins permit (ADR-0046 Stage 2) (#4944)
- ci(security): re-enable CodeQL, same-repo guard on workflow_run jobs, drop dead pip-audit ignore, Jules inventory (#4923) (#4937)
- ci(contracts): fail without a downstream suite, API baselines for every vendored package, wheel + SBOM per release (#4920) (#4939)
- docs(release): closed-stack gap-audit decisions and campaign states (#4921 Phase 1) (#4932)
- docs(adr): fleet ADR home — mirror ADR-0016/0022/0031/0045-0048, add ADR-0049 + reference gate, fix ADR-007 duplicate (#4914) (#4931)

## [1.15.1] - 2026-09-03 (patch bump)

### Changes

#### Features

- feat(spec): key SPEC.md change-log rows by PR instead of a serial version (#1520) (#4949)
- feat(swing_sim): re-land impact-interval club dynamics from PR #4133 (#4130) (#4945)
- feat(governance): divergence ledger + paired-PR gate; package-sharded module inventory (#4915, #4818) (#4934)

#### Fixes

- fix(rate-web): close the second 6 px putting overflow at 390x844 — the green-import row (RM #1507) (#4936)
- fix(hooks): actually register the spec-rows merge driver (#1520) (#4956)
- fix(spec): register the spec-rows merge driver by a worktree-relative path (#1520) (#4953)
- fix(p1am): wire the PLC interlock reset path and stop backend defaults from tripping unmapped tags (#4911) (#4928)
- fix(ci): main-green — rate web narrow-viewport overflow, ci-standard concurrency (RM #1507) (#4927)

#### Other

- ⚡ Bolt: [Performance] Replace Math.min/max spread with single-pass loop in VariationScatter (#4942)
- ⚡ Bolt: Optimize 3D projection bounds calculation (#4943)
- docs(spec): correct the merge-abort claim left in driver_command (#1520) (#4955)
- docs(spec-merge-driver): correct the merge-abort claim shipped in #4949/#4953 (#4954)
- refactor(rate-of-closure): launch-monitor tab consumes the canonical layer where twins permit (ADR-0046 Stage 2) (#4944)
- ci(security): re-enable CodeQL, same-repo guard on workflow_run jobs, drop dead pip-audit ignore, Jules inventory (#4923) (#4937)
- ci(contracts): fail without a downstream suite, API baselines for every vendored package, wheel + SBOM per release (#4920) (#4939)
- docs(release): closed-stack gap-audit decisions and campaign states (#4921 Phase 1) (#4932)
- docs(adr): fleet ADR home — mirror ADR-0016/0022/0031/0045-0048, add ADR-0049 + reference gate, fix ADR-007 duplicate (#4914) (#4931)

All notable changes to this repository are documented here, newest first.
Sections are written **per release**, as a delta since the previous release
marker (the newest `v*` tag), by `scripts/release_changelog.py`; the release
workflow inserts them below this heading. History before 1.15.0 -- including
the ten auto-bump blocks of 2026-06-20 .. 2026-08-29 that each re-dumped the
whole git log -- is preserved verbatim in
[`docs/changelog_archive/CHANGELOG_pre-1.15.0.md`](docs/changelog_archive/CHANGELOG_pre-1.15.0.md).

## [1.15.0] - 2026-09-02

First release cut against a real tag (`v1.15.0`, tagged after the merge of
Tools #4910). `VERSION`, `pyproject.toml` and the workspace `package.json` all
declare 1.15.0 (`scripts/check_version_consistency.py`). This section
summarises, by area, what shipped between the 1.5.0 auto-bump (2026-06-20)
and this milestone; the per-commit list is in the archive.

### Rate of Closure (golf impact / flight / ground)

- Re-landed the slices of the folded consolidation PR #4466 on `main`:
  application + web authority, club + plotting, the React `web/src/model`
  layers, ground playback, variation, `ui/pyqt6` (40 modules), the flight
  integrator, the capability-optimisation cluster, `web_companion` +
  `web_distribution`, and the regional-ground registry contract (#4103).
- Club builder / club tester (#4549, #4562, #4799): mesh inertia, shaft
  dynamics, OEM document, biomech interchange, fitting; heavy-hit hand/body
  coupling with MuJoCo/Drake/OpenSim/Pinocchio model import; leaned loft,
  loft-aware hosels and real blade silhouettes with profile-view acceptance
  gates.
- Putting (#4800 P1-P9, ADR-0045 F2): stroke + impact parameters, green
  surface with 2-D roll and capture physics, putter-head import (mesh MOI,
  PutterSpec v2), stroke interchange, `putting_result/2` with Monte-Carlo
  dispersion and putter fitting, Qt Putting tab, React parity, shared-transport
  putt playback, UpstreamDrift green-surface adapter.
- Playback (ADR-0047 H1/H4, #4800 P8): `ball_flight_trajectory/1` interchange
  record, imported-trajectory replay, flight-side 3-D shot playback on a shared
  timeline model for Qt and React.
- Variation / Morris: TypeScript execution-document parser and fixture
  contract (#4558), metric invariants and `normalized_step` validation (#4459,
  #4458, #4461), evidence binding for variation plans, stable-point trace
  resampling, durable streaming ensemble analysis, complete-trial retention.
- Visual evidence and trusted gates: isolated PyQt runtime and evidence
  capture, deterministic bundled fonts, exact variation baseline approval,
  trusted Playwright/PyQt gates (#4602, #4610), stable-paint requirement for
  React captures, self-contained `web/` for the public mirror channel, LF
  normalisation of the Playwright config (#4629).
- Performance: single-pass CSV exports and `distributionMatrixToCsv`;
  Launch Monitor CSV export.

### Launch monitor (ADR-0046 Stage 1 canonical layer)

- Ported the full P1-P20 ladder into `src/shared/python/launch_monitor/`:
  dispersion, multivariate, trends (P1-P3); comparison, schema, treatment,
  relationships, modelling, profiles + importer (P4-P9); flexible analysis and
  contract v2 (P10-P11); strokes gained, longitudinal, outcome proxy
  (P12-P16); player covariation union port (P18); conformance, corpus merge
  and dataset reference (P17, P19, P20).
- Applied owner rulings D15, D17, D22, D23 and G1-D3 (legacy source-backed SG
  excludes-and-audits instead of raising); shared golden fixtures,
  full-corpus gates and program-manifest reconciliation (#4605);
  `rate_of_closure` now consumes canonical analytics v2 and source-backed
  strokes gained.

### Swing simulation and shared physics

- `swing_sim`: flight slice with React parity, standalone ground
  skid/roll/bounce module, variation execution metadata; read-only provider
  metadata (#4698); deterministic cross-platform wind turbulence hashing
  (#4513).
- `golf_club`: shared assembly-physics contracts; `putter_head` split along
  the serde boundary; stroke-plane tangential-impulse sign fix (#4829).
- `tools-core` (Rust): `flight_ground` slice of #4466; stable-clippy chunk lint
  fix; PyO3 link-library validation in CI; the wheel is now cached in the
  required CI lane.
- Pendulum simulator: continuous torque optimisation, force-source objective
  lab and certified comparisons, proximal-distal Companion Guide (#4446),
  swing-objective comparison (#4766), actuation-limit study (#4775), corrected
  club preset (#4785).

### P1AM control system (PLC / SCADA)

- Firmware: comms watchdog and bumpless setpoints recovered off-main (#3999,
  #4002); raw 0-5 V signal diagnostics and 200 A / 300 V calibration
  defaults; the `p1am-firmware` workflow now compiles the sketch and runs the
  host unit tests.
- Backend: route-introspection hardening and dependency pins (#4476-#4478),
  PID-tag / tuning / websocket / Modbus guard hardening, Python 3.10 test
  session isolation, aria-pressed on custom tab toggles.
- (In flight at this milestone: the interlock reset path and non-tripping
  defaults, Tools #4911.)

### Sidekick, shared library and data processing

- Sidekick: hardened startup and canonical runtime; pandas query-formula
  validation (#4490); process-calculator correctness fixes (#3867, #3868,
  #3976, #3981); Phase S1/S2 unified `rate_of_closure` integration.
- Shared: canonical `ThemePalette` and re-exported tokens (#4686); typed
  `icon_utils` contracts (#4488); 18 pre-existing mypy errors cleared; BLE001
  blind excepts resolved or annotated (#4690); three fixes upstreamed from
  consumers' vendored copies; orphaned DCR / glossary wording synced (#4493);
  Kalman / state-space estimator hardening; `modern_robotics` O-safe
  contracts; safe-eval, plugin-manager, robotics and contracts P2 cleanups
  (#3745).
- Data processing: quarantined directory made verifiable; vectorised UQ
  jackknife, incremental RRT nearest-neighbour, per-op cancellation tokens;
  scripting sandbox timeout race closed; codemap trusted-git resolution and
  per-symbol hoisting (#4489, #4492).
- Mocap: authority and canonical session contracts; `PlotWidget` export
  metadata and module inventory (#4722, #4740).

### Build, CI and repository governance

- Two-tier Python floor enforced (root 3.11+, sub-packages 3.10+) with a
  version-contract test; `rotation_converter` included in package discovery.
- CI: dispatchers converged on `CI_RUNNER_MODE` with a hosted fallback;
  fail-fast on an unprovisioned Python (#4510); isolated benchmark
  environment; merge-hold and Jules-Cleaner fixes; detect-secrets unblocked
  with a meaningful staleness guard; three main-only test failures repaired;
  order- and load-dependent test failures fixed; codemap conftest no longer
  skips the session (#4497).
- Repository: LF normalisation via `.gitattributes` (#4479); SPEC version-row
  dedupe and header-freshness gate (#4827); module-inventory merge driver;
  in-process metric pipelines replace `shell=True`.
- Release engineering (this milestone, Tools #4910 / RM #1507): the release
  workflow bumps only on a `feat`/`fix`/`perf` merge or manual dispatch,
  never on `chore`/`docs`/`ci`/`test`/`refactor` or bot pushes; the changelog
  generator emits the delta since the previous tag and resolves empty
  squash-merge subjects to PR titles; `scripts/check_version_consistency.py`
  keeps `VERSION`, `pyproject.toml`, `package.json` (and any Helm chart) in
  step.

### Dependencies

- Dependabot updates across Python, npm and GitHub Actions (15 commits).
