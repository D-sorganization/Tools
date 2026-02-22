# Completist Audit Report

**Date**: 2026-02-22
**Source**: `.jules/completist_data/`

## Overview
This report quantifies the "completeness" of the codebase by analyzing markers left by developers.

## Summary Stats
| Marker Type | Count | Interpretation |
| :--- | :--- | :--- |
| **TODO** | 26 | Feature requests or reminders. High count implies ongoing heavy development. |
| **FIXME** | 14 | Known defects. High count implies stability risk. |
| **NotImplemented** | 34 | Stubs. |
| **Abstract Methods** | 23 | Architectural structure. |

## Critical Gaps (Top 20 TODOs)
These represent the most urgent feature gaps or cleanup tasks.
1. `./scripts/analyze_completist_data.py:111:            return {"file": filepath, "line": lineno, "text": content, "type": "TODO"}`
2. `./scripts/analyze_completist_data.py:125:        if marker_item["type"] == "TODO":`
3. `./scripts/analyze_completist_data.py:201:        "TODO": 3,`
4. `./scripts/analyze_completist_data.py:274:    chart.append(f'    "Feature Requests (TODO)" : {len(todos)}')`
5. `./scripts/analyze_completist_data.py:320:        f"- **Feature Gaps (TODO)**: {len(todos)}",`
6. `./scripts/pragmatic_programmer_review.py:208:            if "TODO" in content:`
7. `./scripts/pragmatic_programmer_review.py:218:                "title": f"High TODO count ({len(todos)})",`
8. `./scripts/pragmatic_programmer_review.py:221:                "recommendation": "Review TODOs",`
9. `./scripts/generate_assessments.py:159:- **Markers**: 445 `TODO` and 140 `FIXME` markers indicate significant unfinished work.`
10. `./scripts/generate_assessments.py:210:        "High Technical Debt (445 TODOs)",`
11. `./scripts/generate_assessments.py:213:-   445 `TODO` markers.`
12. `./scripts/generate_assessments.py:218:-   Convert valid `TODO` items into GitHub Issues.`
13. `./scripts/generate_assessments.py:303:    f.write("    - **Issue**: 445 `TODO` markers.\n")`
14. `./scripts/tools/code_quality_check.py:34:    (re.compile(r"\bTODO\b"), "TODO placeholder found"),`
15. `./scripts/setup_hooks.py:106:  - quality-check (no TODOs/FIXMEs)`
16. `./scripts/generate_fresh_assessments.py:120:                stats["todos"] += content.count("TODO")`
17. `./scripts/generate_fresh_assessments.py:191:- TODOs: {stats["todos"]}`
18. `./src/tools/quality_utils.py:34:    (re.compile(r"\bTODO\b"), "TODO placeholder found"),`
19. `./src/tools/quality_utils.py:44:        re.compile(r"<[^<>]*TODO[^<>]*>", re.IGNORECASE),`
20. `./src/tools/quality_utils.py:45:        "Angle bracket TODO placeholder",`

## Technical Debt (Top 20 FIXMEs)
These should be prioritized for immediate remediation.
1. `./scripts/analyze_completist_data.py:200:        "FIXME": 2,`
2. `./scripts/analyze_completist_data.py:275:    chart.append(f'    "Technical Debt (FIXME)" : {len(fixmes)}')`
3. `./scripts/generate_assessments.py:159:- **Markers**: 445 `TODO` and 140 `FIXME` markers indicate significant unfinished work.`
4. `./scripts/generate_assessments.py:214:-   140 `FIXME` markers.`
5. `./scripts/generate_assessments.py:217:-   Audit all `FIXME` items and resolve high-priority ones.`
6. `./scripts/tools/code_quality_check.py:35:    (re.compile(r"\bFIXME\b"), "FIXME placeholder found"),`
7. `./scripts/setup_hooks.py:106:  - quality-check (no TODOs/FIXMEs)`
8. `./scripts/generate_fresh_assessments.py:121:                stats["fixmes"] += content.count("FIXME")`
9. `./scripts/generate_fresh_assessments.py:192:- FIXMEs: {stats["fixmes"]}`
10. `./src/tools/quality_utils.py:35:    (re.compile(r"\bFIXME\b"), "FIXME placeholder found"),`
11. `./src/tools/quality_utils.py:48:        re.compile(r"<[^<>]*FIXME[^<>]*>", re.IGNORECASE),`
12. `./src/tools/quality_utils.py:49:        "Angle bracket FIXME placeholder",`
13. `./src/tools/matlab_quality_utils.py:300:        """Check for TODO, FIXME, HACK, XXX, and placeholders."""`
14. `./src/tools/matlab_quality_utils.py:303:            (r"\bFIXME\b", "FIXME placeholder found"),`
