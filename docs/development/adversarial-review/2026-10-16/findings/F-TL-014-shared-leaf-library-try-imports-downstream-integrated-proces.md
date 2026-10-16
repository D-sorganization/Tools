# F-TL-014

**Shared leaf library try-imports downstream integrated_process_simulator with unit-sensitive fallback**

| Severity  | Category     | Confidence | Effort | Status   |
| --------- | ------------ | ---------- | ------ | -------- |
| 🟨 Medium | architecture | high       | S      | proposed |

**Locations:** `src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:69`, `src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:174`, `src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:211`

## Finding

- Severity: Medium - Category: architecture
- Confidence: high
- Location: src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:68-71, :174-176; src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py:211; src/shared/python/chat/**init**.py
- Evidence:

```python
try:
    from integrated_process_simulator.calculators.thermodynamic_properties.species_database import get_species_database
except ImportError:
    @dataclass
    class _SpeciesData:
        molecular_weight: float  # kg/mol
...
mix_mw += frac * species.molecular_weight * 1000.0
```

- Failure scenario: CLAUDE.md:148 says "Tools is a leaf dependency". When Gasification_Model is importable (downstream it is), the real species DB is used; if its `molecular_weight` is g/mol, every mixture MW is 1000× too large, and Tools' CI can never see it. Behaviour differs by import environment.
- Why a newer engineer misses this: "try the real thing, fall back to a stub" looks like graceful degradation; it is an undeclared, untested dependency inversion.
- Proposed change: delete the try-import; make the species table a Tools-owned module with explicit units; let Gasification_Model inject a provider via a Protocol.
- Verification: `grep -r integrated_process_simulator src/shared` returns nothing; extend tests/architecture/test_layer_boundaries.py to forbid downstream package names.
- Effort: S

## Draft Pull Request

**Title:** `refactor(sidekick): remove upward import of integrated_process_simulator`

### Why This Matters (The Pitch)

The shared library that is meant to be a leaf dependency imports from the downstream simulator it feeds, and falls back to a private stub when that import fails. So behaviour depends on which environment you run in, and Tools' own CI can never see the downstream branch. The code multiplies species molecular weight by 1000, which is right only if the downstream database stores kg/mol; if it stores g/mol, every mixture MW is 1000 times too large and nothing in this repo would notice. Deleting the try-import and giving Tools its own species table with explicit units removes an undeclared dependency inversion, and a Protocol lets Gasification_Model inject its own provider on purpose. Both Tools and Gasification_Model maintainers benefit from a boundary that is testable on its own. A layer-boundary test forbidding downstream package names keeps it fixed.

### Proposed PR Body

## Problem

`src/shared/python/sidekick/process_calculators/syngas_compression_engine.py:68-71, :174-176`, `wgs_reactor_calculator.py:211` and `src/shared/python/chat/__init__.py` try to import `integrated_process_simulator...get_species_database`, else define a stub `_SpeciesData` with molecular weight in kg/mol, then scale by `1000.0`. CLAUDE.md:148 says Tools is a leaf dependency.

## Change

- [ ] Delete the try-import of `integrated_process_simulator`
- [ ] Create a Tools-owned species table module with explicit units
- [ ] Define a `Protocol` so Gasification_Model can inject a provider
- [ ] Extend `tests/architecture/test_layer_boundaries.py` to forbid downstream package names

## Verification

`grep -r integrated_process_simulator src/shared` returns nothing; the extended layer-boundary test passes.

## Risk and Rollback

Gasification_Model must adopt the injected provider if it relied on the implicit import; coordinate before merge. Rollback is a revert.

## Handoff

Follow-ups: Gasification_Model change to inject its species provider.

Refs: adversarial review 2026-10-16, F-TL-014

---

_Generated 2026-10-16 from an adversarial review of `Tools` @ `b630cd3`. This is a proposal, not a verdict — verify the cited lines before acting._
