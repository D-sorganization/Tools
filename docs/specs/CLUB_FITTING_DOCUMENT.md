# Club Fitting Document — OEM Interchange Reference

Wire: `golf_club.fitting_document/1` · Parser:
`shared/python/golf_club/fitting_document.py` · Epic: #4549 (C3, #4552)

One JSON document carries everything a fitting run needs to be reproducible:
the rigid assembly, the measured shaft profile, face geometry, the tip-mass
record the shaft-delivery model consumes, an optional integrity-pinned mesh
reference, and provenance. **Parsing fails closed**: unknown fields at any
level, a wrong `format`, malformed hashes/dates, or out-of-band values are
refused with a named reason — there is no silent field-dropping path.

All quantities are SI (meters, kilograms, seconds); angles are degrees where
the field name says `_deg`. JSON must be finite (`NaN`/`Infinity` refused).
Serialization from our side is deterministic: sorted keys, compact
separators — byte-identical for identical inputs, so documents can be
content-addressed.

## Top level

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `format` | string | yes | Exactly `"golf_club.fitting_document/1"` |
| `document_id` | string | yes | Non-empty identifier, OEM-chosen |
| `face` | object | yes | See **face** |
| `assembly` | object | yes | A complete `golf_club.assembly/1` sub-document |
| `shaft_profile` | object | yes | A complete `golf_club.shaft_profile/1` sub-document |
| `tip_mass` | object | yes | See **tip_mass** |
| `mesh_reference` | object | no | See **mesh_reference** |
| `provenance` | object | yes | See **provenance** |

## face

| Field | Type | Constraint |
| --- | --- | --- |
| `loft_deg` | number | `(0, 90)` |
| `lie_deg` | number | `(30, 90)` |
| `bulge_m` | number | `>= 0`; `0` = flat face (irons/wedges) |
| `roll_m` | number | `>= 0`; `0` = flat face |

## tip_mass

The head as the shaft tip sees it — the inputs of the shaft forward-dynamics
model (`quasi_static_centrifugal_alignment/1`).

| Field | Type | Constraint | Meaning |
| --- | --- | --- | --- |
| `mass_kg` | number | `> 0` | Head mass |
| `cg_back_m` | number | `>= 0` | CG behind the shaft axis (face-normal) |
| `cg_toe_m` | number | `>= 0` | CG toe-ward of the shaft axis |
| `cg_drop_m` | number | `>= 0` | CG axial distance below the tip |

## mesh_reference (optional)

Pins an external head STL and the scale needed to reproduce the derived
inertia tensor. Exactly **one** of `density_kg_m3` / `target_mass_kg` must be
present — the same selector `mesh_inertia` enforces.

| Field | Type | Constraint |
| --- | --- | --- |
| `name` | string | Non-empty |
| `sha256` | string | 64 lowercase hex characters (SHA-256 of the STL bytes) |
| `density_kg_m3` | number | `> 0`; mutually exclusive with `target_mass_kg` |
| `target_mass_kg` | number | `> 0`; mutually exclusive with `density_kg_m3` |

The referenced mesh must be watertight with outward winding; the parser of
the mesh itself (`stl_validation`, `mesh_mass_properties`) refuses anything
else. Uniform density is a documented lower-bound proxy for hollow heads —
OEM shell models should supply their measured tensor through the assembly.

## provenance

| Field | Type | Constraint |
| --- | --- | --- |
| `source_kind` | string | One of `oem_export`, `measured`, `parametric`, `cad_derived` |
| `tool_name` | string | Non-empty |
| `exported_at` | string | ISO-8601 date (`YYYY-MM-DD`) or datetime with `Z`/offset |

## Sub-document references

- `assembly`: see `shared/python/golf_club/serialization.py`
  (`golf_club.assembly/1`) — components with mass, CG, inertia tensor,
  declared frames, and a length measurement with explicit datums.
- `shaft_profile`: see `shared/python/golf_club/shaft_serialization.py`
  (`golf_club.shaft_profile/1`) — per-station EI about both axes, GJ,
  linear density, diameters, damping, spine angle, trims and insertion
  depth, plus measurement provenance.

## Minimal producer checklist for OEM integrators

1. Emit exactly the fields above — the parser refuses extras (extend by
   proposing a `/2` format, not by adding fields to `/1`).
2. Use SI units; convert at your boundary, not ours.
3. Hash the exact STL bytes you ship; the reference is refused if the hash
   is not 64 lowercase hex characters.
4. Declare how the shaft was measured in `shaft_profile.provenance` — the
   fitting report carries your identifiers through unchanged.
