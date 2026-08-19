"""Closed-mesh mass properties via the divergence theorem (club-tester C1, #4550).

Shared authority for watertightness, volume, centroid, and the full
inertia tensor of triangle meshes. ``rate_of_closure.club.volumetrics``
delegates here; UpstreamDrift reaches the same implementation through
``vendor/ud-tools`` — per the fleet DRY rule, mesh mass-property math
lives only in this module.

For a watertight, outward-wound mesh, each triangle ``(a, b, c)`` forms a
signed origin-tetrahedron with volume ``s = det(a, b, c) / 6``:

    V    = Σ s
    COG  = Σ s · (a + b + c) / 4  /  V
    P_jk = Σ s / 20 · ( a_j a_k + b_j b_k + c_j c_k + t_j t_k ),  t = a + b + c

(the origin vertex contributes zero everywhere). These are exact for
polyhedra and origin-independent for closed meshes. The center-of-gravity
tensor follows from the second-moment parallel-axis shift
``P_cg = P − V·c⊗c`` and ``I_jk = ρ·(δ_jk·tr(P_cg) − P_cg_jk)``, verified
in tests against the closed forms for a cube (``m·L²/6``), an offset box
(``m/12·(b²+c², …)``), and a UV sphere (``2/5·m·r²``), plus translation
invariance and rotation covariance (``I → R·I·Rᵀ``).

**Uniform-density semantics.** The mesh is treated as a solid of uniform
density. For hollow driver heads this is a *lower-bound proxy*: a shell of
equal mass carries more of it at the perimeter and therefore a larger MOI.
OEM shell models should supply their measured tensor through the
``ClubAssembly`` path; this module is the CAD-derived fallback and the
authority for solid or effectively solid components.

Design by Contract: watertightness is a combinatorial directed-edge check
(every directed edge exactly once with its reverse present, exact-bit
vertex matching); exactly one of ``density_kg_m3`` / ``mass_kg`` selects
the inertia scale; results must be finite, positive, and — for the
tensor — symmetric positive-definite with principal moments obeying the
triangle inequality every physical inertia tensor satisfies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.python.contracts import ensure, require

__all__ = [
    "MeshInertiaReport",
    "is_watertight",
    "mesh_inertia",
    "mesh_volume_centroid",
]


def _directed_edges(triangles: np.ndarray) -> dict[tuple[bytes, bytes], int]:
    """Count of each directed edge, keyed by exact vertex bytes."""
    edges: dict[tuple[bytes, bytes], int] = {}
    for tri in triangles:
        keys = [np.ascontiguousarray(v).tobytes() for v in tri]
        for i in range(3):
            edge = (keys[i], keys[(i + 1) % 3])
            edges[edge] = edges.get(edge, 0) + 1
    return edges


def is_watertight(triangles: np.ndarray) -> bool:
    """Whether every directed edge appears once with its reverse present.

    Exact-bit vertex matching: generated meshes share ring vertices
    bit-for-bit, so this is a true closure check for them; independently
    authored STLs with re-tessellated seams may fail and should fall back
    to declared mass properties.
    """
    tris = np.asarray(triangles, dtype=np.float64)
    require(tris.ndim == 3 and tris.shape[1:] == (3, 3), "triangles must be (n, 3, 3)")
    edges = _directed_edges(tris)
    return all(
        count == 1 and edges.get((b, a), 0) == 1 for (a, b), count in edges.items()
    )


def mesh_volume_centroid(triangles: np.ndarray) -> tuple[float, np.ndarray]:
    """Volume [m³] and centroid [m] of a closed, outward-wound mesh.

    Raises:
        PreconditionError: If the mesh is not watertight.
        PostconditionError: If the signed volume is not positive/finite
            (inward winding or a degenerate solid).
    """
    tris = np.asarray(triangles, dtype=np.float64)
    require(tris.ndim == 3 and tris.shape[1:] == (3, 3), "triangles must be (n, 3, 3)")
    require(bool(np.isfinite(tris).all()), "triangles must be finite")
    require(is_watertight(tris), "mesh must be watertight (closed, matched edges)")

    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    signed = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = float(signed.sum())
    ensure(np.isfinite(volume) and volume > 0.0, "volume must be positive", volume)
    centroid: np.ndarray = (signed[:, None] * (a + b + c) / 4.0).sum(axis=0) / volume
    ensure(bool(np.isfinite(centroid).all()), "centroid must be finite")
    return volume, centroid


def _second_moment_about_origin(triangles: np.ndarray) -> np.ndarray:
    """Exact polyhedral second-moment tensor ∫ x_j x_k dV about the origin."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    signed = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    t = a + b + c
    outer = (
        np.einsum("ij,ik->ijk", a, a)
        + np.einsum("ij,ik->ijk", b, b)
        + np.einsum("ij,ik->ijk", c, c)
        + np.einsum("ij,ik->ijk", t, t)
    )
    second: np.ndarray = np.einsum("i,ijk->jk", signed / 20.0, outer)
    return second


@dataclass(frozen=True)
class MeshInertiaReport:
    """Uniform-density mass properties of a closed mesh, SI, mesh frame.

    Attributes:
        volume_m3: Enclosed volume.
        centroid_m: Center of gravity in the mesh frame.
        mass_kg: Total mass at the resolved density.
        density_kg_m3: The resolved uniform density.
        inertia_at_cog_kg_m2: 3×3 inertia tensor about the center of
            gravity, in mesh-frame axes.
        principal_moments_kg_m2: Eigenvalues of the tensor, ascending.
    """

    volume_m3: float
    centroid_m: tuple[float, float, float]
    mass_kg: float
    density_kg_m3: float
    inertia_at_cog_kg_m2: tuple[tuple[float, float, float], ...]
    principal_moments_kg_m2: tuple[float, float, float]

    def inertia_array(self) -> np.ndarray:
        """The CG tensor as a (3, 3) float array."""
        tensor: np.ndarray = np.asarray(self.inertia_at_cog_kg_m2, dtype=np.float64)
        return tensor


def mesh_inertia(
    triangles: np.ndarray,
    *,
    density_kg_m3: float | None = None,
    mass_kg: float | None = None,
) -> MeshInertiaReport:
    """Uniform-density inertia tensor of a closed, outward-wound mesh.

    Exactly one of ``density_kg_m3`` (measured material density) or
    ``mass_kg`` (target mass; density solved from the enclosed volume)
    must be given, and it must be positive and finite.

    Raises:
        PreconditionError: If the scale selection is invalid or the mesh
            fails the watertight / winding / shape checks.
        PostconditionError: If the resulting tensor is not a physical
            inertia tensor (symmetric positive-definite, principal
            moments obeying the triangle inequalities).
    """
    require(
        (density_kg_m3 is None) != (mass_kg is None),
        "exactly one of density_kg_m3 or mass_kg must be given",
    )
    volume, centroid = mesh_volume_centroid(triangles)
    if density_kg_m3 is not None:
        require(
            isinstance(density_kg_m3, float | int)
            and np.isfinite(density_kg_m3)
            and density_kg_m3 > 0.0,
            "density_kg_m3 must be positive and finite",
            density_kg_m3,
        )
        density = float(density_kg_m3)
        mass = density * volume
    elif isinstance(mass_kg, float | int):
        require(
            bool(np.isfinite(mass_kg)) and mass_kg > 0.0,
            "mass_kg must be positive and finite",
            mass_kg,
        )
        mass = float(mass_kg)
        density = mass / volume
    else:
        raise TypeError("mass_kg must be a number")

    tris = np.asarray(triangles, dtype=np.float64)
    second = _second_moment_about_origin(tris)
    second_cg = second - volume * np.outer(centroid, centroid)
    inertia = density * (np.trace(second_cg) * np.eye(3) - second_cg)
    inertia = (inertia + inertia.T) / 2.0

    ensure(bool(np.isfinite(inertia).all()), "inertia tensor must be finite")
    moments = np.linalg.eigvalsh(inertia)
    ensure(bool((moments > 0.0).all()), "principal moments must be positive")
    i1, i2, i3 = (float(m) for m in moments)
    ensure(
        i1 + i2 >= i3 * (1.0 - 1e-9),
        "principal moments must satisfy the triangle inequality",
        (i1, i2, i3),
    )
    return MeshInertiaReport(
        volume_m3=volume,
        centroid_m=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        mass_kg=mass,
        density_kg_m3=density,
        inertia_at_cog_kg_m2=tuple(
            (float(row[0]), float(row[1]), float(row[2])) for row in inertia
        ),
        principal_moments_kg_m2=(i1, i2, i3),
    )
