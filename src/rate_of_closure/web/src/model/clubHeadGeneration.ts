import { parametricHeadMesh, type ClubSpec, type Vec3 } from "./club";
import { hoselPoint } from "./clubHeads";
import { type HeadMesh } from "./mesh";
import { headCog } from "./volumetrics";

/** A generated representative head with its hosel and volumetric COG. */
export interface GeneratedHead {
  label: string;
  mesh: HeadMesh;
  hosel: Vec3;
  cog: Vec3;
}

/** Build all renderer metadata for one deterministic representative head. */
export function generatedHeadFor(spec: ClubSpec): GeneratedHead {
  return {
    label: spec.name,
    mesh: parametricHeadMesh(spec),
    hosel: hoselPoint(spec),
    cog: headCog(spec).cog,
  };
}
