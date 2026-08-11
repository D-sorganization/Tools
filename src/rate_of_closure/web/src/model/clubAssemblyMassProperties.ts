/** Strict shared ClubAssembly wire validation and deterministic mass assembly. */

export type Vec3 = [number, number, number];
export type Matrix3 = [Vec3, Vec3, Vec3];

export interface ComponentMassPropertiesRecord {
  component_id: string;
  role: string;
  frame_id: string;
  mass_kg: number;
  center_of_mass_m: Vec3;
  inertia_at_com_kg_m2: Matrix3;
}

export interface RigidTransformRecord {
  from_frame_id: string;
  to_frame_id: string;
  rotation: Matrix3;
  translation_m: Vec3;
}

export interface ClubComponentRecord {
  mass_properties: ComponentMassPropertiesRecord;
  transform_to_club: RigidTransformRecord;
}

export interface ClubAssemblyRecord {
  format: "golf_club.assembly/1";
  assembly_id: string;
  frame_id: string;
  components: ClubComponentRecord[];
  club_length: {
    length_m: number;
    convention: "declared_datums" | "sixty_degree_sole_plane";
    measurement_frame_id: string;
    lower_reference_id: string;
    upper_reference_id: string;
  };
}

export interface AssembledMassPropertiesRecord {
  frameId: string;
  componentIds: string[];
  totalMassKg: number;
  centerOfMassM: Vec3;
  inertiaAtComKgM2: Matrix3;
}

const COMPONENT_ROLES = new Set([
  "head",
  "shaft",
  "grip",
  "adapter",
  "ferrule",
  "added_weight",
]);
const TOLERANCE = 1e-10;

function record(value: unknown, name: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be a JSON object`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  name: string,
): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    throw new Error(`${name} fields do not match schema`);
  }
}

function identifier(value: unknown, name: string): string {
  if (typeof value !== "string" || !value || value.trim() !== value) {
    throw new Error(`${name} must be a nonempty trimmed string`);
  }
  return value;
}

function finiteNumber(value: unknown, name: string, positive = false): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be finite`);
  }
  if (positive && value <= 0) throw new Error(`${name} must be positive`);
  return value;
}

function vector3(value: unknown, name: string): Vec3 {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`${name} must contain three values`);
  }
  return value.map((entry, index) =>
    finiteNumber(entry, `${name}[${index}]`),
  ) as Vec3;
}

function matrix3(value: unknown, name: string): Matrix3 {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`${name} must contain three rows`);
  }
  return value.map((row, index) =>
    vector3(row, `${name}[${index}]`),
  ) as Matrix3;
}

function determinant(matrix: Matrix3): number {
  const [a, b, c] = matrix;
  return (
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    a[1] * (b[0] * c[2] - b[2] * c[0]) +
    a[2] * (b[0] * c[1] - b[1] * c[0])
  );
}

function transpose(matrix: Matrix3): Matrix3 {
  return matrix[0].map((_, column) =>
    matrix.map((row) => row[column]),
  ) as Matrix3;
}

function multiply(left: Matrix3, right: Matrix3): Matrix3 {
  const rightT = transpose(right);
  return left.map((row) =>
    rightT.map((column) =>
      row.reduce((sum, value, index) => sum + value * column[index], 0),
    ),
  ) as Matrix3;
}

function matrixVector(matrix: Matrix3, vector: Vec3): Vec3 {
  return matrix.map((row) =>
    row.reduce((sum, value, index) => sum + value * vector[index], 0),
  ) as Vec3;
}

function symmetricEigenvalues(matrix: Matrix3): Vec3 {
  const traceThird = (matrix[0][0] + matrix[1][1] + matrix[2][2]) / 3;
  const offDiagonal = matrix[0][1] ** 2 + matrix[0][2] ** 2 + matrix[1][2] ** 2;
  const spread =
    (matrix[0][0] - traceThird) ** 2 +
    (matrix[1][1] - traceThird) ** 2 +
    (matrix[2][2] - traceThird) ** 2 +
    2 * offDiagonal;
  const scale = Math.sqrt(spread / 6);
  if (scale === 0) return [traceThird, traceThird, traceThird];
  const centered = matrix.map((row, i) =>
    row.map((value, j) => (value - (i === j ? traceThird : 0)) / scale),
  ) as Matrix3;
  const angle =
    Math.acos(Math.max(-1, Math.min(1, determinant(centered) / 2))) / 3;
  const largest = traceThird + 2 * scale * Math.cos(angle);
  const smallest = traceThird + 2 * scale * Math.cos(angle + (2 * Math.PI) / 3);
  const middle = 3 * traceThird - largest - smallest;
  return [smallest, middle, largest].sort((a, b) => a - b) as Vec3;
}

function validateInertia(
  matrix: Matrix3,
  name: string,
  positive: boolean,
): void {
  for (let row = 0; row < 3; row += 1) {
    for (let column = row + 1; column < 3; column += 1) {
      if (Math.abs(matrix[row][column] - matrix[column][row]) > TOLERANCE) {
        throw new Error(`${name} must be symmetric`);
      }
    }
  }
  const moments = symmetricEigenvalues(matrix);
  if (moments[0] < -TOLERANCE || (positive && moments[0] <= 0)) {
    throw new Error(
      `${name} must be ${positive ? "positive definite" : "positive semidefinite"}`,
    );
  }
  if (moments[2] - moments[1] - moments[0] > TOLERANCE) {
    throw new Error(
      `${name} principal moments must satisfy the triangle inequality`,
    );
  }
}

function validateRotation(rotation: Matrix3): void {
  const product = multiply(transpose(rotation), rotation);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      const expected = row === column ? 1 : 0;
      if (Math.abs(product[row][column] - expected) > TOLERANCE) {
        throw new Error("rotation must be proper orthonormal");
      }
    }
  }
  if (Math.abs(determinant(rotation) - 1) > TOLERANCE) {
    throw new Error("rotation must be proper orthonormal");
  }
}

function parseMassProperties(value: unknown): ComponentMassPropertiesRecord {
  const data = record(value, "mass_properties");
  exactKeys(
    data,
    [
      "component_id",
      "role",
      "frame_id",
      "mass_kg",
      "center_of_mass_m",
      "inertia_at_com_kg_m2",
    ],
    "mass_properties",
  );
  const role = identifier(data.role, "component role");
  if (!COMPONENT_ROLES.has(role))
    throw new Error("component role is unsupported");
  const inertia = matrix3(data.inertia_at_com_kg_m2, "component inertia");
  validateInertia(inertia, "component inertia", false);
  return {
    component_id: identifier(data.component_id, "component_id"),
    role,
    frame_id: identifier(data.frame_id, "component frame_id"),
    mass_kg: finiteNumber(data.mass_kg, "component mass_kg", true),
    center_of_mass_m: vector3(
      data.center_of_mass_m,
      "component center_of_mass_m",
    ),
    inertia_at_com_kg_m2: inertia,
  };
}

export function validateRigidTransform(value: unknown): RigidTransformRecord {
  const data = record(value, "rigid transform");
  exactKeys(
    data,
    ["from_frame_id", "to_frame_id", "rotation", "translation_m"],
    "rigid transform",
  );
  const rotation = matrix3(data.rotation, "rotation");
  validateRotation(rotation);
  return {
    from_frame_id: identifier(data.from_frame_id, "transform from_frame_id"),
    to_frame_id: identifier(data.to_frame_id, "transform to_frame_id"),
    rotation,
    translation_m: vector3(data.translation_m, "transform translation_m"),
  };
}

function parseComponent(
  value: unknown,
  assemblyFrame: string,
): ClubComponentRecord {
  const data = record(value, "component");
  exactKeys(data, ["mass_properties", "transform_to_club"], "component");
  const mass = parseMassProperties(data.mass_properties);
  const transform = validateRigidTransform(data.transform_to_club);
  if (mass.frame_id !== transform.from_frame_id) {
    throw new Error("component frame does not match transform source");
  }
  if (transform.to_frame_id !== assemblyFrame) {
    throw new Error("component transform does not target assembly frame");
  }
  return { mass_properties: mass, transform_to_club: transform };
}

/** Validate the current shared ClubAssembly wire without migration or defaults. */
export function validateClubAssembly(value: unknown): ClubAssemblyRecord {
  const data = record(value, "assembly");
  exactKeys(
    data,
    ["format", "assembly_id", "frame_id", "components", "club_length"],
    "assembly",
  );
  if (data.format !== "golf_club.assembly/1")
    throw new Error("assembly format is unsupported");
  const frameId = identifier(data.frame_id, "assembly frame_id");
  if (!Array.isArray(data.components) || data.components.length === 0) {
    throw new Error("assembly components must be a nonempty array");
  }
  const components = data.components.map((item) =>
    parseComponent(item, frameId),
  );
  const ids = components.map((item) => item.mass_properties.component_id);
  if (new Set(ids).size !== ids.length)
    throw new Error("component IDs must be unique");
  const length = record(data.club_length, "club_length");
  exactKeys(
    length,
    [
      "length_m",
      "convention",
      "measurement_frame_id",
      "lower_reference_id",
      "upper_reference_id",
    ],
    "club_length",
  );
  if (
    length.convention !== "declared_datums" &&
    length.convention !== "sixty_degree_sole_plane"
  ) {
    throw new Error("club length convention is unsupported");
  }
  return {
    format: "golf_club.assembly/1",
    assembly_id: identifier(data.assembly_id, "assembly_id"),
    frame_id: frameId,
    components,
    club_length: {
      length_m: finiteNumber(length.length_m, "club length", true),
      convention: length.convention,
      measurement_frame_id: identifier(
        length.measurement_frame_id,
        "length measurement frame",
      ),
      lower_reference_id: identifier(
        length.lower_reference_id,
        "lower reference",
      ),
      upper_reference_id: identifier(
        length.upper_reference_id,
        "upper reference",
      ),
    },
  };
}

function add(left: Matrix3, right: Matrix3): Matrix3 {
  return left.map((row, i) =>
    row.map((value, j) => value + right[i][j]),
  ) as Matrix3;
}

function shiftedInertia(inertia: Matrix3, mass: number, offset: Vec3): Matrix3 {
  const squared = offset.reduce((sum, value) => sum + value * value, 0);
  return inertia.map((row, i) =>
    row.map(
      (value, j) =>
        value + mass * ((i === j ? squared : 0) - offset[i] * offset[j]),
    ),
  ) as Matrix3;
}

/** Assemble all validated component properties in the declared assembly frame. */
export function assembleMassProperties(
  assembly: ClubAssemblyRecord,
): AssembledMassPropertiesRecord {
  const masses = assembly.components.map(
    (item) => item.mass_properties.mass_kg,
  );
  const centers = assembly.components.map((item) => {
    const transform = item.transform_to_club;
    const rotated = matrixVector(
      transform.rotation,
      item.mass_properties.center_of_mass_m,
    );
    return rotated.map(
      (value, index) => value + transform.translation_m[index],
    ) as Vec3;
  });
  const totalMassKg = masses.reduce((sum, mass) => sum + mass, 0);
  const centerOfMassM = [0, 1, 2].map(
    (axis) =>
      centers.reduce(
        (sum, center, index) => sum + masses[index] * center[axis],
        0,
      ) / totalMassKg,
  ) as Vec3;
  let total: Matrix3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  assembly.components.forEach((item, index) => {
    const rotation = item.transform_to_club.rotation;
    const rotated = multiply(
      multiply(rotation, item.mass_properties.inertia_at_com_kg_m2),
      transpose(rotation),
    );
    const offset = centers[index].map(
      (value, axis) => value - centerOfMassM[axis],
    ) as Vec3;
    total = add(total, shiftedInertia(rotated, masses[index], offset));
  });
  validateInertia(total, "assembly inertia", true);
  return {
    frameId: assembly.frame_id,
    componentIds: assembly.components.map(
      (item) => item.mass_properties.component_id,
    ),
    totalMassKg,
    centerOfMassM,
    inertiaAtComKgM2: total,
  };
}

export function requirePositiveInertia(matrix: Matrix3, name: string): void {
  validateInertia(matrix, name, true);
}

/** Express a head component's complete properties in the selected-head frame. */
export function headPropertiesInSelectedFrame(
  head: ClubComponentRecord,
  componentFromSelected: RigidTransformRecord,
): ComponentMassPropertiesRecord {
  const rotationT = transpose(componentFromSelected.rotation);
  const delta = head.mass_properties.center_of_mass_m.map(
    (value, index) => value - componentFromSelected.translation_m[index],
  ) as Vec3;
  return {
    ...head.mass_properties,
    role: "head",
    frame_id: "rate_of_closure.head",
    center_of_mass_m: matrixVector(rotationT, delta),
    inertia_at_com_kg_m2: multiply(
      multiply(rotationT, head.mass_properties.inertia_at_com_kg_m2),
      componentFromSelected.rotation,
    ),
  };
}
