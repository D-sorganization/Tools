import type { Triangle } from "./mesh";

export function writeBinaryStlFixture(): ArrayBuffer {
  const triangles: Triangle[] = [
    [[0, 0, 0], [2, 0, 0], [0, 4, 0]],
    [[0, 0, 6], [0, 4, 6], [2, 0, 6]],
    [[0, 0, 0], [0, 0, 6], [2, 0, 0]],
    [[2, 4, 0], [2, 0, 6], [0, 4, 0]],
  ];
  const buffer = new ArrayBuffer(84 + triangles.length * 50);
  const view = new DataView(buffer);
  view.setUint32(80, triangles.length, true);
  triangles.forEach((triangle, triangleIndex) => triangle.forEach((vertex, vertexIndex) => {
    const base = 84 + triangleIndex * 50 + 12 + vertexIndex * 12;
    vertex.forEach((coordinate, axis) => view.setFloat32(base + axis * 4, coordinate, true));
  }));
  return buffer;
}
