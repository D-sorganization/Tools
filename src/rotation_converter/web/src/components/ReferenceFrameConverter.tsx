import { useCallback, useState } from "react";

type Operation = "twist_frame_conversion" | "homogeneous_transform" | "so3_so3_maps";

interface ReferenceFrameResponse {
  operation: string;
  results: Record<string, unknown>;
  explanation_markdown: string;
  explanation_latex: string;
}

const IDENTITY_4X4 = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

const IDENTITY_3X3 = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

export function ReferenceFrameConverter() {
  const [operation, setOperation] = useState<Operation>("twist_frame_conversion");
  const [transform, setTransform] = useState<number[][]>(IDENTITY_4X4);
  const [twist, setTwist] = useState<number[]>([0, 0, 1, 0.5, 0, 0]);
  const [rotationMatrix, setRotationMatrix] = useState<number[][]>(IDENTITY_3X3);
  const [translation, setTranslation] = useState<number[]>([0, 0, 0]);
  const [so3Vector, setSo3Vector] = useState<number[]>([0, 0, 0.5]);
  const [result, setResult] = useState<ReferenceFrameResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const updateMatrix = useCallback(
    (setter: (rows: number[][]) => void, rows: number[][], i: number, j: number, value: number) => {
      const next = rows.map((row) => [...row]);
      next[i][j] = value;
      setter(next);
    },
    [],
  );

  const handleCompute = useCallback(async () => {
    setError(null);
    setResult(null);

    const payload: Record<string, unknown> = { operation };
    if (operation === "twist_frame_conversion") {
      payload.transform = transform;
      payload.twist = twist;
    } else if (operation === "homogeneous_transform") {
      payload.rotation_matrix = rotationMatrix;
      payload.translation = translation;
    } else {
      payload.so3_vector = so3Vector;
    }

    try {
      const response = await fetch("/api/calc/rotation-converter/reference-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Reference-frame conversion failed");
      }
      setResult(data as ReferenceFrameResponse);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Unknown error";
      setError(message);
    }
  }, [operation, rotationMatrix, so3Vector, transform, translation, twist]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-4">
      <div className="bg-slate-800 rounded-lg p-6 space-y-4 text-white">
        <h2 className="text-xl font-semibold border-b border-slate-700 pb-2">
          Reference-Frame Operations
        </h2>

        {error && <div className="bg-red-900/40 border border-red-500 rounded p-3 text-red-200">{error}</div>}

        <div>
          <label htmlFor="operation-select" className="block text-sm text-slate-300 mb-1">Operation</label>
          <select
            id="operation-select"
            value={operation}
            onChange={(event) => setOperation(event.target.value as Operation)}
            className="w-full bg-slate-700 rounded px-3 py-2 border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          >
            <option value="twist_frame_conversion">Twist Frame Conversion (Adjoint)</option>
            <option value="homogeneous_transform">Homogeneous Transform Builder</option>
            <option value="so3_so3_maps">so(3) ↔ SO(3) Exponential / Log</option>
          </select>
        </div>

        {operation === "twist_frame_conversion" && (
          <div className="space-y-3">
            <p className="text-sm text-slate-300">Homogeneous Transform T (4x4)</p>
            <div className="grid grid-cols-4 gap-2">
              {transform.map((row, i) =>
                row.map((entry, j) => (
                  <input
                    key={`${i}-${j}`}
                    aria-label={`Transform Row ${i + 1}, Column ${j + 1}`}
                    type="number"
                    value={entry}
                    onChange={(event) => updateMatrix(setTransform, transform, i, j, Number(event.target.value))}
                    className="bg-slate-700 rounded px-2 py-1 text-sm border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  />
                )),
              )}
            </div>
            <p className="text-sm text-slate-300">Twist [ωx, ωy, ωz, vx, vy, vz]</p>
            <div className="grid grid-cols-3 gap-2">
              {twist.map((entry, i) => (
                <input
                  key={i}
                  aria-label={["ωx", "ωy", "ωz", "vx", "vy", "vz"][i]}
                  type="number"
                  value={entry}
                  onChange={(event) => {
                    const next = [...twist];
                    next[i] = Number(event.target.value);
                    setTwist(next);
                  }}
                  className="bg-slate-700 rounded px-2 py-1 text-sm border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                />
              ))}
            </div>
          </div>
        )}

        {operation === "homogeneous_transform" && (
          <div className="space-y-3">
            <p className="text-sm text-slate-300">Rotation Matrix R (3x3)</p>
            <div className="grid grid-cols-3 gap-2">
              {rotationMatrix.map((row, i) =>
                row.map((entry, j) => (
                  <input
                    key={`${i}-${j}`}
                    aria-label={`Rotation Matrix Row ${i + 1}, Column ${j + 1}`}
                    type="number"
                    value={entry}
                    onChange={(event) =>
                      updateMatrix(setRotationMatrix, rotationMatrix, i, j, Number(event.target.value))
                    }
                    className="bg-slate-700 rounded px-2 py-1 text-sm border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                  />
                )),
              )}
            </div>
            <p className="text-sm text-slate-300">Translation p (3)</p>
            <div className="grid grid-cols-3 gap-2">
              {translation.map((entry, i) => (
                <input
                  key={i}
                  aria-label={["x", "y", "z"][i]}
                  type="number"
                  value={entry}
                  onChange={(event) => {
                    const next = [...translation];
                    next[i] = Number(event.target.value);
                    setTranslation(next);
                  }}
                  className="bg-slate-700 rounded px-2 py-1 text-sm border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                />
              ))}
            </div>
          </div>
        )}

        {operation === "so3_so3_maps" && (
          <div className="space-y-3">
            <p className="text-sm text-slate-300">so(3) Vector (rotation vector / ωθ)</p>
            <div className="grid grid-cols-3 gap-2">
              {so3Vector.map((entry, i) => (
                <input
                  key={i}
                  aria-label={["x", "y", "z"][i]}
                  type="number"
                  value={entry}
                  onChange={(event) => {
                    const next = [...so3Vector];
                    next[i] = Number(event.target.value);
                    setSo3Vector(next);
                  }}
                  className="bg-slate-700 rounded px-2 py-1 text-sm border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                />
              ))}
            </div>
          </div>
        )}

        <button
          onClick={handleCompute}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-800"
        >
          Compute
        </button>
      </div>

      <div className="bg-slate-800 rounded-lg p-6 text-white space-y-4">
        <h2 className="text-xl font-semibold border-b border-slate-700 pb-2">Educational Output</h2>
        {!result ? (
          <p className="text-slate-400 text-sm italic">
            Choose an operation and compute to see matrices, vectors, and derivations.
          </p>
        ) : (
          <div className="space-y-4">
            <div className="bg-slate-700/50 p-3 rounded">
              <p className="text-xs text-slate-400 mb-1">Results (JSON)</p>
              <pre className="font-mono text-xs whitespace-pre-wrap">{JSON.stringify(result.results, null, 2)}</pre>
            </div>
            <div className="bg-slate-700/50 p-3 rounded">
              <p className="text-xs text-slate-400 mb-1">Explanation (Markdown)</p>
              <pre className="font-mono text-xs whitespace-pre-wrap">{result.explanation_markdown}</pre>
            </div>
            <div className="bg-slate-700/50 p-3 rounded">
              <p className="text-xs text-slate-400 mb-1">Formulas (LaTeX)</p>
              <pre className="font-mono text-xs whitespace-pre-wrap">{result.explanation_latex}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
