import { useCallback, useState } from 'react';

// Common interfaces mapping to the FastAPI Response/Request
interface RotationRepresentations {
    quaternion: number[];
    euler: number[];
    euler_convention: string;
    axis_angle: {
        axis: number[];
        angle: number;
    };
    rodrigues: number[];
    rotation_matrix: number[][];
}

interface ConversionResponse {
    representations: RotationRepresentations;
}

export function RotationConverter() {
    const [inputType, setInputType] = useState('quaternion');
    const [quaternion, setQuaternion] = useState([1, 0, 0, 0]);
    const [euler, setEuler] = useState([0, 0, 0]);
    const [eulerConvention, setEulerConvention] = useState('xyz');
    const [axisAngle, setAxisAngle] = useState([1, 0, 0, 0]); // axis (x, y, z) + angle
    const [rodrigues, setRodrigues] = useState([0, 0, 0]);

    const [results, setResults] = useState<RotationRepresentations | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleCalculate = useCallback(async () => {
        setError(null);
        let payloadValue: number[] = quaternion;

        if (inputType === 'quaternion') {
            payloadValue = quaternion;
        } else if (inputType === 'euler') {
            payloadValue = euler;
        } else if (inputType === 'axis_angle') {
            payloadValue = axisAngle;
        } else if (inputType === 'rodrigues') {
            payloadValue = rodrigues;
        }

        try {
            // NOTE: In UpstreamDrift the API_BASE will be injected or relative
            const response = await fetch('/api/calc/rotation-converter', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    type: inputType,
                    value: payloadValue,
                    euler_convention: eulerConvention,
                }),
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to convert rotation');
            }

            const data: ConversionResponse = await response.json();
            setResults(data.representations);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Failed to convert rotation';
            setError(message);
        }
    }, [inputType, quaternion, euler, eulerConvention, axisAngle, rodrigues]);

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-4">
            {/* Input Panel */}
            <div className="bg-slate-800 rounded-lg p-6 space-y-4 text-white">
                <h2 className="text-xl font-semibold mb-4 border-b border-slate-700 pb-2">Input Orientation</h2>

                {error && (
                    <div className="bg-red-900/50 border border-red-500 rounded p-3 text-red-200 text-sm">
                        {error}
                    </div>
                )}

                <div>
                    <label htmlFor="input-type-select" className="block text-sm text-slate-300 mb-1">Input Type</label>
                    <select
                        id="input-type-select"
                        value={inputType}
                        onChange={(e) => setInputType(e.target.value)}
                        className="w-full bg-slate-700 rounded px-3 py-2 border border-slate-600 focus:border-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                    >
                        <option value="quaternion">Quaternion (w, x, y, z)</option>
                        <option value="euler">Euler Angles</option>
                        <option value="axis_angle">Axis-Angle</option>
                        <option value="rodrigues">Rodrigues Vector</option>
                    </select>
                </div>

                {inputType === 'quaternion' && (
                    <div className="grid grid-cols-4 gap-2">
                        {['w', 'x', 'y', 'z'].map((axis, i) => (
                            <div key={axis}>
                                <label className="block text-xs text-slate-400 mb-1">{axis.toUpperCase()}</label>
                                <input
                                    aria-label={`Quaternion ${axis.toUpperCase()}`}
                                    type="number"
                                    value={quaternion[i]}
                                    onChange={(e) => {
                                        const newQ = [...quaternion];
                                        newQ[i] = Number(e.target.value);
                                        setQuaternion(newQ);
                                    }}
                                    step="0.01"
                                    className="w-full bg-slate-700 rounded px-2 py-1 text-sm border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                                />
                            </div>
                        ))}
                    </div>
                )}

                {inputType === 'euler' && (
                    <div className="space-y-4">
                        <div>
                            <label htmlFor="euler-convention-input" className="block text-sm text-slate-300 mb-1">Convention</label>
                            <input
                                id="euler-convention-input"
                                type="text"
                                value={eulerConvention}
                                onChange={(e) => setEulerConvention(e.target.value)}
                                placeholder="xyz"
                                className="w-full bg-slate-700 rounded px-3 py-2 text-sm border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                            />
                        </div>
                        <div className="grid grid-cols-3 gap-2">
                            {['a', 'b', 'c'].map((label, i) => (
                                <div key={label}>
                                    <label className="block text-xs text-slate-400 mb-1">Axis {i + 1} (rad)</label>
                                    <input
                                        aria-label={`Euler Axis ${i + 1}`}
                                        type="number"
                                        value={euler[i]}
                                        onChange={(e) => {
                                            const newE = [...euler];
                                            newE[i] = Number(e.target.value);
                                            setEuler(newE);
                                        }}
                                        step="0.01"
                                        className="w-full bg-slate-700 rounded px-2 py-1 text-sm border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                                    />
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {inputType === 'axis_angle' && (
                    <div className="grid grid-cols-4 gap-2">
                        {['x', 'y', 'z', 'angle (rad)'].map((label, i) => (
                            <div key={label}>
                                <label className="block text-xs text-slate-400 mb-1">{label}</label>
                                <input
                                    aria-label={`Axis-Angle ${label}`}
                                    type="number"
                                    value={axisAngle[i]}
                                    onChange={(e) => {
                                        const newA = [...axisAngle];
                                        newA[i] = Number(e.target.value);
                                        setAxisAngle(newA);
                                    }}
                                    step="0.01"
                                    className="w-full bg-slate-700 rounded px-2 py-1 text-sm border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                                />
                            </div>
                        ))}
                    </div>
                )}

                {inputType === 'rodrigues' && (
                    <div className="grid grid-cols-3 gap-2">
                        {['rx', 'ry', 'rz'].map((label, i) => (
                            <div key={label}>
                                <label className="block text-xs text-slate-400 mb-1">{label}</label>
                                <input
                                    aria-label={`Rodrigues ${label}`}
                                    type="number"
                                    value={rodrigues[i]}
                                    onChange={(e) => {
                                        const newR = [...rodrigues];
                                        newR[i] = Number(e.target.value);
                                        setRodrigues(newR);
                                    }}
                                    step="0.01"
                                    className="w-full bg-slate-700 rounded px-2 py-1 text-sm border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                                />
                            </div>
                        ))}
                    </div>
                )}

                <div className="pt-4">
                    <button
                        onClick={handleCalculate}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-800"
                    >
                        Compute Equivalents
                    </button>
                </div>
            </div>

            {/* Results Panel */}
            <div className="bg-slate-800 rounded-lg p-6 space-y-4 text-white">
                <h2 className="text-xl font-semibold mb-4 border-b border-slate-700 pb-2">Conversions</h2>

                {!results ? (
                    <div className="text-slate-400 text-sm italic">
                        Provide an initial rotation and calculate to see all output formats here.
                    </div>
                ) : (
                    <div className="space-y-4">

                        {/* Quaternion */}
                        <div className="bg-slate-700/50 p-3 rounded">
                            <p className="text-xs text-slate-400 mb-1">Quaternion (w, x, y, z)</p>
                            <p className="font-mono text-sm">
                                [{results.quaternion.map(n => n.toFixed(4)).join(', ')}]
                            </p>
                        </div>

                        {/* Euler */}
                        <div className="bg-slate-700/50 p-3 rounded">
                            <p className="text-xs text-slate-400 mb-1">Euler Angles ({results.euler_convention})</p>
                            <p className="font-mono text-sm">
                                [{results.euler.map(n => n.toFixed(4)).join(', ')}]
                            </p>
                        </div>

                        {/* Axis-Angle */}
                        <div className="bg-slate-700/50 p-3 rounded">
                            <p className="text-xs text-slate-400 mb-1">Axis-Angle</p>
                            <p className="font-mono text-sm">
                                Axis: [{results.axis_angle.axis.map(n => n.toFixed(4)).join(', ')}]
                            </p>
                            <p className="font-mono text-sm">
                                Angle: {results.axis_angle.angle.toFixed(4)} rad
                            </p>
                        </div>

                        {/* Rodrigues */}
                        <div className="bg-slate-700/50 p-3 rounded">
                            <p className="text-xs text-slate-400 mb-1">Rodrigues Vector</p>
                            <p className="font-mono text-sm">
                                [{results.rodrigues.map(n => n.toFixed(4)).join(', ')}]
                            </p>
                        </div>

                        {/* Rotation Matrix */}
                        <div className="bg-slate-700/50 p-3 rounded overflow-x-auto">
                            <p className="text-xs text-slate-400 mb-1">Rotation Matrix (3x3)</p>
                            <pre className="font-mono text-sm bg-slate-900/50 p-2 rounded">
                                {results.rotation_matrix.map(row =>
                                    `[ ${row.map(n => n.toFixed(4).padStart(7, ' ')).join(', ')} ]`
                                ).join('\n')}
                            </pre>
                        </div>

                    </div>
                )}
            </div>
        </div>
    );
}
