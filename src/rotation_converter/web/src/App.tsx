import { RotationConverter } from "./components/RotationConverter";

function App() {
  return (
    <div className="min-h-screen bg-slate-900">
      <header className="bg-slate-800 shadow-lg">
        <div className="mx-auto max-w-7xl px-4 py-4">
          <h1 className="text-2xl font-bold text-white">Rotation Converter</h1>
          <p className="mt-1 text-sm text-slate-400">
            Convert between quaternion, Euler, axis-angle, Rodrigues, and rotation
            matrix forms.
          </p>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">
        <RotationConverter />
      </main>
    </div>
  );
}

export default App;
