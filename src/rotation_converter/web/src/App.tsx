import { useState } from "react";
import { RotationConverter } from "./components/RotationConverter";
import { ReferenceFrameConverter } from "./components/ReferenceFrameConverter";

function App() {
  const [activeTab, setActiveTab] = useState<"rotation" | "reference">("rotation");

  return (
    <div className="min-h-screen bg-slate-900">
      <header className="bg-slate-800 shadow-lg">
        <div className="mx-auto max-w-7xl px-4 py-4">
          <h1 className="text-2xl font-bold text-white">Rotation Converter</h1>
          <p className="mt-1 text-sm text-slate-400">
            Educational conversion tool for rotations, twists, SE(3), and so(3)/SO(3) maps.
          </p>
          <div className="mt-4 flex gap-2">
            <button
              className={`px-3 py-1 rounded text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
                activeTab === "rotation" ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-200 hover:bg-slate-600"
              }`}
              onClick={() => setActiveTab("rotation")}
              aria-pressed={activeTab === "rotation"}
            >
              Rotation Formats
            </button>
            <button
              className={`px-3 py-1 rounded text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
                activeTab === "reference" ? "bg-blue-600 text-white" : "bg-slate-700 text-slate-200 hover:bg-slate-600"
              }`}
              onClick={() => setActiveTab("reference")}
              aria-pressed={activeTab === "reference"}
            >
              Reference Frames & Lie Groups
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">
        {activeTab === "rotation" ? <RotationConverter /> : <ReferenceFrameConverter />}
      </main>
    </div>
  );
}

export default App;
