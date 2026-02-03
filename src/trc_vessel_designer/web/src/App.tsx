import TRCVesselDesignerCalculator from './components/TRCVesselDesignerCalculator';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">TRC Vessel Designer</h1>
          <p className="text-gray-600 mt-2">
            Thermal Reaction Chamber design tool with SVG visualization,
            volume calculations, and residence time estimation
          </p>
        </header>

        <main>
          <TRCVesselDesignerCalculator />
        </main>

        <footer className="mt-8 text-center text-sm text-gray-500">
          TRC Vessel Designer v1.0.0 - Consolidated from Tools Repository
        </footer>
      </div>
    </div>
  );
}

export default App;
