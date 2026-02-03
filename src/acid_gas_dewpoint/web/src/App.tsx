import { AcidGasDewpointCalculator } from './components/AcidGasDewpointCalculator'

function App() {
  return (
    <div className="min-h-screen bg-slate-900">
      <header className="bg-slate-800 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-white">Acid Gas Dewpoint Calculator</h1>
          <p className="text-slate-400 text-sm mt-1">HF, HCl, H2S dewpoint analysis for syngas applications</p>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 py-6">
        <AcidGasDewpointCalculator />
      </main>
    </div>
  )
}

export default App
