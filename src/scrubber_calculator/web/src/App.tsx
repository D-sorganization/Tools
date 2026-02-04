import { ScrubberCalculator } from './components/ScrubberCalculator'

function App() {
  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-blue-400">Packed Bed Scrubber Calculator</h1>
        <p className="text-slate-400">NTU/HTU-based mass transfer design for acid gas removal</p>
      </header>
      <main>
        <ScrubberCalculator />
      </main>
    </div>
  )
}

export default App
