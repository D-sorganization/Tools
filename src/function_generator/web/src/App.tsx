import { FunctionGenerator } from './components/FunctionGenerator'

function App() {
  return (
    <div className="min-h-screen bg-slate-900">
      <header className="bg-slate-800 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-white">
            Function Generator
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Generate and visualize various waveforms
          </p>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 py-6">
        <FunctionGenerator />
      </main>
    </div>
  )
}

export default App
