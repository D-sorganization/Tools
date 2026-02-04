import { WGSReactorCalculator } from './components/WGSReactorCalculator'

function App() {
  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-blue-400">Water-Gas Shift Reactor Calculator</h1>
        <p className="text-slate-400">Equilibrium composition and reactor sizing for WGS reactions</p>
      </header>
      <main>
        <WGSReactorCalculator />
      </main>
    </div>
  )
}

export default App
