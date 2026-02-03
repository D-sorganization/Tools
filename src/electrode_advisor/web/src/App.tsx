import ElectrodeAdvisorCalculator from './components/ElectrodeAdvisorCalculator';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Electrode Advisor</h1>
          <p className="text-gray-600 mt-2">
            AC Electrode Advancement Module - Electrode positioning guidance, wear analysis, and
            replacement scheduling
          </p>
        </header>

        <main>
          <ElectrodeAdvisorCalculator />
        </main>

        <footer className="mt-8 text-center text-sm text-gray-500">
          Electrode Advisor v1.0.0 - Consolidated from Tools Repository
        </footer>
      </div>
    </div>
  );
}

export default App;
