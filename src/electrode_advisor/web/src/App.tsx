import { useState } from 'react';
import ElectrodeAdvisorCalculator from './components/ElectrodeAdvisorCalculator';
import GlassBath3DViewer from './components/GlassBath3DViewer';

type ViewTab = '2d' | '3d';

function App() {
  const [activeView, setActiveView] = useState<ViewTab>('2d');

  // Shared state that feeds into the 3D viewer
  const [electrodeType] = useState('graphite_standard');
  const [currentLength] = useState(1500);
  const [wornLength] = useState(150);
  const [operatingCurrent] = useState(2500);
  const [plasmaTemp] = useState(1500);

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

        {/* View Toggle */}
        <div className="flex space-x-1 bg-gray-200 p-1 rounded-lg mb-6 max-w-xs">
          <button
            onClick={() => setActiveView('2d')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              activeView === '2d'
                ? 'bg-white shadow text-gray-900'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            2D Calculator
          </button>
          <button
            onClick={() => setActiveView('3d')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              activeView === '3d'
                ? 'bg-white shadow text-gray-900'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            3D Visualization
          </button>
        </div>

        <main>
          {activeView === '2d' && <ElectrodeAdvisorCalculator />}
          {activeView === '3d' && (
            <GlassBath3DViewer
              electrodeType={electrodeType}
              currentLength={currentLength}
              wornLength={wornLength}
              operatingCurrent={operatingCurrent}
              plasmaTemp={plasmaTemp}
            />
          )}
        </main>

        <footer className="mt-8 text-center text-sm text-gray-500">
          Electrode Advisor v2.0.0 - 3D Visualization (see issue #606)
        </footer>
      </div>
    </div>
  );
}

export default App;
