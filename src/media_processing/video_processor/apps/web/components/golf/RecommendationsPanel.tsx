"use client";

interface RecommendationsPanelProps {
  recommendations: string[];
}

export default function RecommendationsPanel({
  recommendations,
}: RecommendationsPanelProps) {
  if (recommendations.length === 0) {
    return null;
  }

  // Categorize recommendations
  const prioritized = recommendations.filter(
    (r) =>
      r.toLowerCase().includes("focus") || r.toLowerCase().includes("major"),
  );
  const drills = recommendations.filter(
    (r) =>
      r.toLowerCase().includes("drill") || r.toLowerCase().includes("practice"),
  );
  const general = recommendations.filter(
    (r) => !prioritized.includes(r) && !drills.includes(r),
  );

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Recommendations
      </h3>

      <div className="space-y-4">
        {/* Priority Recommendations */}
        {prioritized.length > 0 && (
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <svg
                  className="w-6 h-6 text-blue-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
              </div>
              <div>
                <h4 className="text-sm font-medium text-blue-800 mb-2">
                  Priority Focus
                </h4>
                <ul className="space-y-2">
                  {prioritized.map((rec, index) => (
                    <li
                      key={index}
                      className="text-sm text-blue-700 flex items-start"
                    >
                      <span className="mr-2">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Drill Recommendations */}
        {drills.length > 0 && (
          <div className="bg-purple-50 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <svg
                  className="w-6 h-6 text-purple-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"
                  />
                </svg>
              </div>
              <div>
                <h4 className="text-sm font-medium text-purple-800 mb-2">
                  Practice Drills
                </h4>
                <ul className="space-y-2">
                  {drills.map((rec, index) => (
                    <li
                      key={index}
                      className="text-sm text-purple-700 flex items-start"
                    >
                      <span className="mr-2">→</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* General Recommendations */}
        {general.length > 0 && (
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <svg
                  className="w-6 h-6 text-gray-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-800 mb-2">
                  General Tips
                </h4>
                <ul className="space-y-2">
                  {general.map((rec, index) => (
                    <li
                      key={index}
                      className="text-sm text-gray-700 flex items-start"
                    >
                      <span className="mr-2">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Quick Tips Section */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <h4 className="text-sm font-medium text-gray-700 mb-3">
          Quick Reference
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
            <span className="text-2xl">🎯</span>
            <div>
              <p className="text-sm font-medium text-green-800">Ideal Tempo</p>
              <p className="text-xs text-green-600">
                3:1 ratio (backswing to downswing)
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
            <span className="text-2xl">🔄</span>
            <div>
              <p className="text-sm font-medium text-green-800">
                Ideal X-Factor
              </p>
              <p className="text-xs text-green-600">
                45-60° shoulder-hip separation
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
            <span className="text-2xl">🧍</span>
            <div>
              <p className="text-sm font-medium text-green-800">
                Ideal Spine Angle
              </p>
              <p className="text-xs text-green-600">
                30-45° forward tilt at address
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
            <span className="text-2xl">⚖️</span>
            <div>
              <p className="text-sm font-medium text-green-800">
                Weight at Impact
              </p>
              <p className="text-xs text-green-600">70-80% on lead side</p>
            </div>
          </div>
        </div>
      </div>

      {/* Action Button */}
      <div className="mt-6">
        <button className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors flex items-center justify-center space-x-2">
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
            />
          </svg>
          <span>View Training Resources</span>
        </button>
      </div>
    </div>
  );
}
