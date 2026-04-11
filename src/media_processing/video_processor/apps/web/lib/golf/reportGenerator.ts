/**
 * Golf Swing Report Generator
 * Creates comprehensive PDF and HTML reports from swing analysis
 */

import {
  SwingAnalysis,
  SwingComparison,
  SwingReport,
  SwingSession,
  SwingPhase,
} from './types';
import { generateSummary } from './persistence';

/**
 * Generate HTML report content
 */
export function generateHTMLReport(
  session: SwingSession,
  analysis: SwingAnalysis,
  comparison?: SwingComparison
): string {
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Golf Swing Analysis Report - ${new Date(analysis.analysisTimestamp).toLocaleDateString()}</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: #1a1a2e;
      background: #f8f9fa;
      padding: 40px;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      background: white;
      padding: 40px;
      border-radius: 12px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .header {
      text-align: center;
      margin-bottom: 40px;
      padding-bottom: 20px;
      border-bottom: 2px solid #e9ecef;
    }
    .header h1 {
      font-size: 28px;
      color: #1a1a2e;
      margin-bottom: 8px;
    }
    .header .date {
      color: #6c757d;
      font-size: 14px;
    }
    .score-card {
      display: flex;
      justify-content: center;
      margin: 30px 0;
    }
    .overall-score {
      width: 150px;
      height: 150px;
      border-radius: 50%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
    }
    .overall-score .value {
      font-size: 48px;
      font-weight: bold;
      line-height: 1;
    }
    .overall-score .label {
      font-size: 12px;
      opacity: 0.9;
    }
    .section {
      margin: 30px 0;
    }
    .section h2 {
      font-size: 20px;
      color: #1a1a2e;
      margin-bottom: 15px;
      padding-bottom: 8px;
      border-bottom: 1px solid #e9ecef;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
    }
    .metric-card {
      background: #f8f9fa;
      padding: 16px;
      border-radius: 8px;
      text-align: center;
    }
    .metric-card .value {
      font-size: 28px;
      font-weight: bold;
      color: #667eea;
    }
    .metric-card .label {
      font-size: 12px;
      color: #6c757d;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .score-bar {
      display: flex;
      align-items: center;
      margin: 10px 0;
    }
    .score-bar .label {
      width: 100px;
      font-size: 14px;
    }
    .score-bar .bar {
      flex: 1;
      height: 12px;
      background: #e9ecef;
      border-radius: 6px;
      overflow: hidden;
      margin: 0 12px;
    }
    .score-bar .fill {
      height: 100%;
      border-radius: 6px;
      transition: width 0.3s ease;
    }
    .score-bar .value {
      width: 40px;
      text-align: right;
      font-weight: bold;
      font-size: 14px;
    }
    .issue {
      padding: 12px;
      margin: 8px 0;
      border-radius: 8px;
      border-left: 4px solid;
    }
    .issue.major {
      background: #fff5f5;
      border-color: #e53e3e;
    }
    .issue.moderate {
      background: #fffff0;
      border-color: #d69e2e;
    }
    .issue.minor {
      background: #f0fff4;
      border-color: #38a169;
    }
    .issue .title {
      font-weight: bold;
      margin-bottom: 4px;
    }
    .issue .description {
      font-size: 14px;
      color: #4a5568;
    }
    .issue .drill {
      font-size: 13px;
      color: #667eea;
      margin-top: 8px;
      font-style: italic;
    }
    .recommendations {
      background: #f0f7ff;
      padding: 20px;
      border-radius: 8px;
    }
    .recommendations ul {
      list-style: none;
      padding: 0;
    }
    .recommendations li {
      padding: 8px 0;
      padding-left: 24px;
      position: relative;
    }
    .recommendations li::before {
      content: "→";
      position: absolute;
      left: 0;
      color: #667eea;
    }
    .tempo-viz {
      display: flex;
      align-items: center;
      margin: 20px 0;
      padding: 20px;
      background: #f8f9fa;
      border-radius: 8px;
    }
    .tempo-phase {
      flex: 1;
      text-align: center;
    }
    .tempo-phase .duration {
      font-size: 24px;
      font-weight: bold;
      color: #667eea;
    }
    .tempo-phase .label {
      font-size: 12px;
      color: #6c757d;
    }
    .tempo-ratio {
      padding: 10px 20px;
      background: #667eea;
      color: white;
      border-radius: 20px;
      font-weight: bold;
    }
    .comparison-table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
    }
    .comparison-table th,
    .comparison-table td {
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid #e9ecef;
    }
    .comparison-table th {
      background: #f8f9fa;
      font-weight: 600;
    }
    .improvement { color: #38a169; }
    .decline { color: #e53e3e; }
    .footer {
      margin-top: 40px;
      padding-top: 20px;
      border-top: 1px solid #e9ecef;
      text-align: center;
      color: #6c757d;
      font-size: 12px;
    }
    @media print {
      body { background: white; padding: 0; }
      .container { box-shadow: none; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Golf Swing Analysis Report</h1>
      <div class="date">
        ${new Date(analysis.analysisTimestamp).toLocaleDateString('en-US', {
          weekday: 'long',
          year: 'numeric',
          month: 'long',
          day: 'numeric',
        })}
      </div>
      <div class="date">Video: ${session.videoFileName}</div>
    </div>

    <div class="score-card">
      <div class="overall-score">
        <span class="value">${Math.round(analysis.scores.overall)}</span>
        <span class="label">OVERALL SCORE</span>
      </div>
    </div>

    <div class="section">
      <h2>Component Scores</h2>
      ${generateScoreBars(analysis.scores)}
    </div>

    <div class="section">
      <h2>Tempo Analysis</h2>
      <div class="tempo-viz">
        <div class="tempo-phase">
          <div class="duration">${(analysis.tempo.backswingDuration / 1000).toFixed(2)}s</div>
          <div class="label">BACKSWING</div>
        </div>
        <div class="tempo-ratio">${analysis.tempo.tempoRatio.toFixed(1)}:1</div>
        <div class="tempo-phase">
          <div class="duration">${(analysis.tempo.downswingDuration / 1000).toFixed(2)}s</div>
          <div class="label">DOWNSWING</div>
        </div>
      </div>
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="value">${(analysis.tempo.totalSwingDuration / 1000).toFixed(2)}s</div>
          <div class="label">Total Duration</div>
        </div>
        <div class="metric-card">
          <div class="value">${analysis.tempo.rhythm}</div>
          <div class="label">Rhythm Quality</div>
        </div>
        <div class="metric-card">
          <div class="value">${(analysis.tempo.transitionPause / 1000).toFixed(3)}s</div>
          <div class="label">Transition Pause</div>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>Balance Metrics</h2>
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="value">${analysis.balance.swayAmount.toFixed(1)} cm</div>
          <div class="label">Backswing Sway</div>
        </div>
        <div class="metric-card">
          <div class="value">${analysis.balance.slideAmount.toFixed(1)} cm</div>
          <div class="label">Downswing Slide</div>
        </div>
        <div class="metric-card">
          <div class="value">${analysis.balance.hipBump.toFixed(1)} cm</div>
          <div class="label">Hip Bump</div>
        </div>
      </div>
    </div>

    ${analysis.keyPositions.top?.angles ? `
    <div class="section">
      <h2>Key Positions</h2>
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="value">${analysis.keyPositions.top.angles.xFactor.toFixed(1)}°</div>
          <div class="label">X-Factor at Top</div>
        </div>
        <div class="metric-card">
          <div class="value">${analysis.keyPositions.top.angles.shoulderRotation.toFixed(1)}°</div>
          <div class="label">Shoulder Turn</div>
        </div>
        <div class="metric-card">
          <div class="value">${analysis.keyPositions.top.angles.hipRotation.toFixed(1)}°</div>
          <div class="label">Hip Turn</div>
        </div>
        <div class="metric-card">
          <div class="value">${analysis.keyPositions.top.angles.spineAngle.toFixed(1)}°</div>
          <div class="label">Spine Angle</div>
        </div>
      </div>
    </div>
    ` : ''}

    ${analysis.posture.headStability !== undefined ? `
    <div class="section">
      <h2>Posture Analysis</h2>
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="value">${analysis.posture.headStability.toFixed(0)}%</div>
          <div class="label">Head Stability</div>
        </div>
        <div class="metric-card">
          <div class="value">${analysis.posture.earlyExtension ? 'Yes' : 'No'}</div>
          <div class="label">Early Extension</div>
        </div>
        <div class="metric-card">
          <div class="value">${analysis.posture.reverseSpineTilt ? 'Yes' : 'No'}</div>
          <div class="label">Reverse Spine</div>
        </div>
      </div>
    </div>
    ` : ''}

    ${analysis.issues.length > 0 ? `
    <div class="section">
      <h2>Identified Issues (${analysis.issues.length})</h2>
      ${analysis.issues.map(issue => `
        <div class="issue ${issue.severity}">
          <div class="title">${issue.name}</div>
          <div class="description">${issue.description}</div>
          ${issue.drillRecommendation ? `<div class="drill">Drill: ${issue.drillRecommendation}</div>` : ''}
        </div>
      `).join('')}
    </div>
    ` : ''}

    ${analysis.recommendations.length > 0 ? `
    <div class="section">
      <h2>Recommendations</h2>
      <div class="recommendations">
        <ul>
          ${analysis.recommendations.map(rec => `<li>${rec}</li>`).join('')}
        </ul>
      </div>
    </div>
    ` : ''}

    ${comparison ? generateComparisonSection(comparison) : ''}

    <div class="footer">
      <p>Generated by Golf Swing Analyzer • Powered by UpstreamDrift</p>
      <p>Session ID: ${session.id}</p>
    </div>
  </div>
</body>
</html>
  `.trim();
}

/**
 * Generate score bars HTML
 */
function generateScoreBars(scores: SwingAnalysis['scores']): string {
  const getBarColor = (score: number): string => {
    if (score >= 80) return '#38a169';
    if (score >= 60) return '#d69e2e';
    return '#e53e3e';
  };

  const scoreItems = [
    { label: 'Tempo', value: scores.tempo },
    { label: 'Balance', value: scores.balance },
    { label: 'Plane', value: scores.plane },
    { label: 'Posture', value: scores.posture },
    { label: 'Rotation', value: scores.rotation },
    { label: 'Timing', value: scores.timing },
    { label: 'Consistency', value: scores.consistency },
  ];

  return scoreItems.map(item => `
    <div class="score-bar">
      <span class="label">${item.label}</span>
      <div class="bar">
        <div class="fill" style="width: ${item.value}%; background: ${getBarColor(item.value)}"></div>
      </div>
      <span class="value">${Math.round(item.value)}</span>
    </div>
  `).join('');
}

/**
 * Generate comparison section HTML
 */
function generateComparisonSection(comparison: SwingComparison): string {
  return `
    <div class="section">
      <h2>Swing Comparison</h2>
      <p style="margin-bottom: 15px;">
        Overall Improvement:
        <strong class="${comparison.overallImprovement > 50 ? 'improvement' : 'decline'}">
          ${comparison.overallImprovement.toFixed(1)}%
        </strong>
      </p>
      <table class="comparison-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Previous</th>
            <th>Current</th>
            <th>Change</th>
          </tr>
        </thead>
        <tbody>
          ${comparison.differences.map(diff => `
            <tr>
              <td>${diff.metric}</td>
              <td>${typeof diff.value1 === 'number' ? diff.value1.toFixed(2) : diff.value1}</td>
              <td>${typeof diff.value2 === 'number' ? diff.value2.toFixed(2) : diff.value2}</td>
              <td class="${diff.improvement ? 'improvement' : diff.delta !== 0 ? 'decline' : ''}">
                ${diff.delta > 0 ? '+' : ''}${diff.delta.toFixed(2)}
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

/**
 * Generate CSV export of metrics
 */
export function generateCSVReport(analysis: SwingAnalysis): string {
  const rows: string[] = [];

  // Header
  rows.push('Metric,Value,Unit,Phase');

  // Scores
  Object.entries(analysis.scores).forEach(([key, value]) => {
    rows.push(`Score - ${key},${value},points,Overall`);
  });

  // Tempo metrics
  rows.push(`Backswing Duration,${analysis.tempo.backswingDuration},ms,Tempo`);
  rows.push(`Downswing Duration,${analysis.tempo.downswingDuration},ms,Tempo`);
  rows.push(`Tempo Ratio,${analysis.tempo.tempoRatio},:1,Tempo`);
  rows.push(`Total Duration,${analysis.tempo.totalSwingDuration},ms,Tempo`);

  // Balance metrics
  rows.push(`Sway Amount,${analysis.balance.swayAmount},cm,Balance`);
  rows.push(`Slide Amount,${analysis.balance.slideAmount},cm,Balance`);
  rows.push(`Hip Bump,${analysis.balance.hipBump},cm,Balance`);

  // Key position angles
  if (analysis.keyPositions.address?.angles) {
    const angles = analysis.keyPositions.address.angles;
    rows.push(`Address - Spine Angle,${angles.spineAngle},degrees,Address`);
    rows.push(`Address - Knee Flexion,${(angles.leftKneeFlexion + angles.rightKneeFlexion) / 2},degrees,Address`);
  }

  if (analysis.keyPositions.top?.angles) {
    const angles = analysis.keyPositions.top.angles;
    rows.push(`Top - Shoulder Rotation,${angles.shoulderRotation},degrees,Top`);
    rows.push(`Top - Hip Rotation,${angles.hipRotation},degrees,Top`);
    rows.push(`Top - X-Factor,${angles.xFactor},degrees,Top`);
    rows.push(`Top - Spine Angle,${angles.spineAngle},degrees,Top`);
  }

  if (analysis.keyPositions.impact?.angles) {
    const angles = analysis.keyPositions.impact.angles;
    rows.push(`Impact - Shoulder Rotation,${angles.shoulderRotation},degrees,Impact`);
    rows.push(`Impact - Hip Rotation,${angles.hipRotation},degrees,Impact`);
    rows.push(`Impact - X-Factor,${angles.xFactor},degrees,Impact`);
  }

  return rows.join('\n');
}

/**
 * Download report as file
 */
export function downloadReport(
  content: string,
  filename: string,
  mimeType: string
): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

/**
 * Generate and download HTML report
 */
export function downloadHTMLReport(
  session: SwingSession,
  analysis: SwingAnalysis,
  comparison?: SwingComparison
): void {
  const html = generateHTMLReport(session, analysis, comparison);
  const filename = `swing-report-${new Date().toISOString().slice(0, 10)}.html`;
  downloadReport(html, filename, 'text/html');
}

/**
 * Generate and download CSV report
 */
export function downloadCSVReport(analysis: SwingAnalysis): void {
  const csv = generateCSVReport(analysis);
  const filename = `swing-metrics-${new Date().toISOString().slice(0, 10)}.csv`;
  downloadReport(csv, filename, 'text/csv');
}

/**
 * Generate and download JSON export
 */
export function downloadJSONExport(
  session: SwingSession,
  analysis: SwingAnalysis
): void {
  const report: SwingReport = {
    generatedAt: Date.now(),
    session,
    analysis,
    charts: [],
    summary: generateSummary(analysis),
  };

  const json = JSON.stringify(report, null, 2);
  const filename = `swing-export-${session.id.slice(0, 8)}.json`;
  downloadReport(json, filename, 'application/json');
}
