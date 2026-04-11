/**
 * Golf Swing Analysis Persistence Layer
 * Handles storage and retrieval of swing analysis data
 * Uses IndexedDB for client-side storage with JSON export capability
 */

import {
  SwingAnalysis,
  SwingSession,
  SwingComparison,
  SwingReport,
  PoseFrame,
} from './types';

const DB_NAME = 'GolfSwingAnalyzer';
const DB_VERSION = 1;

// Store names
const STORES = {
  SESSIONS: 'sessions',
  ANALYSES: 'analyses',
  POSE_DATA: 'poseData',
  USER_SETTINGS: 'userSettings',
} as const;

/**
 * Initialize IndexedDB
 */
export async function initDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => {
      reject(new Error('Failed to open IndexedDB'));
    };

    request.onsuccess = () => {
      resolve(request.result);
    };

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      // Sessions store
      if (!db.objectStoreNames.contains(STORES.SESSIONS)) {
        const sessionsStore = db.createObjectStore(STORES.SESSIONS, { keyPath: 'id' });
        sessionsStore.createIndex('timestamp', 'timestamp');
        sessionsStore.createIndex('userId', 'userId');
        sessionsStore.createIndex('videoFileName', 'videoFileName');
      }

      // Analyses store
      if (!db.objectStoreNames.contains(STORES.ANALYSES)) {
        const analysesStore = db.createObjectStore(STORES.ANALYSES, { keyPath: 'sessionId' });
        analysesStore.createIndex('analysisTimestamp', 'analysisTimestamp');
        analysesStore.createIndex('scores.overall', 'scores.overall');
      }

      // Pose data store (for large pose frame arrays)
      if (!db.objectStoreNames.contains(STORES.POSE_DATA)) {
        db.createObjectStore(STORES.POSE_DATA, { keyPath: 'sessionId' });
      }

      // User settings store
      if (!db.objectStoreNames.contains(STORES.USER_SETTINGS)) {
        db.createObjectStore(STORES.USER_SETTINGS, { keyPath: 'key' });
      }
    };
  });
}

/**
 * Save a swing session with its analysis
 */
export async function saveSession(
  session: SwingSession,
  analysis: SwingAnalysis
): Promise<void> {
  const db = await initDatabase();

  // Store pose frames separately to avoid bloating the analysis record
  const poseFrames = analysis.poseFrames;
  const analysisWithoutFrames = {
    ...analysis,
    poseFrames: [], // Store reference only
  };

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(
      [STORES.SESSIONS, STORES.ANALYSES, STORES.POSE_DATA],
      'readwrite'
    );

    transaction.onerror = () => reject(transaction.error);
    transaction.oncomplete = () => resolve();

    transaction.objectStore(STORES.SESSIONS).put(session);
    transaction.objectStore(STORES.ANALYSES).put(analysisWithoutFrames);
    transaction.objectStore(STORES.POSE_DATA).put({
      sessionId: session.id,
      frames: poseFrames,
    });
  });
}

/**
 * Get a session by ID
 */
export async function getSession(sessionId: string): Promise<SwingSession | undefined> {
  const db = await initDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORES.SESSIONS, 'readonly');
    const request = transaction.objectStore(STORES.SESSIONS).get(sessionId);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Get analysis with pose frames
 */
export async function getAnalysisWithFrames(
  sessionId: string
): Promise<SwingAnalysis | undefined> {
  const db = await initDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction([STORES.ANALYSES, STORES.POSE_DATA], 'readonly');

    const analysisRequest = transaction.objectStore(STORES.ANALYSES).get(sessionId);
    const poseRequest = transaction.objectStore(STORES.POSE_DATA).get(sessionId);

    let analysis: SwingAnalysis | undefined;
    let poseData: { sessionId: string; frames: PoseFrame[] } | undefined;

    analysisRequest.onsuccess = () => {
      analysis = analysisRequest.result;
      if (analysis && poseData) {
        analysis.poseFrames = poseData.frames;
        resolve(analysis);
      }
    };

    poseRequest.onsuccess = () => {
      poseData = poseRequest.result;
      if (analysis && poseData) {
        analysis.poseFrames = poseData.frames;
        resolve(analysis);
      }
    };

    transaction.oncomplete = () => {
      if (!analysis) resolve(undefined);
    };

    transaction.onerror = () => reject(transaction.error);
  });
}

/**
 * Get all sessions, sorted by timestamp
 */
export async function getAllSessions(
  limit: number = 50,
  userId?: string
): Promise<SwingSession[]> {
  const db = await initDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORES.SESSIONS, 'readonly');
    const store = transaction.objectStore(STORES.SESSIONS);
    const index = store.index('timestamp');

    const sessions: SwingSession[] = [];
    const request = index.openCursor(null, 'prev'); // Descending order

    request.onsuccess = () => {
      const cursor = request.result;
      if (cursor && sessions.length < limit) {
        const session = cursor.value as SwingSession;
        if (!userId || session.userId === userId) {
          sessions.push(session);
        }
        cursor.continue();
      } else {
        resolve(sessions);
      }
    };

    request.onerror = () => reject(request.error);
  });
}

/**
 * Delete a session and its related data
 */
export async function deleteSession(sessionId: string): Promise<void> {
  const db = await initDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(
      [STORES.SESSIONS, STORES.ANALYSES, STORES.POSE_DATA],
      'readwrite'
    );

    transaction.onerror = () => reject(transaction.error);
    transaction.oncomplete = () => resolve();

    transaction.objectStore(STORES.SESSIONS).delete(sessionId);
    transaction.objectStore(STORES.ANALYSES).delete(sessionId);
    transaction.objectStore(STORES.POSE_DATA).delete(sessionId);
  });
}

/**
 * Compare two swings and generate a comparison report
 */
export async function compareSwings(
  sessionId1: string,
  sessionId2: string
): Promise<SwingComparison | undefined> {
  const [analysis1, analysis2] = await Promise.all([
    getAnalysisWithFrames(sessionId1),
    getAnalysisWithFrames(sessionId2),
  ]);

  if (!analysis1 || !analysis2) {
    return undefined;
  }

  const differences: SwingComparison['differences'] = [];

  // Compare tempo
  differences.push({
    metric: 'Tempo Ratio',
    value1: analysis1.tempo.tempoRatio,
    value2: analysis2.tempo.tempoRatio,
    delta: analysis2.tempo.tempoRatio - analysis1.tempo.tempoRatio,
    improvement: Math.abs(analysis2.tempo.tempoRatio - 3) < Math.abs(analysis1.tempo.tempoRatio - 3),
  });

  // Compare scores
  const scoreMetrics: (keyof typeof analysis1.scores)[] = [
    'overall',
    'tempo',
    'balance',
    'plane',
    'posture',
    'rotation',
    'timing',
  ];

  for (const metric of scoreMetrics) {
    differences.push({
      metric: `${metric.charAt(0).toUpperCase() + metric.slice(1)} Score`,
      value1: analysis1.scores[metric],
      value2: analysis2.scores[metric],
      delta: analysis2.scores[metric] - analysis1.scores[metric],
      improvement: analysis2.scores[metric] > analysis1.scores[metric],
    });
  }

  // Compare balance metrics
  differences.push({
    metric: 'Sway Amount (cm)',
    value1: analysis1.balance.swayAmount,
    value2: analysis2.balance.swayAmount,
    delta: analysis2.balance.swayAmount - analysis1.balance.swayAmount,
    improvement: analysis2.balance.swayAmount < analysis1.balance.swayAmount,
  });

  // Compare key angles if available
  if (analysis1.keyPositions.top?.angles && analysis2.keyPositions.top?.angles) {
    differences.push({
      metric: 'X-Factor at Top',
      value1: analysis1.keyPositions.top.angles.xFactor,
      value2: analysis2.keyPositions.top.angles.xFactor,
      delta: analysis2.keyPositions.top.angles.xFactor - analysis1.keyPositions.top.angles.xFactor,
      improvement: analysis2.keyPositions.top.angles.xFactor > analysis1.keyPositions.top.angles.xFactor,
    });
  }

  // Calculate overall improvement
  const improvements = differences.filter((d) => d.improvement).length;
  const overallImprovement = (improvements / differences.length) * 100;

  return {
    swing1: analysis1,
    swing2: analysis2,
    differences,
    overallImprovement,
  };
}

/**
 * Export session data to JSON
 */
export async function exportSessionToJSON(sessionId: string): Promise<string> {
  const [session, analysis] = await Promise.all([
    getSession(sessionId),
    getAnalysisWithFrames(sessionId),
  ]);

  if (!session || !analysis) {
    throw new Error('Session not found');
  }

  const report: SwingReport = {
    generatedAt: Date.now(),
    session,
    analysis,
    charts: [],
    summary: generateSummary(analysis),
  };

  return JSON.stringify(report, null, 2);
}

/**
 * Import session data from JSON
 */
export async function importSessionFromJSON(jsonData: string): Promise<string> {
  const report = JSON.parse(jsonData) as SwingReport;

  if (!report.session || !report.analysis) {
    throw new Error('Invalid import data');
  }

  // Generate new IDs to avoid conflicts
  const newSessionId = crypto.randomUUID();
  const session = {
    ...report.session,
    id: newSessionId,
    timestamp: Date.now(),
  };
  const analysis = {
    ...report.analysis,
    sessionId: newSessionId,
  };

  await saveSession(session, analysis);

  return newSessionId;
}

/**
 * Generate a text summary of the analysis
 */
export function generateSummary(analysis: SwingAnalysis): string {
  const lines: string[] = [];

  lines.push(`Golf Swing Analysis Report`);
  lines.push(`Generated: ${new Date(analysis.analysisTimestamp).toLocaleString()}`);
  lines.push('');

  lines.push(`Overall Score: ${analysis.scores.overall.toFixed(0)}/100`);
  lines.push('');

  lines.push('Component Scores:');
  lines.push(`  Tempo: ${analysis.scores.tempo.toFixed(0)}`);
  lines.push(`  Balance: ${analysis.scores.balance.toFixed(0)}`);
  lines.push(`  Plane: ${analysis.scores.plane.toFixed(0)}`);
  lines.push(`  Posture: ${analysis.scores.posture.toFixed(0)}`);
  lines.push(`  Rotation: ${analysis.scores.rotation.toFixed(0)}`);
  lines.push(`  Timing: ${analysis.scores.timing.toFixed(0)}`);
  lines.push('');

  lines.push('Tempo Analysis:');
  lines.push(`  Backswing: ${analysis.tempo.backswingDuration.toFixed(0)}ms`);
  lines.push(`  Downswing: ${analysis.tempo.downswingDuration.toFixed(0)}ms`);
  lines.push(`  Ratio: ${analysis.tempo.tempoRatio.toFixed(2)}:1`);
  lines.push(`  Rhythm: ${analysis.tempo.rhythm}`);
  lines.push('');

  if (analysis.issues.length > 0) {
    lines.push('Identified Issues:');
    for (const issue of analysis.issues) {
      lines.push(`  [${issue.severity.toUpperCase()}] ${issue.name}: ${issue.description}`);
    }
    lines.push('');
  }

  if (analysis.recommendations.length > 0) {
    lines.push('Recommendations:');
    for (const rec of analysis.recommendations) {
      lines.push(`  - ${rec}`);
    }
  }

  return lines.join('\n');
}

/**
 * Get user settings
 */
export async function getUserSetting<T>(key: string, defaultValue: T): Promise<T> {
  const db = await initDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORES.USER_SETTINGS, 'readonly');
    const request = transaction.objectStore(STORES.USER_SETTINGS).get(key);

    request.onsuccess = () => {
      resolve(request.result?.value ?? defaultValue);
    };
    request.onerror = () => reject(request.error);
  });
}

/**
 * Set user setting
 */
export async function setUserSetting<T>(key: string, value: T): Promise<void> {
  const db = await initDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORES.USER_SETTINGS, 'readwrite');
    const request = transaction.objectStore(STORES.USER_SETTINGS).put({ key, value });

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

/**
 * Clear all data (for testing or reset)
 */
export async function clearAllData(): Promise<void> {
  const db = await initDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(
      [STORES.SESSIONS, STORES.ANALYSES, STORES.POSE_DATA, STORES.USER_SETTINGS],
      'readwrite'
    );

    transaction.onerror = () => reject(transaction.error);
    transaction.oncomplete = () => resolve();

    transaction.objectStore(STORES.SESSIONS).clear();
    transaction.objectStore(STORES.ANALYSES).clear();
    transaction.objectStore(STORES.POSE_DATA).clear();
    transaction.objectStore(STORES.USER_SETTINGS).clear();
  });
}

/**
 * Get storage usage statistics
 */
export async function getStorageStats(): Promise<{
  sessionCount: number;
  totalSize: number;
  oldestSession: number | null;
  newestSession: number | null;
}> {
  const db = await initDatabase();

  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORES.SESSIONS, 'readonly');
    const store = transaction.objectStore(STORES.SESSIONS);
    const countRequest = store.count();
    const cursorRequest = store.index('timestamp').openCursor();

    let sessionCount = 0;
    let oldestSession: number | null = null;
    let newestSession: number | null = null;

    countRequest.onsuccess = () => {
      sessionCount = countRequest.result;
    };

    cursorRequest.onsuccess = () => {
      const cursor = cursorRequest.result;
      if (cursor) {
        const timestamp = (cursor.value as SwingSession).timestamp;
        if (oldestSession === null || timestamp < oldestSession) {
          oldestSession = timestamp;
        }
        if (newestSession === null || timestamp > newestSession) {
          newestSession = timestamp;
        }
        cursor.continue();
      }
    };

    transaction.oncomplete = () => {
      // Estimate storage size (IndexedDB doesn't provide exact size easily)
      const estimatedSize = sessionCount * 50000; // ~50KB per session estimate

      resolve({
        sessionCount,
        totalSize: estimatedSize,
        oldestSession,
        newestSession,
      });
    };

    transaction.onerror = () => reject(transaction.error);
  });
}
