const fs = require('fs');
const content = fs.readFileSync('src/data_processing/data_processor/web/src/hooks/useDataProcessor.ts', 'utf-8');

const search = `  // Differentiation
  const differentiateSignals = useCallback(
    (config: DifferentiationConfig): ProcessingResult => {
      try {
        const { filteredData, timeColumn } = state;
        if (filteredData.length === 0 || !timeColumn) {
          return { success: false, error: 'No data or time column' };
        }

        const windowSize = config.windowSize || 11;
        const result = filteredData.map((row, i) => {
          const newRow = { ...row };
          for (const signal of config.signals) {
            const suffix = config.order === 1 ? 'd' : config.order === 2 ? 'd2' : \`d\${config.order}\`;
            const derivName = \`\${signal}_\${suffix}\`;

            if (i === 0 || i === filteredData.length - 1) {
              newRow[derivName] = 0;
            } else if (config.method === 'spline') {
              // Simple central difference for spline approximation
              const dt = getTimeDelta(filteredData[i + 1][timeColumn], filteredData[i - 1][timeColumn]);
              const dy = (filteredData[i + 1][signal] as number) - (filteredData[i - 1][signal] as number);
              newRow[derivName] = dy / dt;
            } else {
              // Rolling polynomial using moving average approximation
              const halfWindow = Math.floor(windowSize / 2);
              const start = Math.max(0, i - halfWindow);
              const end = Math.min(filteredData.length, i + halfWindow + 1);

              if (end - start >= 3) {
                const dt = getTimeDelta(filteredData[end - 1][timeColumn], filteredData[start][timeColumn]);
                const dy = (filteredData[end - 1][signal] as number) - (filteredData[start][signal] as number);
                newRow[derivName] = dy / dt;
              } else {
                newRow[derivName] = 0;
              }
            }
          }
          return newRow;
        });

        // Update signals list
        const suffix = config.order === 1 ? 'd' : config.order === 2 ? 'd2' : \`d\${config.order}\`;
        const newSignals = [
          ...state.signals,
          ...config.signals.map((s) => \`\${s}_\${suffix}\`),
        ];`;

const replace = `  // Differentiation
  const differentiateSignals = useCallback(
    (config: DifferentiationConfig): ProcessingResult => {
      try {
        const { filteredData, timeColumn } = state;
        const len = filteredData.length;
        if (len === 0 || !timeColumn) {
          return { success: false, error: 'No data or time column' };
        }

        const windowSize = config.windowSize || 11;
        const suffix = config.order === 1 ? 'd' : config.order === 2 ? 'd2' : \`d\${config.order}\`;
        const derivNames = config.signals.map(signal => \`\${signal}_\${suffix}\`);

        // ⚡ Bolt Optimization: Replace filteredData.map() with a single-pass loop and hoist time delta calculations.
        // Evaluating timestamps (getTimeDelta) inside the inner signal loop caused O(N * M) overhead.
        // Hoisting dt out of the inner loop drastically reduces garbage collection pauses.
        const result = new Array(len);

        for (let i = 0; i < len; i++) {
          const newRow = { ...filteredData[i] };

          if (i === 0 || i === len - 1) {
            for (let j = 0; j < config.signals.length; j++) {
              newRow[derivNames[j]] = 0;
            }
          } else if (config.method === 'spline') {
            // Simple central difference for spline approximation
            const dt = getTimeDelta(filteredData[i + 1][timeColumn], filteredData[i - 1][timeColumn]);

            for (let j = 0; j < config.signals.length; j++) {
              const signal = config.signals[j];
              const dy = (filteredData[i + 1][signal] as number) - (filteredData[i - 1][signal] as number);
              newRow[derivNames[j]] = dt !== 0 ? dy / dt : 0;
            }
          } else {
            // Rolling polynomial using moving average approximation
            const halfWindow = Math.floor(windowSize / 2);
            const start = Math.max(0, i - halfWindow);
            const end = Math.min(len, i + halfWindow + 1);

            if (end - start >= 3) {
              const dt = getTimeDelta(filteredData[end - 1][timeColumn], filteredData[start][timeColumn]);

              for (let j = 0; j < config.signals.length; j++) {
                const signal = config.signals[j];
                const dy = (filteredData[end - 1][signal] as number) - (filteredData[start][signal] as number);
                newRow[derivNames[j]] = dt !== 0 ? dy / dt : 0;
              }
            } else {
              for (let j = 0; j < config.signals.length; j++) {
                newRow[derivNames[j]] = 0;
              }
            }
          }
          result[i] = newRow;
        }

        // Update signals list
        const newSignals = [
          ...state.signals,
          ...derivNames,
        ];`;

if (content.includes(search)) {
    fs.writeFileSync('src/data_processing/data_processor/web/src/hooks/useDataProcessor.ts', content.replace(search, replace));
    console.log("Successfully patched");
} else {
    console.log("Search string not found!");
}
