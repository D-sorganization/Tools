---
issue: 5917
summary: "Theme colors now come from themes.json alone: the sidekick design-token API and designTokens.ts read the catalog directly, scripts/sync_design_tokens.py regenerates the design_tokens.json themes table, a CI test fails on drift, colors.py loaders share one catalog path, and the theme README states the do-not-copy consumption contract"
---
