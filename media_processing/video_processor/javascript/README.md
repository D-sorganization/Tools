# JavaScript/Node.js Code

This folder contains JavaScript and Node.js code for the Golf Swing Video Analysis Platform.

## Structure

```
javascript/
├── src/          # Source code modules
├── tests/        # Unit and integration tests
└── utils/        # Utility functions and helpers
```

## Purpose

Similar to the `python/` and `matlab/` folders, this directory contains reusable JavaScript/Node.js modules that can be used across the project, including:
- Utility functions for video processing
- Shared business logic
- Node.js server-side utilities
- Reusable algorithms and calculations

## Usage

The main application code lives in `apps/web/` (Next.js app), but shared JavaScript/Node.js modules go here.

## Testing

Run tests with:
```bash
npm test
```

## Code Standards

Follow the rules in `.cursor/rules/webdevrules.md` and `.cursor/rules/.cursorrules.md`:
- TypeScript strict mode
- No placeholders (no TODO, FIXME, etc.)
- Complete error handling
- Type all functions
- Include tests for all functions
