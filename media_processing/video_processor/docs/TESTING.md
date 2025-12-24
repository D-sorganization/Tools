# Testing Guide

This document provides comprehensive information about testing the Golf Swing Video Analyzer platform.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Test Coverage](#test-coverage)
- [Best Practices](#best-practices)
- [CI/CD Integration](#cicd-integration)

---

## Quick Start

### Install Dependencies

```bash
cd apps/web
npm install
```

### Run All Tests

```bash
# Run unit tests
npm test

# Run E2E tests
npm run test:e2e

# Run tests with coverage
npm run test:coverage
```

---

## Test Structure

```
Video_Processor/
├── apps/web/
│   ├── __tests__/                    # Component tests
│   ├── hooks/__tests__/               # Hook tests
│   ├── lib/__tests__/                 # Utility tests
│   ├── components/**/__tests__/       # Component-specific tests
│   └── test/
│       ├── setup.ts                   # Test configuration
│       └── utils.tsx                  # Test utilities
├── tests/
│   ├── e2e/                           # End-to-end tests
│   └── fixtures/                      # Test data files
├── vitest.config.ts                   # Vitest configuration
└── playwright.config.ts               # Playwright configuration
```

---

## Running Tests

### Unit Tests (Vitest)

```bash
# Run all unit tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage

# Run specific test file
npm test hooks/__tests__/useVideoFrame.test.ts

# Run tests matching pattern
npm test -- --grep "error"
```

### E2E Tests (Playwright)

```bash
# Run all E2E tests
npm run test:e2e

# Run E2E tests with UI
npm run test:e2e:ui

# Run specific browser
npm run test:e2e -- --project=chromium

# Run in headed mode (see browser)
npm run test:e2e -- --headed

# Debug mode
npm run test:e2e -- --debug
```

### Coverage Reports

```bash
# Generate coverage report
npm run test:coverage

# Open coverage report in browser
open coverage/index.html
```

**Coverage Thresholds:**
- Lines: 80%
- Functions: 80%
- Branches: 70%
- Statements: 80%

---

## Writing Tests

### Unit Tests

#### Example: Testing a Hook

```typescript
// hooks/__tests__/useVideoFrame.test.ts
import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useVideoFrame } from '../useVideoFrame';
import { createMockVideoElement } from '@/test/utils';

describe('useVideoFrame', () => {
  it('should calculate current frame correctly', () => {
    const mockVideo = createMockVideoElement({
      currentTime: 1.0,
      duration: 10.0,
    }) as HTMLVideoElement;

    const { result } = renderHook(() =>
      useVideoFrame({ videoElement: mockVideo, fps: 30 })
    );

    expect(result.current.getCurrentFrame()).toBe(30);
  });
});
```

#### Example: Testing a Component

```typescript
// components/video/__tests__/VideoPlayer.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import VideoPlayer from '../VideoPlayer';

describe('VideoPlayer', () => {
  it('should render video element', () => {
    render(<VideoPlayer videoUrl="test.mp4" />);
    const video = screen.getByRole('video') as HTMLVideoElement;
    expect(video).toBeInTheDocument();
  });
});
```

#### Example: Testing Error Handling

```typescript
// lib/__tests__/errors.test.ts
import { describe, it, expect } from 'vitest';
import { ValidationError } from '../errors';

describe('ValidationError', () => {
  it('should create error with 400 status', () => {
    const error = new ValidationError('Invalid input');

    expect(error.statusCode).toBe(400);
    expect(error.message).toBe('Invalid input');
  });
});
```

### E2E Tests

#### Example: User Flow Test

```typescript
// tests/e2e/video-upload.spec.ts
import { test, expect } from '@playwright/test';

test('should upload and play video', async ({ page }) => {
  await page.goto('/');

  // Upload video
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles('tests/fixtures/test-video.mp4');

  // Wait for video to load
  await expect(page.locator('video')).toBeVisible();

  // Play video
  await page.getByRole('button', { name: /play/i }).click();

  // Verify video is playing
  await expect(page.locator('video')).toHaveJSProperty('paused', false);
});
```

---

## Test Utilities

### Available Test Utilities

Located in `apps/web/test/utils.tsx`:

#### `createMockFile(name, size, type)`
Create a mock File object for testing uploads.

```typescript
const file = createMockFile('test.mp4', 1024 * 1024, 'video/mp4');
```

#### `createMockVideoElement(overrides)`
Create a mock HTMLVideoElement.

```typescript
const video = createMockVideoElement({
  currentTime: 1.0,
  duration: 10.0,
});
```

#### `createMockBlob(content, type)`
Create a mock Blob.

```typescript
const blob = createMockBlob('content', 'video/mp4');
```

#### `createMockFabricCanvas()`
Create a mock Fabric.js canvas.

```typescript
const canvas = createMockFabricCanvas();
```

#### `waitForCondition(condition, timeout)`
Wait for a condition to be true.

```typescript
await waitForCondition(() => result.current.isLoaded, 5000);
```

---

## Test Coverage

### Current Coverage

Run `npm run test:coverage` to see current coverage.

### Coverage Goals

| Metric | Target | Current |
|--------|--------|---------|
| Lines | 80% | TBD |
| Functions | 80% | TBD |
| Branches | 70% | TBD |
| Statements | 80% | TBD |

### Improving Coverage

1. **Identify gaps:**
   ```bash
   npm run test:coverage
   open coverage/index.html
   ```

2. **Focus on critical paths:**
   - Video upload and validation
   - Error handling
   - User interactions
   - Data transformations

3. **Add tests incrementally:**
   - Start with untested files
   - Add edge cases
   - Test error conditions

---

## Best Practices

### 1. Test Structure (AAA Pattern)

```typescript
it('should do something', () => {
  // Arrange
  const input = 'test';
  const expected = 'TEST';

  // Act
  const result = transform(input);

  // Assert
  expect(result).toBe(expected);
});
```

### 2. Test Names

✅ **Good:**
```typescript
it('should return 0 when video element is null')
it('should throw ValidationError for oversized files')
it('should display error message on upload failure')
```

❌ **Bad:**
```typescript
it('test 1')
it('should work')
it('returns something')
```

### 3. One Assertion Per Test (Generally)

✅ **Good:**
```typescript
it('should have correct status code', () => {
  expect(error.statusCode).toBe(400);
});

it('should have correct message', () => {
  expect(error.message).toBe('Invalid input');
});
```

❌ **Avoid:**
```typescript
it('should have correct properties', () => {
  expect(error.statusCode).toBe(400);
  expect(error.message).toBe('Invalid input');
  expect(error.code).toBe('VALIDATION_ERROR');
  // Too many unrelated assertions
});
```

### 4. Mock External Dependencies

```typescript
// Mock Next.js router
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
  }),
}));

// Mock file operations
vi.mock('fs', () => ({
  readFile: vi.fn(),
}));
```

### 5. Clean Up After Tests

```typescript
afterEach(() => {
  cleanup(); // React Testing Library cleanup
  vi.clearAllMocks(); // Clear mock calls
});
```

### 6. Test Edge Cases

Always test:
- Empty inputs
- Null/undefined values
- Boundary values (0, max, max+1)
- Error conditions
- Async failures

### 7. Don't Test Implementation Details

✅ **Good:** Test behavior
```typescript
it('should display error message to user', () => {
  const { getByText } = render(<Component />);
  expect(getByText('Error message')).toBeInTheDocument();
});
```

❌ **Bad:** Test implementation
```typescript
it('should call setState with error', () => {
  // Testing internal implementation
});
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Run unit tests
        run: npm test -- --coverage

      - name: Run E2E tests
        run: npm run test:e2e

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
```

### Pre-commit Hook

Add to `.husky/pre-commit`:

```bash
#!/bin/sh
npm test -- --run
```

---

## Debugging Tests

### Debug Unit Tests

```bash
# Run specific test in debug mode
node --inspect-brk node_modules/.bin/vitest run path/to/test.ts

# Or use VS Code debugger with this launch.json:
{
  "type": "node",
  "request": "launch",
  "name": "Debug Vitest Tests",
  "runtimeExecutable": "npm",
  "runtimeArgs": ["test"],
  "console": "integratedTerminal",
  "internalConsoleOptions": "neverOpen"
}
```

### Debug E2E Tests

```bash
# Run with headed browser
npm run test:e2e -- --headed

# Debug mode (pause before each action)
npm run test:e2e -- --debug

# Use Playwright Inspector
PWDEBUG=1 npm run test:e2e
```

---

## Common Issues

### Issue: Tests timeout

**Solution:** Increase timeout for async operations
```typescript
it('should load video', async () => {
  await expect(page.locator('video')).toBeVisible({ timeout: 10000 });
}, { timeout: 15000 });
```

### Issue: Mock not working

**Solution:** Ensure mock is hoisted
```typescript
vi.mock('module', () => ({ ... })); // Must be at top level
```

### Issue: Canvas errors in tests

**Solution:** Mock canvas (already done in test/setup.ts)

### Issue: Video element not loading

**Solution:** Use mock video element or real test fixtures

---

## Resources

- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/react)
- [Playwright Documentation](https://playwright.dev/)
- [Testing Best Practices](https://github.com/goldbergyoni/javascript-testing-best-practices)

---

## Contributing

When adding new features:

1. ✅ Write tests FIRST (TDD approach)
2. ✅ Ensure tests pass: `npm test`
3. ✅ Check coverage: `npm run test:coverage`
4. ✅ Add E2E tests for user flows
5. ✅ Update this documentation if needed

**Minimum requirements before PR:**
- All tests pass
- Coverage thresholds met (80% lines, 80% functions)
- No failing E2E tests
- Test added for new functionality

---

*Last updated: 2025-11-16*
*Testing framework: Vitest + Playwright*
*Coverage target: 80%*
