/**
 * Test Utilities
 *
 * Reusable utilities for testing components.
 */

import { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { vi, beforeEach, afterEach } from 'vitest';

/**
 * Custom render function that wraps components with providers
 */
export function renderWithProviders(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) {
  // Add providers here when needed (e.g., Context providers, Router, etc.)
  // For now, just use the default render
  return render(ui, options);
}

/**
 * Create a mock File object for testing
 */
export function createMockFile(
  name: string = 'test-video.mp4',
  size: number = 1024 * 1024, // 1MB
  type: string = 'video/mp4'
): File {
  const blob = new Blob(['mock video content'], { type });
  const file = new File([blob], name, { type });

  // Override the size property
  Object.defineProperty(file, 'size', {
    value: size,
    configurable: true,
  });

  return file;
}

/**
 * Create a mock HTMLVideoElement for testing
 */
export function createMockVideoElement(
  overrides?: Partial<HTMLVideoElement>
): Partial<HTMLVideoElement> {
  return {
    currentTime: 0,
    duration: 10,
    paused: true,
    play: vi.fn(() => Promise.resolve()),
    pause: vi.fn(),
    load: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    videoWidth: 1920,
    videoHeight: 1080,
    ...overrides,
  };
}

/**
 * Create a mock Blob for testing
 */
export function createMockBlob(
  content: string = 'mock content',
  type: string = 'application/octet-stream'
): Blob {
  return new Blob([content], { type });
}

/**
 * Wait for a condition to be true
 */
export async function waitForCondition(
  condition: () => boolean,
  timeout: number = 5000
): Promise<void> {
  const startTime = Date.now();

  while (!condition()) {
    if (Date.now() - startTime > timeout) {
      throw new Error('Timeout waiting for condition');
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

/**
 * Mock localStorage
 */
export function mockLocalStorage() {
  const storage: Record<string, string> = {};

  return {
    getItem: (key: string) => storage[key] || null,
    setItem: (key: string, value: string) => {
      storage[key] = value;
    },
    removeItem: (key: string) => {
      delete storage[key];
    },
    clear: () => {
      Object.keys(storage).forEach((key) => delete storage[key]);
    },
    get length() {
      return Object.keys(storage).length;
    },
    key: (index: number) => Object.keys(storage)[index] || null,
  };
}

/**
 * Create a mock fabric.Canvas object
 */
export function createMockFabricCanvas(): any {
  return {
    setDimensions: vi.fn(),
    renderAll: vi.fn(),
    dispose: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
    getObjects: vi.fn(() => []),
    clear: vi.fn(),
    remove: vi.fn(),
    add: vi.fn(),
    getActiveObjects: vi.fn(() => []),
    discardActiveObject: vi.fn(),
    setActiveObject: vi.fn(),
    getElement: vi.fn(() => ({
      parentElement: {
        clientWidth: 800,
        clientHeight: 600,
      },
    })),
    freeDrawingBrush: {
      width: 2,
      color: '#000000',
    },
    isDrawingMode: false,
    selection: true,
    defaultCursor: 'default',
  };
}

/**
 * Suppress console methods during tests
 */
export function suppressConsole(methods: ('log' | 'warn' | 'error')[] = ['error']) {
  const original: Record<string, any> = {};

  beforeEach(() => {
    methods.forEach((method) => {
      original[method] = console[method];
      console[method] = vi.fn();
    });
  });

  afterEach(() => {
    methods.forEach((method) => {
      console[method] = original[method];
    });
  });
}

// Re-export commonly used testing library utilities
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';
