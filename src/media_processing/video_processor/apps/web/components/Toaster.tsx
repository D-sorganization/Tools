/**
 * Toast Notifications Component
 *
 * Provides user-facing toast notifications for success, error, and info messages.
 * Uses Sonner library for a great user experience.
 *
 * @see https://sonner.emilkowal.ski/
 */

'use client';

import { Toaster as SonnerToaster } from 'sonner';

export function Toaster() {
  return (
    <SonnerToaster
      position="top-right"
      expand={false}
      richColors
      closeButton
      duration={5000}
      toastOptions={{
        style: {
          background: 'white',
          color: '#1f2937', // gray-800
          border: '1px solid #e5e7eb', // gray-200
        },
        className: 'shadow-lg',
      }}
    />
  );
}
