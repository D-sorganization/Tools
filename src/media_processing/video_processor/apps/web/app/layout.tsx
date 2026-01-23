import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { Toaster } from '@/components/Toaster';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Golf Swing Video Analyzer',
  description: 'Advanced AI-powered golf swing video analysis platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ErrorBoundary>
          {children}
        </ErrorBoundary>
        <Toaster />
      </body>
    </html>
  );
}
