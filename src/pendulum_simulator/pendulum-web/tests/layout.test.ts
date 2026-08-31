import { describe, expect, it } from 'vitest';

import { readFileSync } from 'node:fs';

const appCss = readFileSync(new URL('../src/App.css', import.meta.url), 'utf8');

describe('scrollable comparison workspace', () => {
    it('allows the document to extend below the simulator viewport', () => {
        expect(appCss).toMatch(/html,\s*body,\s*#root\s*{[^}]*min-height:\s*100%/s);
        expect(appCss).toMatch(/body\s*{[^}]*overflow-y:\s*auto/s);
        expect(appCss).toMatch(/\.app\s*{[^}]*min-height:\s*100vh/s);
        expect(appCss).not.toMatch(/\.app\s*{[^}]*height:\s*100vh[^}]*overflow:\s*hidden/s);
    });
});

describe('force-source animation card registration', () => {
    it('reserves one invariant title row before every animation stage', () => {
        expect(appCss).toMatch(/\.force-source-animation-grid article\s*{[^}]*display:\s*grid[^}]*grid-template-rows:\s*3lh auto 1fr/s);
        expect(appCss).toMatch(/\.force-source-animation-grid h3\s*{[^}]*height:\s*3lh/s);
    });

    it('uses one undistorted 192 by 176 stage for every scenario', () => {
        expect(appCss).toMatch(/\.force-source-animation-stage\s*{[^}]*aspect-ratio:\s*192\s*\/\s*176/s);
        expect(appCss).toMatch(/\.force-source-animation-stage svg\s*{[^}]*width:\s*100%[^}]*height:\s*100%/s);
    });
});
