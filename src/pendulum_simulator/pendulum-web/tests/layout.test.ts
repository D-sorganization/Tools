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
