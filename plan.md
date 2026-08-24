1. Install `dompurify` in `rate_of_closure/web` as a direct dependency for sanitization (already done).
2. Install `@types/dompurify` in `rate_of_closure/web` as a dev dependency. (already done).
3. Import `DOMPurify` into `rate_of_closure/web/src/components/Derivation.tsx`.
4. Wrap the `katex.renderToString(tex, ...)` string output with `DOMPurify.sanitize(...)` to sanitize KaTeX output before passing it to `dangerouslySetInnerHTML`. We can use DOMPurify in default mode which handles MathML / KaTeX output safely, or customize tags if strictly needed. KaTeX generates fairly standard HTML/MathML spans. Let's start with default sanitization.
5. Create `.jules/sentinel.md` entry documenting the finding.
6. Verify changes with `pnpm run type-check`, `pnpm lint`, and `pnpm test`.
7. Pre-commit check
8. Submit PR.
