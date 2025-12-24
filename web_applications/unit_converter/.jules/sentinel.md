# Sentinel's Security Journal

## 2024-05-22 - Stored XSS via localStorage
**Vulnerability:** Unescaped category labels from localStorage in `app.js` allowed Stored XSS.
**Learning:** Even internal storage mechanisms like `localStorage` should be treated as untrusted input sources, as they can be manipulated (Self-XSS or persistence for other XSS).
**Prevention:** Always escape data before rendering to `innerHTML`, regardless of the source.
