# Unit Converter - Developer Guidelines

## ⚠️ Governance & Workflow

**IMPORTANT:** The authoritative governance, CI/CD workflows, and general coding standards are defined in the root [AGENTS.md](../../AGENTS.md).

This document provides specific technical context for the **Unit Converter** project.

---

## 🛠️ Tech Stack & Constraints

1.  **Vanilla JavaScript**:
    - No build step (webpack/vite/etc.) is used.
    - Code must run directly in modern browsers.
    - No external runtime dependencies (keep it lightweight).

2.  **CSS**:
    - Use standard CSS variables for theming.
    - Follow BEM naming convention where possible.

3.  **Testing**:
    - Tests are run via `jest` (see `package.json`).
    - `converter.js` must remain testable in Node.js environment (handle `localStorage` mocking if needed).

## 📂 Project Structure

- `app.js`: Main UI logic and event handling.
- `converter.js`: Core conversion logic (business logic).
- `style.css`: Application styling.
- `index.html`: Main entry point.

## 🚀 Deployment

This is a static web application. It is deployed via the standard CI/CD pipeline defined in `.github/workflows/ci-standard.yml`.
