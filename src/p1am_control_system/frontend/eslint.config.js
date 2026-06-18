import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

/**
 * Flat ESLint config for the P1AM control-system frontend.
 *
 * Uses typescript-eslint for type-aware-friendly linting plus the
 * react-hooks plugin so the rules-of-hooks and exhaustive-deps checks
 * run in CI (previously `npm run lint` failed because eslint was neither
 * installed nor configured).
 */
export default tseslint.config(
  {
    ignores: ["dist", "coverage", "node_modules"],
  },
  {
    files: ["**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2020,
      globals: {
        ...globals.browser,
      },
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // rules-of-hooks stays an error (correctness); exhaustive-deps is advisory
      // because several effects here intentionally run once on mount.
      "react-hooks/exhaustive-deps": "warn",
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          // Unused `catch (err)` bindings are idiomatic here; don't fail on them.
          caughtErrors: "none",
        },
      ],
      // `any` survives in a few legacy payload-handling spots; tracked for
      // removal by the runtime-validation work (#3545). Warn rather than block
      // so the lint gate is usable today.
      "@typescript-eslint/no-explicit-any": "warn",
    },
  },
  {
    // Test files run under Vitest globals (describe/it/expect).
    files: ["**/*.{test,spec}.{ts,tsx}", "src/test/**"],
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
  },
);
