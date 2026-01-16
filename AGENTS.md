# AGENTS.md

## 🤖 Agent Personas & Directives

**Audience:** This document is the authoritative guide for AI agents working in this repository.

**Core Mission:**

- Write high-quality, maintainable, and secure code.
- Adhere strictly to the project's architectural and stylistic standards.
- Act as a responsible pair programmer, always verifying assumptions and testing changes.

---

## 🛡️ Safety & Security (CRITICAL)

1. **Secrets Management**:
   - **NEVER** commit API keys, passwords, tokens, or database connection strings.
   - Use `.env` files and `python-dotenv` for secrets.
   - Create `.env.example` templates for required environment variables.
2. **Code Review**:
   - Review all generated code for security vulnerabilities (SQL injection, unsafe file I/O, etc.).
   - Do not accept code you do not understand.
3. **Data Protection**:
   - Do not commit large binary files (>50MB) or personal data.

---

## 🐍 Python Coding Standards

### 1. Code Quality & Style

- **Logging vs. Print**:
  - ❌ **DO NOT** use `print()` statements for application output.
  - ✅ **USE** the `logging` module.
  - _Example_: `logger.info("Processing complete")` instead of `print("Processing complete")`.
- **Imports**:
  - ❌ **NO** wildcard imports (`from module import *`).
  - ✅ **Explicitly** import required classes/functions.
- **Exception Handling**:
  - ❌ **NO** bare `except:` clauses.
  - ✅ **Catch specific exceptions** (e.g., `except ValueError:`) or at least `except Exception:`.
- **Type Hinting**:
  - Use Python type hints for function arguments and return values.

### 2. Project Structure

```
project_name/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── src/
│   └── project_name/
│       ├── __init__.py
│       └── main.py
└── tests/
```

### 3. Testing

- Use `unittest` or `pytest`.
- Write unit tests for individual functions and integration tests for workflows.

---

## 🔢 MATLAB Coding Standards

### 1. Structure

```
matlab_project/
├── main.m
├── src/
│   ├── functions/
│   └── classes/
└── tests/
```

### 2. Best Practices

- Use clear comment blocks for function documentation.
- Avoid `.asv` and `.m~` files in commits (add to `.gitignore`).
- Use `functiontests` for testing.

---

## 🔄 Git Workflow & Version Control

### 1. Commit Messages

Use **Conventional Commits** format:

- `feat(scope): description` (New feature)
- `fix(scope): description` (Bug fix)
- `docs(scope): description` (Documentation)
- `style(scope): description` (Formatting)
- `refactor(scope): description` (Code restructuring)
- `test(scope): description` (Adding tests)
- `chore(scope): description` (Maintenance)

### 2. Branching Strategy

- `main`: Production-ready code.
- `develop`: Integration branch.
- `feature/name`: New features.
- `hotfix/name`: Critical bug fixes.

---

## 📝 Documentation

- **README.md**: Every project must have a README with Description, Installation, and Usage sections.
- **Docstrings**: Use Google or NumPy style docstrings for Python.
- **Comments**: Explain _why_, not just _what_.

---

## 🌐 Web Development Standards (HTML/CSS/JS)

### 1. HTML

- **Semantic HTML**: Use `<header>`, `<nav>`, `<main>`, `<footer>`, `<article>`, `<section>` appropriately.
- **Accessibility**: Ensure all `<img>` tags have `alt` attributes. Use ARIA labels where necessary.
- **Structure**: Maintain a clean and indented structure.

### 2. CSS

- **Naming Convention**: Use **BEM** (Block Element Modifier) for class names where possible (e.g., `.card__title--large`).
- **Responsiveness**: Design **Mobile-First**. Use media queries to adapt to larger screens.
- **Linting**: Use `stylelint` with standard config.
  - Avoid ID selectors for styling.
  - Avoid `!important`.

### 3. JavaScript

- **Modern Syntax**: Use ES6+ features (arrow functions, template literals, destructuring).
- **Variables**: Use `const` by default, `let` if reassignment is needed. ❌ **NEVER** use `var`.
- **Async/Await**: Prefer `async/await` over raw Promises/callbacks.
- **Linting**: Use `eslint`.
- **Equality**: Always use strict equality `===` and `!==`.

---

## ⚙️ C++ Coding Standards

### 1. Style Guide

- Follow the **Google C++ Style Guide**.
- **Formatting**: Use `clang-format`.
  - Indent width: 4 spaces (as seen in `.clang-format`).
  - Column limit: 0 (no hard limit, but keep it readable).
  - Brace wrapping: Allman style (braces on new line) is configured in some repos, but consistency within the specific repo is key.

### 2. Modern C++

- Use **C++11/14/17** features.
- **Memory Management**:
  - ❌ **Avoid** raw pointers (`new`/`delete`).
  - ✅ **Use** smart pointers: `std::unique_ptr` for exclusive ownership, `std::shared_ptr` for shared ownership.
- **RAII**: Use Resource Acquisition Is Initialization for resource management.

### 3. Safety

- Avoid C-style casts; use `static_cast`, `dynamic_cast`, etc.
- Initialize all variables upon declaration.

---

## 🚨 Emergency Procedures

If sensitive data is accidentally committed:

1.  **Stop** immediately.
2.  Use `git filter-branch` or BFG Repo-Cleaner to remove the file from history.
3.  Force push only if necessary and coordinated with the team.

---

## ⚖️ Governance & Process

### 1. Atomic Commits
- Commits must be atomic, focusing on a single task or fix.
- **NEVER** combine documentation updates with feature code or refactoring.
- **NEVER** introduce "Shadow CI/CD" or unauthorized workflows.

### 2. Feature Commits in Documentation PRs
- **STRICTLY FORBIDDEN:** Do not hide feature code inside documentation pull requests.
- All code changes must be visible and reviewed in appropriate PRs with correct labels (`feat`, `fix`, etc.).

### 3. CI/CD Truth
- The file `.github/workflows/ci-standard.yml` is the **only** source of truth for CI/CD.
- Unauthorized workflows will be automatically removed.

---

## 🛠️ GitHub CLI & Workflow Reference

Always use Github CLI for making pull requests. 
Whenever you finish a task for the user, push it to remote. 
NEVER try to use GitKraken or anything other than Github CLI for Pull request creation. 
All pull requests should be verified to pass the ruff, black, and mypy requirements in the ci / cd pipeline before they are created.

### For PR Creation:
- Always check if PR already exists first using `gh pr list --state open`
- Use simple, concise titles and descriptions for initial creation
- Wrap GitHub CLI commands in powershell `-Command "..."`
- Use single quotes inside double quotes for string parameters

### For PR Management:
- Use `gh pr view [number]` to get PR details and status
- Use `gh pr checks [number]` to see CI/CD status
- Use `gh run list --branch [branch-name]` to see workflow runs
- Check for failing checks and address them systematically

### For CI/CD Issue Resolution:
- Identify failing checks using `gh pr checks`
- Examine workflow run logs using `gh run view [run-id]`
- Make fixes on the same branch and push to update the PR
- Verify fixes by checking updated CI status

### Command Templates for Future Use:

```bash
# Create PR:
powershell -Command "gh pr create --title 'Your Title' --body 'Your description'"

# Check PR status:
powershell -Command "gh pr view [PR_NUMBER]"

# Check CI/CD status:
powershell -Command "gh pr checks [PR_NUMBER]"

# List recent runs:
powershell -Command "gh run list --branch [BRANCH_NAME] --limit 5"

# View specific run:
powershell -Command "gh run view [RUN_ID]"
```
