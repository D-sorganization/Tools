# VS Code Setup Guide - Golf Swing Video Platform

## 🎯 Essential Extensions

### Must Install (Recommended)

I've created `.vscode/extensions.json` which will prompt you to install these automatically when you open the project in VS Code.

#### Core Development
1. **ESLint** (`dbaeumer.vscode-eslint`)
   - JavaScript/TypeScript linting
   - Shows errors in real-time

2. **Prettier** (`esbenp.prettier-vscode`)
   - Code formatter
   - Automatically formats on save

3. **TypeScript Hero** (`ms-vscode.vscode-typescript-next`)
   - Enhanced TypeScript support
   - Better IntelliSense

4. **Tailwind CSS IntelliSense** (`bradlc.vscode-tailwindcss`)
   - Autocomplete for Tailwind classes
   - Hover to see actual CSS

5. **Prisma** (`prisma.prisma`)
   - Database schema editing
   - Syntax highlighting for Prisma

#### React/Next.js Specific
6. **ES7+ React/Redux/React-Native snippets** (`dsznajder.es7-react-js-snippets`)
   - Code snippets for React
   - Speeds up development

7. **Auto Rename Tag** (`formulahendry.auto-rename-tag`)
   - Automatically renames paired HTML/JSX tags

#### Git
8. **GitLens** (`eamodio.gitlens`)
   - Enhanced Git integration
   - See who changed what and when

#### Code Quality
9. **Error Lens** (`usernamehw.errorlens`)
   - Shows errors inline
   - No need to open Problems panel

10. **Code Spell Checker** (`streetsidesoftware.code-spell-checker`)
    - Catches typos in code
    - Prevents embarrassing mistakes

#### Productivity
11. **Python** (`ms-python.python`) - If using MATLAB Runtime bridge
12. **YAML** (`redhat.vscode-yaml`) - For CI/CD configs
13. **JSON** (Built-in) - For JSON files

### Optional (Nice to Have)

14. **MATLAB** (`mathworks.matlab`)
    - If you want MATLAB syntax highlighting
    - Note: MATLAB itself runs separately

15. **Thunder Client** (`rangav.vscode-thunder-client`)
    - API testing inside VS Code
    - Alternative to Postman

16. **Todo Tree** (`gruntfuggly.todo-tree`)
    - Highlights TODO, FIXME, etc.
    - Helps track tasks

---

## ⚙️ Settings Configuration

I've created `.vscode/settings.json` with optimal settings for this project:

### Key Settings

- **Format on Save**: Automatically formats code when you save
- **Organize Imports**: Automatically organizes imports
- **ESLint Auto-fix**: Automatically fixes ESLint errors
- **Tab Size**: 2 spaces (standard for JavaScript/TypeScript)
- **Tailwind CSS IntelliSense**: Enabled with experimental features
- **TypeScript**: Uses workspace TypeScript version

### What This Means for You

1. **Write code** - No need to manually format
2. **Save** - Code automatically formatted and organized
3. **Errors show immediately** - Inline in the editor
4. **Tailwind autocomplete** - When you type `className="..."`
5. **TypeScript IntelliSense** - Better autocomplete and error checking

---

## 🚀 Quick Start

### 1. Install Extensions

When you open the project in VS Code, you'll see a notification:
"This workspace has extension recommendations"

Click **"Install All"** to install all recommended extensions.

### Manual Installation (if needed):

```
Ctrl+Shift+X (Windows/Linux) or Cmd+Shift+X (Mac)
Search and install each extension from the list above
```

### 2. Verify Installation

After installing extensions, you should see:
- ✅ ESLint status in bottom-right corner
- ✅ Prettier status in bottom-right corner
- ✅ Better autocomplete in TypeScript files
- ✅ Tailwind class suggestions when typing `className`

### 3. Test Formatting

Open any TypeScript file and:
1. Add some messy code
2. Press `Ctrl+S` (or `Cmd+S` on Mac)
3. Code should automatically format

---

## 🎨 Using the Extensions

### ESLint

**What it does**: Finds errors and bad patterns in your code

**Example**:
```typescript
const myVariable; // ESLint will show error: must have initializer
```

**Auto-fix**:
- Save file → ESLint auto-fixes issues
- Or right-click → "Fix all ESLint problems"

### Prettier

**What it does**: Formats code consistently

**Example**:
```typescript
// You write:
const x={a:1,b:2}

// Prettier formats to:
const x = { a: 1, b: 2 };
```

**Manual format**: `Shift+Alt+F` (or `Shift+Option+F` on Mac)

### Tailwind CSS IntelliSense

**What it does**: Autocompletes Tailwind classes

**Example**:
Type `className="bg-` → See all background colors
Hover over `className="text-center"` → See actual CSS

### Prisma

**What it does**: Helps with database schema editing

**Example**:
Type `model User` → Get autocomplete for Prisma schema
Syntax highlighting for `.prisma` files

---

## 🔧 Troubleshooting

### Extensions Not Working?

1. **Reload VS Code**: `Ctrl+Shift+P` → "Reload Window"
2. **Check Extension Status**: Bottom-right corner should show extension icons
3. **Re-install**: Uninstall and reinstall the extension

### Formatting Not Working?

1. **Check Default Formatter**:
   - Right-click in file → "Format Document With..."
   - Select Prettier

2. **Check Settings**: Open `.vscode/settings.json`
   - Should have `"editor.formatOnSave": true`

3. **Check Prettier**:
   - Open any file
   - `Shift+Alt+F` should format

### ESLint Not Working?

1. **Check Output**:
   - `View` → `Output`
   - Select "ESLint" from dropdown
   - Look for errors

2. **Reload**: `Ctrl+Shift+P` → "ESLint: Restart ESLint Server"

### IntelliSense Not Working?

1. **Check TypeScript**:
   - Bottom-right corner
   - Should show TypeScript version

2. **Reload**: `Ctrl+Shift+P` → "TypeScript: Restart TS Server"

---

## 📚 Keyboard Shortcuts

These will make you faster:

| Shortcut | Action |
|----------|--------|
| `Ctrl+S` | Save (also auto-formats) |
| `Shift+Alt+F` | Format Document |
| `Ctrl+Shift+P` | Command Palette |
| `Ctrl+P` | Quick File Open |
| `Ctrl+` | Toggle Terminal |
| `Ctrl+B` | Toggle Sidebar |
| `F5` | Debug: Start |
| `Ctrl+F` | Find |
| `Ctrl+Shift+F` | Find in Files |
| `Alt+Up/Down` | Move line up/down |
| `Shift+Alt+Up/Down` | Copy line up/down |
| `Ctrl+/` | Toggle comment |

---

## 🎓 Next Steps

1. **Install Extensions**: VS Code will prompt you, or install manually
2. **Reload VS Code**: `Ctrl+Shift+P` → "Reload Window"
3. **Start Coding**: Open any `.tsx` or `.ts` file
4. **See Magic**: Auto-complete, formatting, and linting should work!

---

## 💡 Pro Tips

### 1. Multi-cursor Editing
```
Alt+Click: Add cursor
Ctrl+Alt+Up/Down: Add cursor above/below
Shift+Alt+Left/Right: Select word
```

### 2. Quick Refactoring
```
F2: Rename symbol (rename variable everywhere)
Ctrl+. : Quick fixes
```

### 3. Code Snippets
Type in `.tsx` file:
```
rafce → React arrow function component with export
```

### 4. Emmet (HTML/JSX)
Type and press Tab:
```
div#container → <div id="container"></div>
.className → <div className="className"></div>
```

---

**Your VS Code is now optimized for React/Next.js development! 🚀**

*If you have issues, check the Troubleshooting section above or check the extension documentation.*
