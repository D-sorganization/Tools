# GitHub Copilot Development & Version Control Rules

## 1. Repository Structure and Organization

### Python Project Structure
```
project_name/
├── README.md
├── requirements.txt
├── setup.py (for packages)
├── .gitignore
├── .env.example
├── src/
│   └── project_name/
│       ├── __init__.py
│       ├── main.py
│       ├── gui/
│       │   ├── __init__.py
│       │   └── main_window.py
│       └── utils/
│           ├── __init__.py
│           └── helpers.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── docs/
├── data/ (if applicable)
└── scripts/
```

### MATLAB Project Structure
```
matlab_project/
├── README.md
├── main.m
├── src/
│   ├── functions/
│   ├── classes/
│   └── gui/
├── data/
├── tests/
├── docs/
└── output/
```

## 2. Git Workflow Rules

### Branch Strategy
- **main/master**: Production-ready code only
- **develop**: Integration branch for features
- **feature/**: Individual features (`feature/add-login-gui`)
- **hotfix/**: Critical bug fixes (`hotfix/fix-crash-bug`)

### Commit Rules
1. **Atomic commits**: One logical change per commit
2. **Descriptive messages**: Use conventional commit format
   ```
   type(scope): description
   
   Examples:
   feat(gui): add user login dialog
   fix(data): resolve CSV parsing error
   docs: update installation instructions
   ```

3. **Commit message types**:
   - `feat`: New feature
   - `fix`: Bug fix
   - `docs`: Documentation
   - `style`: Formatting, no code change
   - `refactor`: Code restructuring
   - `test`: Adding tests
   - `chore`: Maintenance

### Pre-Commit Checklist
- [ ] Code runs without errors
- [ ] All tests pass
- [ ] No sensitive data (API keys, passwords)
- [ ] Requirements.txt updated (Python)
- [ ] Documentation updated if needed

## 3. GitHub Copilot Safety Rules

### Code Review Before Accepting
1. **Always review suggestions** before accepting
2. **Understand the code** - don't accept what you don't understand
3. **Check for security issues**:
   - Hardcoded credentials
   - SQL injection vulnerabilities
   - Unsafe file operations
   - Network security issues

### Copilot Usage Guidelines
- Use Copilot for **boilerplate** and **common patterns**
- **Verify algorithms** and complex logic manually
- **Test generated code** thoroughly
- **Document generated functions** in your own words
- **Customize suggestions** to match your coding style

### What to Double-Check
- Database queries and connections
- File I/O operations
- API calls and network requests
- Error handling and exception management
- Security-sensitive operations

## 4. File Management Rules

### .gitignore Essentials
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
.env
pip-log.txt
pip-delete-this-directory.txt

# MATLAB
*.asv
*.m~
slprj/
*.slx.autosave

# IDE
.vscode/settings.json
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Data and outputs
data/sensitive/
output/temp/
logs/
*.log

# Dependencies
node_modules/
```

### Sensitive Data Protection
1. **Never commit**:
   - API keys, tokens, passwords
   - Database connection strings
   - Personal data
   - Large binary files (>50MB)

2. **Use environment variables**:
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   API_KEY = os.getenv('API_KEY')
   ```

3. **Create .env.example**:
   ```
   API_KEY=your_api_key_here
   DATABASE_URL=your_database_url_here
   ```

## 5. Python-Specific Best Practices

### Virtual Environments
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements Management
```bash
# Generate requirements.txt
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

### PyQt6 GUI Standards
```python
# main_window.py template
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Application Name")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()
    
    def init_ui(self):
        # Initialize UI components
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

## 6. Testing and Quality Assurance

### Testing Strategy
- **Unit tests** for individual functions
- **Integration tests** for workflows
- **GUI tests** for PyQt6 applications

### Python Testing
```python
# test_example.py
import unittest
from src.project_name.main import your_function

class TestYourFunction(unittest.TestCase):
    def test_basic_functionality(self):
        result = your_function(input_data)
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()
```

### MATLAB Testing
```matlab
% test_function.m
function tests = test_function
    tests = functiontests(localfunctions);
end

function test_basic_functionality(testCase)
    result = your_function(input_data);
    verifyEqual(testCase, result, expected_output);
end
```

## 7. Documentation Standards

### README.md Template
```markdown
# Project Name

## Description
Brief description of what the project does.

## Installation
```bash
# Installation steps
```

## Usage
```python
# Usage examples
```

## Contributing
Guidelines for contributing to the project.

## License
Project license information.
```

### Code Documentation
- **Python**: Use docstrings (Google or NumPy style)
- **MATLAB**: Use comment blocks with clear descriptions

## 8. Backup and Recovery

### Remote Backup Strategy
1. **Multiple remotes**: GitHub + GitLab/Bitbucket backup
2. **Regular pushes**: Push daily at minimum
3. **Tagged releases**: Tag stable versions
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

### Local Backup
- Keep local copies of important data files
- Export environment configurations
- Document setup procedures

## 9. Collaboration Rules

### Pull Request Process
1. Create feature branch
2. Make changes and commit
3. Push to remote repository
4. Create pull request
5. Request code review
6. Address feedback
7. Merge after approval

### Code Review Checklist
- [ ] Code follows project conventions
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Performance considerations addressed

## 10. Emergency Procedures

### If You Accidentally Commit Sensitive Data
```bash
# Remove file from history (use with caution)
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch path/to/file' \
--prune-empty --tag-name-filter cat -- --all

# Force push (dangerous - use only if necessary)
git push origin --force --all
```

### Recovery Commands
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Recover deleted branch
git reflog
git checkout -b recovered-branch <commit-hash>
```

## 11. VS Code Settings for Copilot

### Recommended settings.json
```json
{
    "github.copilot.enable": {
        "*": true,
        "yaml": false,
        "plaintext": false
    },
    "github.copilot.editor.enableAutoCompletions": true,
    "github.copilot.inlineSuggest.enable": true,
    "files.autoSave": "onFocusChange",
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

## Quick Reference Commands

### Daily Workflow
```bash
# Start work
git pull origin main
git checkout -b feature/your-feature

# During work
git add .
git commit -m "feat: descriptive message"

# End of day
git push origin feature/your-feature

# Merge to main (after review)
git checkout main
git pull origin main
git merge feature/your-feature
git push origin main
git branch -d feature/your-feature
```

### Status Checks
```bash
git status          # Check current state
git log --oneline   # View commit history
git branch -a       # List all branches
git remote -v       # View remote repositories
```