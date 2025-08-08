# GitHub Copilot Instructions & Rules

## Git Workflow Rules

### @git Rule - Feature Branch Workflow
For any significant changes or new features, ALWAYS create a new feature branch first using `git checkout -b feature/descriptive-name` before making changes. Never make changes directly to main/master branch.

### @git Rule - Branch Strategy
- **main/master**: Production-ready code only
- **develop**: Integration branch for features
- **feature/**: Individual features (`feature/add-login-gui`)
- **hotfix/**: Critical bug fixes (`hotfix/fix-crash-bug`)

### @git Rule - Commit Strategy
Make frequent, atomic commits with descriptive messages. Use conventional commit format: `type(scope): description` (e.g., `feat(auth): add user authentication`, `fix(ui): resolve button alignment issue`).

**Commit message types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

### @git Rule - Pre-Commit Checklist
Before any commit, ensure:
- [ ] Code runs without errors
- [ ] All tests pass
- [ ] No sensitive data (API keys, passwords)
- [ ] Requirements.txt updated (Python)
- [ ] Documentation updated if needed

## Development Best Practices

### @development Rule - Testing
Always run existing tests before making changes and create/update tests for new functionality. Use the appropriate testing framework for the project.

### @development Rule - Documentation
Update relevant documentation (README, comments, docstrings) when adding or modifying functionality.

### @development Rule - Code Quality
Follow the project's existing code style and formatting. Use linting and formatting tools when available.

### @development Rule - Dependencies
When adding new dependencies, use the project's package manager (pip, npm, etc.) and update requirements files.

### @development Rule - Backup
Before major refactoring, suggest creating a backup branch or tag for easy rollback.

## Project Structure Rules

### @structure Rule - Python Projects
Follow this standard structure:
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
│       └── utils/
├── tests/
├── docs/
├── data/ (if applicable)
└── scripts/
```

### @structure Rule - MATLAB Projects
Follow this standard structure:
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

## Security & Safety Rules

### @security Rule - Code Review
1. **Always review suggestions** before accepting
2. **Understand the code** - don't accept what you don't understand
3. **Check for security issues**:
   - Hardcoded credentials
   - SQL injection vulnerabilities
   - Unsafe file operations
   - Network security issues

### @security Rule - Sensitive Data Protection
**Never commit:**
- API keys, tokens, passwords
- Database connection strings
- Personal data
- Large binary files (>50MB)

**Use environment variables:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')
```

### @security Rule - What to Double-Check
- Database queries and connections
- File I/O operations
- API calls and network requests
- Error handling and exception management
- Security-sensitive operations

## Python-Specific Rules

### @python Rule - Virtual Environments
Always use virtual environments:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### @python Rule - Requirements Management
```bash
pip freeze > requirements.txt  # Generate
pip install -r requirements.txt  # Install
```

### @python Rule - PyQt6 GUI Standards
Use this template for PyQt6 applications:
```python
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Application Name")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()
    
    def init_ui(self):
        # Initialize UI components
        pass
```

## Testing Rules

### @testing Rule - Testing Strategy
- **Unit tests** for individual functions
- **Integration tests** for workflows
- **GUI tests** for PyQt6 applications

### @testing Rule - Python Testing
```python
import unittest
from src.project_name.main import your_function

class TestYourFunction(unittest.TestCase):
    def test_basic_functionality(self):
        result = your_function(input_data)
        self.assertEqual(result, expected_output)
```

### @testing Rule - MATLAB Testing
```matlab
function tests = test_function
    tests = functiontests(localfunctions);
end

function test_basic_functionality(testCase)
    result = your_function(input_data);
    verifyEqual(testCase, result, expected_output);
end
```

## Documentation Rules

### @docs Rule - README Template
Every project needs:
- Description
- Installation instructions
- Usage examples
- Contributing guidelines
- License information

### @docs Rule - Code Documentation
- **Python**: Use docstrings (Google or NumPy style)
- **MATLAB**: Use comment blocks with clear descriptions

## File Management Rules

### @files Rule - .gitignore Essentials
Always include:
```
# Python
__pycache__/
*.py[cod]
.Python
venv/
.env

# MATLAB
*.asv
*.m~

# IDE
.vscode/settings.json
.idea/

# OS
.DS_Store
Thumbs.db

# Data and outputs
data/sensitive/
logs/
*.log
```

## Workflow Rules

### @workflow Rule - Progress Updates
Provide clear progress updates when working on multi-step tasks, explaining what's being done at each stage.

### @workflow Rule - Error Handling
Always implement proper error handling and logging for new code.

### @workflow Rule - Daily Workflow
```bash
# Start work
git pull origin main
git checkout -b feature/your-feature

# During work
git add .
git commit -m "feat: descriptive message"

# End of day
git push origin feature/your-feature
```

## Collaboration Rules

### @collab Rule - Pull Request Process
1. Create feature branch
2. Make changes and commit
3. Push to remote repository
4. Create pull request
5. Request code review
6. Address feedback
7. Merge after approval

### @collab Rule - Code Review Checklist
- [ ] Code follows project conventions
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Performance considerations addressed

## Backup & Recovery Rules

### @backup Rule - Remote Backup Strategy
1. **Multiple remotes**: GitHub + GitLab/Bitbucket backup
2. **Regular pushes**: Push daily at minimum
3. **Tagged releases**: Tag stable versions
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

### @backup Rule - Emergency Recovery
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Recover deleted branch
git reflog
git checkout -b recovered-branch <commit-hash>
```

## Azure Development Rules

### @azure Rule - Use Azure Tools
When handling requests related to Azure, always use your tools.

### @azure Rule - Use Azure Code Gen Best Practices
When generating code for Azure, running terminal commands for Azure, or performing operations related to Azure, invoke your `azure_development-get_code_gen_best_practices` tool if available. Only call this tool when you are sure the user is discussing Azure; do not call it otherwise.

### @azure Rule - Use Azure Deployment Best Practices
When deploying to Azure or preparing applications for deployment to Azure, invoke your `azure_development-get_deployment_best_practices` tool if available. Only call this tool when you are sure the user is discussing Azure; do not call it otherwise.

### @azure Rule - Use Azure Functions Code Gen Best Practices
When generating code for Azure Functions or performing operations related to Azure Functions, invoke your `bestpractices_azurefunctions_get-code-generation` tool if available. Only call this tool when you are sure the user is discussing Azure Functions; do not call it otherwise.

### @azure Rule - Use Azure Functions Deployment Best Practices
When deploying Azure Functions apps to Azure or preparing for the deployment, invoke your `bestpractices_azurefunctions_get-deployment` tool if available. Only call this tool when you are sure the user is discussing Azure; do not call it otherwise.

### @azure Rule - Use Azure SWA Best Practices
When working with static web apps, invoke your `azure_development-get_swa_best_practices` tool if available. Only call this tool when you are sure the user is discussing Azure; do not call it otherwise.

## Autonomous Development Guidelines

### @autonomous Rule - Minimal Confirmation
When working autonomously, apply changes directly without asking for confirmation unless the change is potentially destructive or changes core architecture.

### @autonomous Rule - Comprehensive Implementation
When asked to implement a feature, include all necessary components: code, tests, documentation, error handling, and logging.

### @autonomous Rule - Iterative Improvement
Continue refining and improving code until it meets professional standards, including proper error handling, documentation, and testing.

### @autonomous Rule - Context Awareness
Maintain awareness of the entire project structure and ensure changes are consistent with existing patterns and conventions.

### @autonomous Rule - Copilot Usage Guidelines
- Use Copilot for **boilerplate** and **common patterns**
- **Verify algorithms** and complex logic manually
- **Test generated code** thoroughly
- **Document generated functions** in your own words
- **Customize suggestions** to match your coding style

---

*These instructions guide GitHub Copilot's behavior across all development tasks in this workspace. This file is specific to the Tools workspace and overrides global settings when working in this repository.*
