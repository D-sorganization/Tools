# Cursor IDE Development & Version Control Rules

## 1. Repository Structure and Organization

### Python Project Structure
```
project_name/
├── README.md
├── requirements.txt
├── setup.py (for packages)
├── .gitignore
├── .env.example
├── .cursor-rules (Cursor-specific)
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
├── .cursor-rules
├── src/
│   ├── functions/
│   ├── classes/
│   └── gui/
├── data/
├── tests/
├── docs/
└── output/
```

## 2. Cursor-Specific Configuration

### .cursor-rules File
Create a `.cursor-rules` file in your project root:

```
# Python Development Rules
- Use Python 3.9+ syntax and best practices
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Prefer dataclasses over dictionaries for structured data
- Use pathlib instead of os.path for file operations
- Always include docstrings for functions and classes

# PyQt6 GUI Rules
- Separate UI logic from business logic
- Use signals and slots for event handling
- Follow Qt naming conventions (camelCase for methods)
- Create reusable custom widgets
- Use Qt Designer files (.ui) when appropriate

# MATLAB Rules
- Use descriptive function and variable names
- Include function documentation headers
- Validate input arguments at function start
- Use vectorized operations instead of loops when possible
- Follow MATLAB naming conventions (camelCase for functions)

# Version Control Rules
- Make atomic commits with descriptive messages
- Use conventional commit format: type(scope): description
- Never commit secrets, API keys, or sensitive data
- Update requirements.txt when adding dependencies
- Write clear commit messages explaining the "why"

# Code Quality Rules
- Add error handling for external dependencies
- Write unit tests for core functionality
- Use logging instead of print statements for debugging
- Validate user inputs in GUI applications
- Handle exceptions gracefully with user-friendly messages
```

### Cursor Settings Configuration
Create `.vscode/settings.json` (Cursor uses VS Code settings):

```json
{
    "cursor.cpp.disabledLanguages": ["plaintext", "markdown"],
    "cursor.general.enableCtrlKChatShortcut": true,
    "cursor.chat.model": "gpt-4",
    "cursor.autocomplete.enabled": true,
    "cursor.autocomplete.model": "claude-3.5-sonnet",
    "cursor.autocomplete.acceptOnTab": true,
    "cursor.autocomplete.suggestOnEveryChange": true,
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "files.autoSave": "onFocusChange",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

## 3. Git Workflow Rules

### Branch Strategy
- **main/master**: Production-ready code only
- **develop**: Integration branch for features
- **feature/**: Individual features (`feature/add-login-gui`)
- **hotfix/**: Critical bug fixes (`hotfix/fix-crash-bug`)

### Commit Rules with Cursor
1. **Use Cursor's AI commit messages**: Let Cursor suggest commit messages, then review and edit
2. **Conventional commit format**:
   ```
   type(scope): description
   
   Examples:
   feat(gui): add user authentication dialog
   fix(data): resolve CSV parsing memory leak
   docs: update API documentation
   refactor(utils): optimize helper functions
   ```

3. **Cursor-enhanced commit workflow**:
   - Use Ctrl+Shift+G to open source control
   - Stage changes selectively
   - Use Cursor's AI to generate commit messages
   - Review and customize the AI suggestion
   - Add detailed description if needed

### Pre-Commit Checklist
- [ ] Code runs without errors
- [ ] Cursor's AI suggestions reviewed and understood
- [ ] All tests pass
- [ ] No sensitive data (API keys, passwords)
- [ ] Requirements.txt updated (Python)
- [ ] Documentation updated if needed
- [ ] Cursor rules followed

## 4. Cursor AI Safety Rules

### AI Code Review Process
1. **Always review AI suggestions** before accepting
2. **Use Cursor Chat (Ctrl+L)** to ask questions about generated code
3. **Verify complex algorithms** by asking Cursor to explain them
4. **Test AI-generated code** thoroughly before committing

### Cursor Chat Best Practices
- **Ask for explanations**: "Explain this function's logic"
- **Request alternatives**: "Show me a different approach to this problem"
- **Seek optimization**: "How can I make this code more efficient?"
- **Security review**: "Are there any security issues with this code?"

### What to Double-Check with Cursor
- Use chat to verify database queries and connections
- Ask Cursor to review file I/O operations for safety
- Request security analysis for API calls
- Have Cursor explain error handling strategies
- Verify algorithm correctness for complex calculations

### Cursor-Specific Safety Commands
```
# In Cursor Chat (Ctrl+L):
"Review this code for security vulnerabilities"
"Explain what this function does step by step"
"Are there any edge cases I should consider?"
"How can I improve error handling here?"
"Is this the most efficient approach?"
```

## 5. File Management Rules

### .gitignore for Cursor Projects
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

# MATLAB
*.asv
*.m~
slprj/
*.slx.autosave

# Cursor specific
.cursor/
*.cursor-chat

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

# AI generated files (review before ignoring)
# ai_generated/
```

### Cursor Project Configuration
Create `.cursor/project.json`:
```json
{
    "name": "your-project-name",
    "description": "Brief project description for AI context",
    "language": "python",
    "framework": "pyqt6",
    "preferences": {
        "coding_style": "pep8",
        "documentation_style": "google",
        "testing_framework": "unittest"
    }
}
```

## 6. Python-Specific Best Practices with Cursor

### Enhanced Virtual Environment Workflow
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Let Cursor know about your environment
# Use Ctrl+Shift+P -> "Python: Select Interpreter"
```

### Cursor-Enhanced PyQt6 Development
```python
# main_window.py - Cursor optimized template
import sys
from typing import Optional
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, pyqtSignal

class MainWindow(QMainWindow):
    """Main application window with Cursor AI assistance."""
    
    # Define signals for better separation of concerns
    data_changed = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Application Name")
        self.setGeometry(100, 100, 800, 600)
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self) -> None:
        """Initialize the user interface components."""
        # Use Cursor to generate UI components
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        # Add your widgets here
    
    def _connect_signals(self) -> None:
        """Connect signals and slots."""
        # Use Cursor to help with signal connections
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

### Cursor Code Generation Workflow
1. **Write comments first**: Describe what you want in comments
2. **Use Cursor's autocomplete**: Press Tab to accept suggestions
3. **Refine with chat**: Use Ctrl+L to ask for improvements
4. **Iterate**: Use Cursor to refactor and optimize

## 7. Testing Strategy with Cursor

### Cursor-Assisted Test Writing
```python
# test_example.py - Let Cursor help generate tests
import unittest
from unittest.mock import patch, MagicMock
from src.project_name.main import YourClass

class TestYourClass(unittest.TestCase):
    """Test cases for YourClass - use Cursor to generate test scenarios."""
    
    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.instance = YourClass()
    
    def test_basic_functionality(self) -> None:
        """Test basic functionality - ask Cursor for edge cases."""
        # Use Cursor chat: "Generate test cases for this function"
        result = self.instance.your_method("test_input")
        self.assertEqual(result, "expected_output")
    
    @patch('src.project_name.main.external_dependency')
    def test_with_mocking(self, mock_dependency: MagicMock) -> None:
        """Test with mocked dependencies."""
        # Cursor can help generate mock scenarios
        mock_dependency.return_value = "mocked_result"
        result = self.instance.method_using_dependency()
        self.assertEqual(result, "expected_result")

if __name__ == '__main__':
    unittest.main()
```

### MATLAB Testing with Cursor
```matlab
% test_function.m - Cursor can help generate MATLAB tests
function tests = test_function
    % Ask Cursor: "Generate comprehensive test cases for this MATLAB function"
    tests = functiontests(localfunctions);
end

function test_basic_functionality(testCase)
    % Use Cursor to suggest test scenarios
    input_data = [1, 2, 3, 4, 5];
    expected_output = [2, 4, 6, 8, 10];
    result = your_function(input_data);
    verifyEqual(testCase, result, expected_output, 'AbsTol', 1e-10);
end

function test_edge_cases(testCase)
    % Let Cursor suggest edge cases to test
    verifyError(testCase, @()your_function([]), 'MATLAB:expectedNonempty');
end
```

## 8. Documentation with Cursor AI

### AI-Enhanced Documentation Workflow
1. **Write code first**: Focus on functionality
2. **Use Cursor for docstrings**: Select function, use Ctrl+L: "Generate docstring"
3. **Enhance README**: Ask Cursor to help improve documentation
4. **Generate examples**: Use AI to create usage examples

### Cursor-Generated Documentation Template
```python
def complex_function(data: list, threshold: float = 0.5) -> dict:
    """
    Process data based on threshold criteria.
    
    This function analyzes input data and applies filtering based on the
    specified threshold value. Generated with Cursor AI assistance.
    
    Args:
        data: List of numerical values to process
        threshold: Filtering threshold (default: 0.5)
    
    Returns:
        Dictionary containing processed results with keys:
        - 'filtered': List of values above threshold
        - 'count': Number of filtered items
        - 'average': Average of filtered values
    
    Raises:
        ValueError: If data is empty or threshold is negative
        TypeError: If data contains non-numerical values
    
    Example:
        >>> result = complex_function([0.2, 0.8, 0.6, 0.3], 0.5)
        >>> print(result['filtered'])
        [0.8, 0.6]
    """
    # Implementation here
    pass
```

## 9. Cursor Workflow Optimizations

### Daily Development Routine
1. **Morning setup**:
   ```bash
   git pull origin main
   git checkout -b feature/your-feature
   # Open Cursor, activate virtual environment
   ```

2. **During development**:
   - Use Ctrl+K for quick AI edits
   - Use Ctrl+L for complex questions
   - Use Ctrl+Shift+L for file-specific questions
   - Regular commits with AI-suggested messages

3. **End of day**:
   ```bash
   # Use Cursor to review changes before commit
   git add .
   # Let Cursor suggest commit message, then review
   git commit -m "AI-suggested message (reviewed and approved)"
   git push origin feature/your-feature
   ```

### Cursor Keyboard Shortcuts
- **Ctrl+K**: Quick AI edit
- **Ctrl+L**: Open AI chat
- **Ctrl+Shift+L**: Chat about current file
- **Ctrl+I**: Inline AI suggestions
- **Tab**: Accept AI autocomplete
- **Esc**: Reject AI suggestion

## 10. Security and Privacy with Cursor

### Data Privacy Rules
1. **Review code before sending**: Understand what Cursor sees
2. **Avoid sensitive data**: Don't include API keys in prompts
3. **Use local models when possible**: Configure for sensitive projects
4. **Regular privacy audits**: Review what data has been shared

### Cursor Privacy Settings
```json
{
    "cursor.privacy.enableTelemetry": false,
    "cursor.privacy.includeCodeContext": true,
    "cursor.privacy.includeFileContents": false,
    "cursor.general.enableCtrlKChatShortcut": true,
    "cursor.chat.clearHistoryOnExit": true
}
```

## 11. Troubleshooting and Recovery

### Common Cursor Issues
1. **AI suggestions not working**:
   - Check internet connection
   - Restart Cursor
   - Clear AI cache: Ctrl+Shift+P -> "Cursor: Clear Cache"

2. **Performance issues**:
   - Disable AI for large files
   - Use .gitignore to exclude AI from data folders
   - Reduce autocomplete frequency

### Emergency Git Commands
```bash
# Undo AI-generated changes
git reset --hard HEAD~1

# Create backup before major AI refactoring
git branch backup-before-ai-refactor

# Compare AI suggestions
git diff HEAD~1

# Recover from AI-generated errors
git checkout HEAD~1 -- problematic_file.py
```

## 12. Advanced Cursor Techniques

### Multi-file AI Operations
1. **Project-wide refactoring**:
   - Select multiple files
   - Use Ctrl+Shift+L: "Refactor this pattern across all files"

2. **Consistent naming**:
   - Ask Cursor: "Make variable names consistent across the project"

3. **Architecture review**:
   - Use chat: "Review the overall architecture of this project"

### Custom AI Instructions
Add to your `.cursor-rules`:
```
# Custom Instructions for Your Project
- Always use type hints in Python functions
- Prefer composition over inheritance
- Use dependency injection for testability
- Follow the single responsibility principle
- Write defensive code with proper error handling
- Use logging.getLogger(__name__) for logging
- Implement proper resource cleanup (context managers)
- Use pathlib for all file operations
- Prefer f-strings for string formatting
- Use dataclasses for structured data
```

## Quick Reference Commands

### Daily Cursor Workflow
```bash
# Start development session
git status
git pull origin main
cursor .  # Open Cursor in current directory

# During development (in Cursor)
# Ctrl+K: Quick edits
# Ctrl+L: Complex questions
# Tab: Accept suggestions
# Regular commits with AI help

# End session
git add .
git commit -m "feat: AI-assisted feature implementation"
git push origin feature-branch
```

### AI Assistance Commands
```
# In Cursor Chat (Ctrl+L):
"Explain this error message"
"Optimize this function for performance"
"Add error handling to this code"
"Generate unit tests for this class"
"Refactor this code to be more readable"
"Check for potential bugs in this implementation"
"Suggest improvements for this algorithm"
```

### Code Quality Checks
```
# Ask Cursor to review:
"Review this code for Python best practices"
"Check for potential security vulnerabilities"
"Suggest performance optimizations"
"Verify error handling is adequate"
"Ensure proper type hints are used"
```