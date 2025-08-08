# Branching Workflow Rule

## Overview
This document establishes the mandatory branching workflow for all development work on this project.

## Rule: Always Use Feature Branches

**ALL changes to the codebase must be made in feature branches, not directly on the main branch.**

## Workflow Process

### 1. Creating a Feature Branch
```bash
# Always start from main branch
git checkout main
git pull origin main

# Create and switch to a new feature branch
git checkout -b feature/descriptive-feature-name
```

### 2. Making Changes
- Make all your changes in the feature branch
- Commit frequently with descriptive commit messages
- Test thoroughly before considering merge

### 3. Testing Before Merge
- Ensure all functionality works as expected
- Run any existing tests
- Test the integrated data processor if changes affect it
- Verify no regressions in existing features

### 4. Merging to Main
```bash
# Switch back to main
git checkout main
git pull origin main

# Merge the feature branch
git merge feature/descriptive-feature-name

# Push to remote
git push origin main

# Clean up - delete the feature branch
git branch -d feature/descriptive-feature-name
git push origin --delete feature/descriptive-feature-name
```

## Branch Naming Convention

Use descriptive names that clearly indicate the purpose:
- `feature/folder-tool-reorganization`
- `feature/new-data-processing-feature`
- `feature/bugfix-plotting-issue`
- `feature/ui-improvements`

## Exceptions

**NO EXCEPTIONS** - This rule applies to:
- Bug fixes
- New features
- Documentation updates
- Configuration changes
- Any code modifications

## Benefits

1. **Safety**: Prevents breaking the main branch
2. **Collaboration**: Allows multiple developers to work on different features
3. **Testing**: Enables thorough testing before integration
4. **Rollback**: Easy to revert changes if issues are discovered
5. **History**: Clean commit history on main branch

## Enforcement

- All pull requests must come from feature branches
- Direct commits to main are not allowed
- Code reviews should verify the branching workflow was followed

## Example Workflow

```bash
# Starting a new feature
git checkout main
git pull origin main
git checkout -b feature/add-new-export-format

# Making changes and commits
# ... make changes ...
git add .
git commit -m "Add new export format functionality"

# ... more changes ...
git add .
git commit -m "Add tests for new export format"

# Testing and finalizing
# ... test thoroughly ...

# Merging to main
git checkout main
git pull origin main
git merge feature/add-new-export-format
git push origin main
git branch -d feature/add-new-export-format
```

## Date Established
January 2025 - This rule is effective immediately and applies to all future development work.
