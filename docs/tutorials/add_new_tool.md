# How to Add a New Tool

The Tools Repository uses a plugin-based architecture managed by `PluginManager`. Adding a new tool is easy!

## Step 1: Create Your Tool

Create your script or application in the `tools/` or `python/src/` directory.
Example: `python/src/my_new_tool/app.py`

```python
def main():
    print("Hello from my new tool!")

if __name__ == "__main__":
    main()
```

## Step 2: Register in `tools.json`

Open `tools.json` in the repository root and add an entry under the appropriate category.

```json
{
  "My Category": [
    {
      "name": "My New Tool",
      "path": "python/src/my_new_tool/app.py",
      "type": "python",
      "desc": "A brief description of what this tool does"
    }
  ]
}
```

### Supported Types

- `"python"`: Launches a Python script.
- `"matlab"`: Launches a MATLAB script.
- `"bat"` / `"sh"`: Launches a shell script.
- `"browser"`: Opens an HTML file or URL in the default browser.

## Step 3: Test It

Run `UnifiedToolsLauncher.py`. Your tool should appear in the menu!

## Advanced: Auto-Discovery

The `PluginManager` is designed to support future auto-discovery. Currently, explicit registration in `tools.json` is required to ensure order and metadata quality.
