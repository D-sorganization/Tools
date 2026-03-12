# Assessment G_Dependencies

## Grade: 4/10

### Assessment Notes

- AUTO-FIXED: Missing `requirements.txt` was added to declare `pythonnet` and `pytest`. System-level dependency (`mono` or .NET on Linux) is a major constraint not explicitly handled in automated setup scripts.
- Additional required dependencies such as `numpy`, `pandas` and `pyyaml` are missing from `requirements.txt`.
