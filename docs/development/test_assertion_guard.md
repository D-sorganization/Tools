# Changed Test Assertion Guard

`scripts/check_test_assertions.py` blocks changed Python test files that contain
no behavioral assertion. It accepts standard `assert` statements,
`pytest.raises(...)`, `self.assertRaises(...)`, and unittest/mock-style assert
method calls.

Run locally:

```bash
python scripts/check_test_assertions.py
```

CI passes its changed Python file list explicitly:

```bash
python scripts/check_test_assertions.py --changed-files changed_python_files.txt
```

Fixture-only files must be explicitly allowlisted in
`scripts/test_assertion_allowlist.txt`; keep patterns narrow and repository
relative.
