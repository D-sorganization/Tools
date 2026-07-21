import subprocess
print("Running calculator tests...")
res = subprocess.run(["uv", "run", "pytest", "src/web_applications/calculator/tests"], capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
