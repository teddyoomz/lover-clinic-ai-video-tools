"""
Smart dependency checker.
Runs uv pip install only when requirements.txt has changed since last install.
Called by start.js before launching the server.
"""
import hashlib
import pathlib
import subprocess
import sys

req  = pathlib.Path("requirements.txt")
hfile = pathlib.Path("../.req_hash")

current = hashlib.md5(req.read_bytes()).hexdigest()
stored  = hfile.read_text().strip() if hfile.exists() else ""

if current == stored:
    print("✅ Dependencies up to date — skipping install.")
    sys.exit(0)

print("📦 requirements.txt changed — updating dependencies...")
result = subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"])

if result.returncode == 0:
    hfile.write_text(current)
    print("✅ Dependencies updated successfully.")
else:
    print("❌ Dependency install failed. Check output above.")
    sys.exit(1)
