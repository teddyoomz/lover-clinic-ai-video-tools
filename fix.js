module.exports = {
  run: [
    // Diagnose & auto-repair: broken torch, stale deps, CUDA DLL issues
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: ["python smart.py fix"]
      }
    }
  ]
}
