module.exports = {
  run: [
    // Ensure nodejs is installed (needed by yt-dlp as JS runtime for YouTube)
    {
      method: "shell.run",
      params: {
        message: "conda install -c conda-forge nodejs -y"
      }
    },
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
