module.exports = {
  run: [
    // Step 1: Install Python deps + auto-detect GPU + install correct torch
    //         smart.py handles: nvidia(CUDA12.8) / amd(DirectML|ROCm) /
    //         apple-arm(MPS) / fallback(CPU) + verify + auto CPU-fallback
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: ["python smart.py install"]
      }
    },
    // Step 2: Symlink env for disk-space savings
    {
      method: "fs.link",
      params: {
        venv: "app/env"
      }
    }
  ]
}
