module.exports = {
  run: [
    // Step 1: Install Python deps + auto-detect GPU + install correct torch
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: ["python smart.py install"]
      }
    },
    // Step 2: Symlink env for disk-space savings
    // NOTE: fs.link may overwrite CUDA torch with shared CPU version from cache
    {
      method: "fs.link",
      params: {
        venv: "app/env"
      }
    },
    // Step 3: Force-reinstall correct GPU torch after fs.link
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: ["python smart.py torch"]
      }
    },
    // Step 4: Auto-start after install completes
    {
      method: "script.start",
      params: {
        uri: "start.js"
      }
    }
  ]
}
