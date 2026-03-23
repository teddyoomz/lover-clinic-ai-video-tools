module.exports = {
  run: [
    // Step 1: Install Python dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "uv pip install pydantic==2.10.6"
        ]
      }
    },
    // Step 2: Install PyTorch — auto-detects GPU/CPU/platform
    //   nvidia win32  → CUDA 12.8
    //   nvidia linux  → CUDA 12.8
    //   amd win32     → DirectML
    //   amd linux     → ROCm 6.3
    //   apple arm64   → CPU index (MPS at runtime)
    //   apple x86     → CPU
    //   fallback      → CPU
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    // Step 3: Verify torch loads on THIS machine.
    //   Prints hardware report (GPU name, VRAM, driver).
    //   If CUDA DLL fails (WinError 127), auto-falls back to CPU torch.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: ["python verify_torch.py"]
      }
    },
    // Step 4: Symlink env for disk-space savings
    {
      method: "fs.link",
      params: {
        venv: "app/env"
      }
    }
  ]
}
