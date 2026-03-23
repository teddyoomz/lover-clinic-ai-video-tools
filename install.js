module.exports = {
  run: [
    // Step 1: Install ffmpeg via conda (skip if already available)
    {
      method: "shell.run",
      params: {
        when: "{{!which('ffmpeg')}}",
        message: "conda install -c conda-forge ffmpeg -y"
      }
    },
    // Step 2: Install Python dependencies
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
    // Step 3: Install PyTorch (platform-specific via torch.js)
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
    // Step 4: Symlink env for disk-space savings
    {
      method: "fs.link",
      params: {
        venv: "app/env"
      }
    }
  ]
}
