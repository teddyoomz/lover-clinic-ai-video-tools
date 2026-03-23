module.exports = {
  daemon: true,
  run: [
    // Step 1: Pull latest code from GitHub
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    // Step 2: Smart dep check — reinstalls ONLY if requirements.txt changed
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: ["python check_deps.py"]
      }
    },
    // Step 3: Launch server
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          PYTORCH_ENABLE_MPS_FALLBACK: "1"
        },
        message: [
          "python app.py --port {{port}}"
        ],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    // Step 4: Surface the URL
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
