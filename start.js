module.exports = {
  daemon: true,
  run: [
    // Step 1: Smart pre-launch check (deps hash + torch smoke test + auto-fix)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: ["python smart.py start"]
      }
    },
    // Step 2: Launch server
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          PYTORCH_ENABLE_MPS_FALLBACK: "1"
        },
        message: ["python app.py --port {{port}}"],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    // Step 3: Surface URL
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    },
    // Step 4: Open in system default browser (not Pinokio iframe)
    {
      method: "shell.run",
      params: {
        message: ["python -c \"import webbrowser; webbrowser.open('{{local.url}}')\""]
      }
    }
  ]
}
