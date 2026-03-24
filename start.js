module.exports = {
  daemon: true,
  run: [
    // Step 1: Smart pre-launch check + auto-heal (deps hash, torch, core AI imports)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: ["python smart.py start"]
      }
    },
    // Step 2: Launch server — detect URL OR import-error crash for auto-heal
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          PYTORCH_ENABLE_MPS_FALLBACK: "1"
        },
        message: ["python app.py --port {{port}}"],
        on: [
          // Normal: server started → capture URL
          {
            event: "/(http:\\/\\/[0-9.:]+)/",
            done: true
          },
          // Auto-heal: import error → run full fix then re-start
          {
            event: "/ImportError|ModuleNotFoundError|No module named/i",
            done: true
          }
        ]
      }
    },
    // Step 3: Surface URL (or, if import error triggered, url will be empty → restart)
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    },
    // Step 4: Open in system default browser
    {
      when: "{{local.url}}",
      method: "shell.run",
      params: {
        message: ["python -c \"import webbrowser; webbrowser.open('{{local.url}}')\""]
      }
    }
  ]
}
