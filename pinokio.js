module.exports = {
  version: "5.0",
  title: "Lover Clinic AI Video Tools",
  description: "🔥 AI Image & Video Suite — Photo Upscale, Video Upscale, BG Removal, Enhancement & Image Tools",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let installed = info.exists("app/env")
    let running = {
      install: info.running("install.js"),
      start:   info.running("start.js"),
      update:  info.running("update.js"),
      reset:   info.running("reset.js"),
      fix:     info.running("fix.js"),
    }

    // ── Installing ───────────────────────────────────────────
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-spinner fa-spin",
        text: "Installing…",
        href: "install.js",
      }]
    }

    // ── Fixing ───────────────────────────────────────────────
    if (running.fix) {
      return [{
        default: true,
        icon: "fa-solid fa-spinner fa-spin",
        text: "Fixing…",
        href: "fix.js",
      }]
    }

    // ── Installed ────────────────────────────────────────────
    if (installed) {

      if (running.start) {
        let local = info.local("start.js")
        if (local && local.url) {
          // App is live — show Terminal only (app already opened in system browser)
          // Do NOT use href: local.url — that would load Gradio inside Pinokio webview (wastes RAM)
          return [{
            default: true,
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }]
        }
        return [{
          default: true,
          icon: "fa-solid fa-spinner fa-spin",
          text: "Starting…",
          href: "start.js",
        }]
      }

      if (running.update) {
        return [{
          default: true,
          icon: "fa-solid fa-spinner fa-spin",
          text: "Updating…",
          href: "update.js",
        }]
      }

      if (running.reset) {
        return [{
          default: true,
          icon: "fa-solid fa-spinner fa-spin",
          text: "Resetting…",
          href: "reset.js",
        }]
      }

      // ── Idle — full menu ──────────────────────────────────
      return [{
        default: true,
        icon: "fa-solid fa-power-off",
        text: "Start",
        href: "start.js",
      }, {
        icon: "fa-solid fa-rotate",
        text: "Update",
        href: "update.js",
      }, {
        icon: "fa-solid fa-wrench",
        text: "Fix",
        href: "fix.js",
      }, {
        icon: "fa-solid fa-plug",
        text: "Re-install",
        href: "install.js",
      }, {
        icon: "fa-solid fa-trash-can",
        text: "Reset",
        href: "reset.js",
        confirm: "This will delete the Python environment. Are you sure?",
      }]
    }

    // ── Not installed ─────────────────────────────────────────
    return [{
      default: true,
      icon: "fa-solid fa-plug",
      text: "Install",
      href: "install.js",
    }]
  }
}
