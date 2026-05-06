---
updated_at: 2026-05-06
status: idle
branch: master
last_commit: fce338f
tests: n/a (no test suite)
app_version: 1.5.0
---

## State
- Pinokio launcher app, Python+Gradio 6, RTX 5090 / torch 2.7.0+cu128
- Auto-update on Start: git pull (smart.py) + yt-dlp upgrade
- All recent fixes pushed to origin/master, ready for users to pull on next launch

## What this session shipped
- Hybrid Smart Auto watermark tracking (SAM2 → validate → edge fallback)
- Tightened SAM2 validation: centroid 20px + mask size ≤3x check
- Audio preservation in watermark video output (`-map 1:a?`)
- Reuse SAM2-extracted JPEGs for edge tracker fallback (no re-decode)
- Transition guard — masks both old/new positions to kill 1-frame flash
- Version system (`VERSION` file) + auto-update on every Start
- Facebook Reels download fix: yt-dlp auto-update + 90s stall detection + browser UA
- See checkpoint: `.agents/sessions/2026-05-06-watermark-hybrid-and-autoupdate.md`

## Next action
idle — wait for user feedback after Restart/test

## Outstanding user-triggered actions
- Restart Pinokio launcher to trigger auto-update + test Facebook Reels download
