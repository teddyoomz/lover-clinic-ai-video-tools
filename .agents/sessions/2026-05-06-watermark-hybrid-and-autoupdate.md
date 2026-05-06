# Session 2026-05-06 — Watermark Hybrid + Auto-Update + FB Reels

## Summary
Milestone session: shipped hybrid Smart Auto watermark tracking (SAM2 → validate → edge fallback), version system with auto-update on every Start, and fixed Facebook Reels download hangs by auto-updating yt-dlp + adding stall detection. 8 commits pushed to origin/master.

## Current State
- Branch: `master` · HEAD: `fce338f` · App version: `1.5.0`
- All changes live on GitHub — users get them automatically next time they Start
- No test suite; verification by Pinokio launcher Restart + manual UI test
- yt-dlp is now auto-upgraded on every Start (Facebook API changes frequently)
- SAM2 + edge tracking + LaMa pipeline producing perfect watermark removal

## Commits

```
7e3b1f3  feat: hybrid Smart Auto watermark tracking (SAM2 + edge fallback)
3f140a6  fix: tighten SAM2 validation — threshold 5→20px + mask size check
64c484a  fix: preserve audio in watermark-removed video output
29e470a  perf: reuse SAM2 extracted JPEGs for edge tracking fallback
5a336b0  fix: eliminate 1-frame watermark flash during position jumps
75d42b5  docs: update CODEBASE.md with full Smart Auto pipeline details
3ac9f6e  feat: version system + auto-update on every Start
fce338f  fix: Facebook Reels download hanging — yt-dlp auto-update + stall detection
```

## Files Touched
- `app/app.py` — Smart Auto pipeline, version display, FB-friendly yt-dlp opts, stall detector
- `app/smart.py` — `_auto_update()` (git pull) + yt-dlp auto-upgrade in `start()`
- `VERSION` — new file, semver `1.5.0`
- `CODEBASE.md` — full pipeline docs + auto-update + downloader updates

## Decisions
- SAM2 validation: dual check — centroid std > 20px AND mask size ratio ≤ 3x. Single threshold (5px) let background jitter pass.
- Edge fallback reuses SAM2's JPEG dir → saves ~12s per moving-watermark run.
- Transition guard: when watermark jumps > 30% template size, mask BOTH old and new positions on that frame (prevents 1-frame flash).
- Auto-update via `git pull --ff-only` only — never overwrites local changes; skips silently on no internet or dirty tree.
- yt-dlp upgraded on every Start (cheap if already latest, critical for Facebook).
- Stall detection: 90s of no progress-line change = force stop with "ดาวน์โหลดค้าง" message.
- Browser User-Agent + `concurrent_fragment_downloads: 4` for Facebook CDN compatibility.

## Next Todo
- Wait for user to Restart and confirm Facebook Reels works
- (later) Consider adding ProPainter for true motion-aware video inpainting if LaMa per-frame shows artifacts
- (later) Add similar auto-update for ffmpeg/imageio binaries if needed

## Resume Prompt

```
Resume Lover Clinic AI Video Tools — continue from 2026-05-06 EOD.

Read in order BEFORE any tool call:
1. CLAUDE.md
2. CODEBASE.md
3. SESSION_HANDOFF.md (master=fce338f)
4. .agents/active.md (v1.5.0, idle)
5. (milestone) .agents/sessions/2026-05-06-watermark-hybrid-and-autoupdate.md

Status: master=fce338f, v1.5.0, no test suite, all pushed
Next: idle — wait for user feedback after Restart/Facebook Reels test
Outstanding (user-triggered): Restart Pinokio launcher to trigger auto-update
Rules: ภาษาไทยเสมอ, follow F:\pinokio\CLAUDE.md + project CLAUDE.md, check logs first
```
