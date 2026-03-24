# Gradio 6 Override Knowledge Base

Lessons learned from fighting Gradio 6's CSS and JS in this project.
Use this before spending time guessing — every technique here has been battle-tested.

---

## 1. CSS Specificity War

### Why `!important` alone fails

Gradio 6 injects its base CSS via `<link rel="stylesheet">` in `<head>`, loaded early.
Your custom CSS (passed via `launch(css=CSS)`) is injected as a `<style>` tag AFTER.

**Source order rule for equal specificity + `!important`:**
Later stylesheet wins → your `!important` beats Gradio's IF specificity is equal or higher.

**But Svelte scoped classes add specificity:**
Gradio uses scoped classes like `.tab-wrapper.svelte-11gaq1` (specificity `0,2,0`).
Your `.tab-wrapper` alone is `0,1,0` — Gradio wins even with your `!important`.

**Fix:** Match or beat the svelte specificity:
```css
/* Beat .tab-wrapper.svelte-11gaq1 (0,2,0) by using the same hash */
.tab-wrapper.svelte-11gaq1 { height: auto !important; }

/* Or use element + class (0,1,1) — still less than (0,2,0) — NOT sufficient */
div.tab-wrapper { height: auto !important; }  /* ← loses */
```

The svelte hash for the Tabs component in Gradio 6.9.0 is: **`svelte-11gaq1`**
(File: `assets/Walkthrough-CTImcoqj.css`)
Check the hash after a Gradio upgrade: `grep -o 'svelte-[a-z0-9]*' Walkthrough-CTImcoqj.css | head -1`

---

## 2. Gradio Button CSS Override (Inside HTML Components)

### Problem
Gradio injects its stylesheet **after** the page loads via JS, overriding even `#id .class !important` rules in static `<style>` blocks inside `gr.HTML()`.

**Timeline:**
1. Page loads → Gradio's bundle JS runs
2. Svelte components mount
3. Gradio dynamically appends its `<link>` stylesheet to `<head>`
4. Your `<style>` block inside `gr.HTML()` is already in the DOM but now loses to the later `<link>`

### Fix: JS `style.setProperty(..., 'important')`
The only way to definitively win: set inline styles via JavaScript with `!important`:

```javascript
element.style.setProperty('font-size', '12px', 'important');
element.style.setProperty('padding', '4px 8px', 'important');
```

Inline styles with `!important` have the highest possible specificity and always win, regardless of when stylesheets are injected.

### Pattern used in this project (`_fitToolbar()`)
```javascript
function _fitToolbar() {
  var toolbar = document.getElementById('lc-toolbar');
  if (!toolbar) return;
  var cw = toolbar.clientWidth;
  if (!cw || cw < 10) return;  // not rendered yet — skip, ResizeObserver will retry

  var fs  = Math.max(10, Math.min(13, cw * 0.019));
  var px  = Math.max(6,  Math.min(11, cw * 0.015));
  var py  = Math.max(4,  Math.min(6,  cw * 0.009));

  toolbar.querySelectorAll('button').forEach(function(b) {
    b.style.setProperty('font-size',   fs.toFixed(1)+'px', 'important');
    b.style.setProperty('padding',     py.toFixed(1)+'px '+px.toFixed(1)+'px', 'important');
    b.style.setProperty('white-space', 'nowrap', 'important');
    b.style.setProperty('box-sizing',  'border-box', 'important');
  });
}

_fitToolbar();

// Re-run when container resizes (e.g. switching tabs, resizing window)
if (window.ResizeObserver) {
  new ResizeObserver(function() { _fitToolbar(); })
    .observe(document.getElementById('lc-toolbar') || document.body);
} else {
  window.addEventListener('resize', _fitToolbar);
}
```

**Key:** Use `ResizeObserver` on the container (not `window.resize`) because Gradio's 2-column layout changes container width without changing window width.
**Key:** If `clientWidth === 0`, the element is inside a hidden tab — skip and wait for ResizeObserver to fire when it becomes visible (Gradio uses `display:none` for inactive tabs).

---

## 3. Gradio Tab Overflow ("..." Button)

### How Gradio 6 tab overflow works
Source: `assets/Walkthrough.svelte_svelte_type_style_lang-DyYjqewg.js`

HTML structure:
```html
<div class="tab-wrapper svelte-11gaq1">
  <!-- Measurement container: position:absolute, 1px×1px, all tabs, no click handlers -->
  <div class="tab-container visually-hidden svelte-11gaq1" aria-hidden="true">
    <button class="svelte-11gaq1">Tab 1</button>
    <button class="svelte-11gaq1">Tab 2</button>
    <!-- ALL tabs rendered here for width measurement -->
  </div>

  <!-- Visible tablist: only contains visible_tabs (computed subset) -->
  <div class="tab-container svelte-11gaq1" role="tablist">
    <button role="tab" data-tab-id="...">Tab 1</button>
    <!-- Tabs that "fit" go here -->
  </div>

  <!-- "..." overflow button: SIBLING of tablist, NOT inside it -->
  <span class="overflow-menu svelte-11gaq1">
    <button><!-- 3-dots SVG --></button>
    <div class="overflow-dropdown svelte-11gaq1"><!-- hidden tabs --></div>
  </span>
</div>
```

The overflow detection function `handle_menu_overflow()`:
```javascript
const tab_nav_size = get(tab_nav_el).getBoundingClientRect();
let max_width = tab_nav_size.width;  // ← THIS decides how many tabs fit
// Loops from last tab to first, finds last tab whose right edge < max_width
// visible_tabs = tabs.slice(0, last_visible_index + 1)
// overflow_tabs = tabs.slice(last_visible_index + 1)
```

### The Fix: Override `getBoundingClientRect`
Make Gradio think the tablist is 9999px wide → all tabs fit → no overflow:

```javascript
function _fixTabOverflow() {
  var tabNav = document.querySelector('[role="tablist"]');
  if (!tabNav) { setTimeout(_fixTabOverflow, 400); return; }
  if (tabNav._lcTabFixed) return;
  tabNav._lcTabFixed = true;

  // Override getBoundingClientRect on the element instance
  // (shadows the prototype method — Gradio's code calls this directly)
  tabNav.getBoundingClientRect = function() {
    var r = Element.prototype.getBoundingClientRect.call(this);
    return {
      x: r.x, y: r.y, width: 9999, height: r.height,
      top: r.top, right: r.left + 9999, bottom: r.bottom, left: r.left,
      toJSON: function() { return this; }
    };
  };

  // Hide the overflow span (SIBLING of tablist, not child)
  var wrapper = tabNav.closest('.tab-wrapper');
  if (wrapper) {
    var span = wrapper.querySelector('span');
    if (span) span.style.setProperty('display', 'none', 'important');
  }

  // Trigger Gradio's re-computation
  window.dispatchEvent(new Event('resize'));
}
setTimeout(_fixTabOverflow, 800);  // Wait for Gradio Svelte app to mount
```

Then add CSS so the now-full tablist wraps and centers:
```css
/* Allow tab-wrapper to grow beyond its fixed height */
.tab-wrapper.svelte-11gaq1 {
    height: auto !important;
    padding-bottom: 0 !important;
    flex-wrap: wrap !important;
}
/* Tablist wraps + centers */
div[role="tablist"] {
    height: auto !important;
    overflow: visible !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
}
/* Hide overflow span via correct selector (sibling, not child of tablist) */
.tab-wrapper.svelte-11gaq1 > span,
.tab-wrapper > span:last-child {
    display: none !important;
}
/* Remove Gradio's ::after border line (we add border-bottom on the tablist itself) */
.tab-container.svelte-11gaq1::after {
    display: none !important;
}
```

**Common mistake:** Trying to hide the "..." button with `div[role="tablist"] > button:not([role="tab"])`.
This NEVER works because the "..." is in a `<span>` that is a **sibling** of `div[role="tablist"]`, not inside it.

---

## 4. Where to Inject JS

### `launch(js=...)` — runs on page load, global scope
```python
app.launch(js=MY_JS, css=MY_CSS, ...)
```
- Injected as `<script>` tag in `document.head`
- Runs once when the SPA loads (before Gradio's Svelte components mount)
- Use `setTimeout(..., 800)` or retry loops to wait for components
- Best for: global event listeners, DOM patches, tab fixes

### `.then(fn=None, js=...)` — runs after a Gradio event
```python
component.change(fn).then(fn=None, js=INIT_JS)
```
- Runs as a JS function after the Python callback completes
- Good for: initializing canvas/custom widgets after an image is loaded
- The JS string must be a valid function expression: `"() => { ... }"`

---

## 5. Gradio CSS File Locations (v6.9.0)

```
app/env/Lib/site-packages/gradio/templates/frontend/assets/
├── index-JtiO_h80.css          # Main Gradio CSS bundle (38KB)
├── index-Bq2njFOY.js           # Main Gradio JS bundle (255KB)
├── Walkthrough-CTImcoqj.css    # Tabs + Walkthrough component CSS  ← tab styles here
├── Walkthrough.svelte_svelte_type_style_lang-DyYjqewg.js  ← Tabs JS logic here
└── Index-BPZgH1F4.js           # Walkthrough/Stepper component
```

To find tab-related CSS after a Gradio upgrade:
```bash
grep -rl "tab-wrapper" app/env/Lib/site-packages/gradio/templates/frontend/assets/
```

To find the svelte hash:
```bash
grep -o 'svelte-[a-z0-9]*' Walkthrough-CTImcoqj.css | head -1
```

---

## 6. Gradio Hidden Tab Behavior

Inactive tabs use `display:none !important` (class `.hidden.svelte-11gaq1`).
All tab content IS rendered to DOM at startup, but hidden.

Consequences:
- `element.clientWidth === 0` when inside a hidden tab
- `element.getBoundingClientRect()` returns zeros
- Canvas `getContext('2d')` works but dimensions are 0
- `ResizeObserver` fires when a tab becomes visible (size changes from 0 → real size)

**Pattern:** Use `if (!cw || cw < 10) return;` in resize handlers.
ResizeObserver will re-fire when the tab opens. Don't try to force-initialize in a hidden state.

---

## 7. Python f-string CSS Escaping

When building HTML/CSS strings with Python f-strings, CSS `{}` must be escaped:
```python
# Wrong — Python tries to interpolate {color}
style = f"<style>.btn {{ color: {my_color}; }}</style>"

# Correct — double braces escape to literal braces in f-strings
style = f"<style>.btn {{ color: {my_color}; }}</style>"
#                    ^^                     ^^
```

All `{` and `}` in the CSS that are NOT template variables must be doubled: `{{` and `}}`.

---

## 8. Persisting Dynamic CSS Across Gradio Re-renders

### Problem
`element.style.setProperty(...)` sets inline styles on a DOM element **inside** a `gr.HTML`
component. When Gradio re-renders that component (e.g. a new image is loaded and Python
returns new HTML), the entire DOM subtree is **replaced**. All inline styles are wiped.

### Symptom
- Lock works on first load
- After uploading a second image or any state change that triggers `gr.HTML` update, the lock is gone
- Intermittent — depends on whether Gradio re-rendered the component

### Fix: inject a `<style>` tag into `<head>` instead of inline styles
`<head>` is **never** touched by Gradio component re-renders. CSS injected there persists forever.

```javascript
function _lockWrap() {
  var sid = '__lc_wrap_h';   // unique ID so we can replace, not stack
  var s = document.getElementById(sid);
  if (!s) { s = document.createElement('style'); s.id = sid; document.head.appendChild(s); }
  s.textContent = '#lc-crop-wrap{'
    + 'height:'     + maxH + 'px!important;'
    + 'max-height:' + maxH + 'px!important;'
    + 'min-height:' + maxH + 'px!important;'
    + 'overflow:auto!important}';
}
```

**Why this beats `style.setProperty` with `'important'`:**
- Inline styles (even with `!important`) live on the DOM element — gone when the element is replaced
- A `<style>` tag in `<head>` is a global CSS rule — survives any component re-render
- Setting `s.textContent` replaces the previous rule instantly (no stale values)

**Use this pattern whenever** you need a dynamic CSS value (computed at JS runtime) that must
survive Gradio component re-renders.

---

## 9. Quick Cheat Sheet

| Problem | Solution |
|---------|----------|
| CSS not overriding Gradio button style | JS `element.style.setProperty('prop', 'val', 'important')` |
| Dynamic CSS lost after Gradio re-render | Inject `<style id="uid">` into `<head>` instead of inline style |
| Buttons wrong size on resize | `ResizeObserver` on container element (not `window.resize`) |
| Gradio "..." tab overflow button shows | Override `getBoundingClientRect` on `[role="tablist"]` to return `width:9999` |
| "..." button selector not working | It's `.tab-wrapper > span`, NOT inside `div[role="tablist"]` |
| CSS rule not winning despite `!important` | Match Gradio's svelte hash: `.tab-wrapper.svelte-11gaq1` |
| Tab content not rendering (width=0) | Element is in hidden tab; use `ResizeObserver` and skip if `clientWidth < 10` |
| Gradio CSS file to inspect | `assets/Walkthrough-CTImcoqj.css` for tabs |
| Gradio tab JS logic | `assets/Walkthrough.svelte_svelte_type_style_lang-DyYjqewg.js` |
