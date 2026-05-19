# CloakBrowser Integration Plan for OASIS

This document outlines the strategy for replacing standard Playwright with **CloakBrowser** in the OASIS UX Simulation App. CloakBrowser is a stealth Chromium binary with source-level C++ fingerprint patches that passes bot detection systems (like Cloudflare Turnstile and reCAPTCHA v3) out of the box.

By integrating CloakBrowser, OASIS will be able to scrape and simulate user behavior on heavily protected websites without being blocked, and without relying on third-party CAPTCHA solving services like 2Captcha for most sites.

---

## 1. Architecture Changes

Currently, OASIS uses standard Playwright in two places:
1. `captcha_guard.py` (Pre-flight check and session extraction)
2. `runner.py` (Mode 2: Browser-Action Simulation)

**The Plan:** We will replace `async_playwright().chromium.launch()` with `cloakbrowser.launch_async()` in both modules.

### 1.1 `captcha_guard.py` Refactor
The current guard attempts to detect CAPTCHAs and click checkboxes or use 2Captcha. With CloakBrowser, the browser itself is stealthy, so many CAPTCHAs (like Cloudflare Turnstile non-interactive challenges) will auto-resolve.

**Changes:**
- Import `launch_async` from `cloakbrowser`.
- Remove the complex `add_init_script` block that attempts to hide `webdriver` and `plugins` (CloakBrowser handles this at the C++ level).
- Enable `humanize=True` to ensure any required clicks (like the Turnstile "I am human" checkbox) use human-like Bézier curves and timing, passing behavioral checks.
- Keep the 2Captcha fallback as a last resort, but expect it to be used much less frequently.
- Keep the session export (`ctx.storage_state()`) so the cleared session can be handed off to the scraper and Mode 2.

### 1.2 `runner.py` (Mode 2) Refactor
Mode 2 simulates personas navigating the live website. If the site has behavioral bot detection (like FingerprintJS or deviceandbrowserinfo.com), standard Playwright will be flagged.

**Changes:**
- Replace Playwright launch with `cloakbrowser.launch_async()`.
- Enable `humanize=True` so that when personas click elements or scroll, the actions look human.
- Inject the `storage_state` from the `captcha_guard` pre-flight check so the personas inherit the cleared session and don't face CAPTCHAs again.
- **Fingerprint Strategy:** Use a deterministic seed (`--fingerprint=seed`) based on the persona ID or run ID. This ensures that if a persona opens multiple pages, their fingerprint remains consistent, making them look like a returning visitor rather than a bot rotating identities.

---

## 2. Configuration & Environment

CloakBrowser supports proxies and geo-location spoofing, which are crucial for enterprise scraping.

**New Environment Variables to add to `.env` and `config.py`:**
- `CLOAKBROWSER_PROXY`: Optional proxy string (e.g., `http://user:pass@proxy:8080` or `socks5://...`).
- `CLOAKBROWSER_GEOIP`: Boolean (default `false`). If true, CloakBrowser will auto-detect timezone and locale from the proxy exit IP.
- `CLOAKBROWSER_HUMANIZE`: Boolean (default `true`). Enables human-like mouse/keyboard behavior.

---

## 3. Installation & Deployment

CloakBrowser is a drop-in replacement, but it requires downloading a ~200MB custom Chromium binary on first run.

**Changes to `requirements.txt`:**
- Add `cloakbrowser` (and optionally `cloakbrowser[geoip]` if we want the auto-timezone feature).

**Changes to Dockerfile / CI:**
- To prevent the binary from downloading at runtime (which slows down the first scan), we should pre-download it during the Docker build process.
- Add the following to the Dockerfile:
  ```dockerfile
  RUN pip install cloakbrowser[geoip] && python -m cloakbrowser install
  ```
- **Linux Font Dependencies:** As noted in the CloakBrowser docs, minimal Linux environments need specific fonts to pass Kasada/Akamai canvas checks. We must add these to the Dockerfile `apt-get install` step:
  ```dockerfile
  RUN apt-get update && apt-get install -y fonts-noto-color-emoji fonts-freefont-ttf fonts-unifont fonts-ipafont-gothic fonts-wqy-zenhei fonts-tlwg-loma-otf
  ```

---

## 4. Implementation Steps

1. **Update Dependencies:** Add `cloakbrowser[geoip]` to `ux_sim_app/requirements.txt`.
2. **Update Config:** Add the new proxy and stealth settings to `ux_sim_app/core/config.py`.
3. **Refactor `captcha_guard.py`:** Swap Playwright for `cloakbrowser.launch_async(humanize=True, proxy=...)`. Remove legacy JS stealth scripts.
4. **Refactor `runner.py`:** Swap Playwright for `cloakbrowser.launch_async(humanize=True, proxy=...)` in `_mode2_one`.
5. **Update Documentation:** Update `README.md` to explain the new stealth capabilities and the optional proxy environment variables.
6. **Test:** Run a scan against a known Cloudflare-protected site to verify the Turnstile challenge auto-resolves or passes with a humanized click.
