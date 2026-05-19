# Changelog

All notable changes to the OASIS UX Simulation App are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — feat/simulation-backend-server

### Added

- **CAPTCHA Guard** (`captcha_guard.py`): Pre-flight browser check using CloakBrowser stealth Chromium. Detects and resolves Cloudflare Turnstile, reCAPTCHA v2/v3, and hCaptcha. Falls back to 2Captcha token injection as a last resort. Aborts the full analysis pipeline if a CAPTCHA cannot be solved within the configured timeout, preventing silent data quality failures.
- **CloakBrowser Integration** (`captcha_guard.py`, `runner.py`): Replaced standard Playwright browser launch with CloakBrowser `launch_context_async(humanize=True)` in both the CAPTCHA guard and Mode 2 persona simulations. Each persona receives a deterministic fingerprint seed (`abs(hash(persona.name)) % 90000 + 10000`) for consistent device identity across page visits.
- **CAPTCHA Session Handoff**: Cleared browser session cookies are saved to a temporary JSON file and passed from the CAPTCHA guard to the scraper (for subpage fetches) and to all Mode 2 persona browser contexts, ensuring the solved session is reused throughout the analysis.
- **Marpit PDF Report Pipeline** (`slide_generator.py`, `marpit_render.js`): Replaced legacy Python HTML string concatenation with a proper Marpit rendering engine. Pipeline: `SlideReportData → Markdown AST → Node.js Marpit render → self-contained HTML → Playwright PDF`. The OASIS theme is a CSS design-token system with seven slide classes (`cover`, `toc`, `divider`, `divider-light`, `issue`, `strength`, `back`).
- **AI Redesign Screenshots**: When "Generate AI Redesigns" is enabled, GPT-4o Vision generates an improved HTML mockup for each UX issue. The mockup is rendered in a headless Playwright browser and captured as a JPEG screenshot, which is embedded directly in the issue slide (replacing the previous unreliable `<iframe>` approach).
- **Real World Data Tab** (Tab 2, optional): Gathers live social signals from Reddit, Hacker News, GitHub, Bluesky, ScrapeCreators, Brave Search, and Perplexity Sonar in parallel. Results are scored by engagement and synthesized by an LLM into a briefing paragraph. The briefing is injected into the persona generation prompt so personas reflect current community sentiment. The tab is fully optional — skipping it does not affect any other step.
- **Helmholtz LLM Fallback Hierarchy** (`llm.py`): All LLM calls now route through a single `llm.chat()` function with a 5-model fallback chain: Qwen3.6-35B → Qwen3.6-27B → Qwen3.5-35B (Helmholtz Blablador) → GPT-4o → GPT-4o-mini (OpenAI). Retryable errors (429, 500, 501, 502, 503, 504, proxy errors, timeouts) trigger automatic model cycling.
- **Persistent Background Event Loop** (`steps/shared.py`): Replaced the previous `_run()` pattern (which spawned a new thread and event loop per call) with a single persistent background `asyncio` event loop started at module load time.
- **Parallel Simulation Modes** (`steps/simulate.py`): Mode 1, Mode 2, and Mode 3 now execute concurrently via `asyncio.gather()`, reducing total simulation wall-clock time.
- **NotebookLM Integration** (`integrations/notebooklm.py`): Optional integration with Google NotebookLM. Authenticates via `notebooklm login` CLI (opens a real browser for Google OAuth), saves the session to `~/.notebooklm/storage_state.json`, and queries the notebook for UX best practices across 15 categories. Falls back to the built-in knowledge base silently if not configured.
- **Playwright Video Recording**: Mode 2 browser sessions record video of each persona's navigation. Recordings are accessible in the Session Recordings tab.
- **Cookie Banner Dismissal**: Mode 2 automatically detects and dismisses cookie consent banners before persona navigation begins.
- **Multi-Provider LLM Settings**: Supports OpenAI, OpenRouter, and any custom OpenAI-compatible endpoint. Separate selectors for text and vision models. Configured via `.env` or the Settings tab.
- **`app.py` Refactor** (`ui/steps/`): Split the 1,200-line `app.py` monolith into a `steps/` package: `shared.py`, `scrape.py`, `simulate.py`, `ux_scan.py`, `report.py`.
- **Pytest Suite** (`tests/test_pipeline.py`): 16 tests covering all key modules with mocked LLM responses.
- **Node.js Smoke Tests** (`marpit_smoke_test.js`): 14 assertions covering the Marpit theme CSS and rendering pipeline.
- **TTL Cache for Real World Data**: `gather_and_synthesize()` caches results for 30 minutes per topic to avoid redundant API calls during iterative testing.
- **Sentence-Boundary Slide Clamping** (`slide_generator.py`): `clamp_text()` now prefers sentence boundaries (`. ! ?`) before falling back to word boundaries, then hard cut.

### Changed

- **Scraper** (`scraper.py`): Rewrote to use structured semantic selectors (`_SECTION_SELECTORS`) for content extraction. Subpage fetches now propagate the CAPTCHA session cookie via Playwright/CloakBrowser instead of plain `httpx`.
- **UX Scanner** (`ux/scanner.py`): `_take_screenshots()` now uses CloakBrowser with the cleared CAPTCHA session. `scan_website()` accepts `storage_state_path` parameter.
- **Marpit Theme** (`marpit_render.js`): Unified CSS custom property type scale. All font sizes, colors, and spacing are defined as tokens. Black text throughout. Ghost watermark uses navy tint on linen backgrounds.
- **Slide Content Clamping**: All user-supplied strings are clamped at word/sentence boundaries to prevent overflow on the 960×540 slide canvas.
- **PDF Export**: Playwright `page.add_style_tag()` injects `-webkit-print-color-adjust: exact` before PDF capture, fixing washed-out background colors.

### Fixed

- `NotebookLMClient.from_browser()` (non-existent method) replaced with `from_storage()` pattern.
- `btn_nlm_auth.click()` output wiring corrected (was `[nlm_status_box, nlm_status_box]`, now `[nlm_status_box, nlm_notebook_label]`).
- CloakBrowser navigation timeout increased from 30s to 60s to handle sites with slow CDN redirect chains.
- Dead `generator.py` top-level import removed from `app.py`.
- `browser.close()` guarded with `if browser is not None` for CloakBrowser lifecycle compatibility.

---

## [Initial Release] — CAMEL-AI OASIS v1.0

The original OASIS social simulation framework from CAMEL-AI, supporting up to one million LLM agents on Twitter/Reddit-style platforms. See [camel-ai/oasis](https://github.com/camel-ai/oasis) for the full original changelog.
