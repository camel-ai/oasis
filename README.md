<div align="center">
  <a href="https://github.com/Greene-ctrl/aux">
    <img src="assets/banner.png" alt="AUX Banner">
  </a>
</div>

<br>

<div align="center">

<h1> AUX: Usability Testing Suite
</h1>

**AI-Powered User Experience Testing & Social Simulation Platform**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Playwright](https://img.shields.io/badge/Playwright-1.44+-green.svg)](https://playwright.dev/python/)
[![Gradio](https://img.shields.io/badge/Gradio-4.20+-orange.svg)](https://gradio.app/)

</div>

<br>

<p align="left">
<strong>AUX (Automated User Experience)</strong> is an open-source usability testing suite that uses Large Language Models (LLMs) to simulate how diverse, realistic user personas interact with websites, content, and brand imagery. Built as an evolution of agentic simulation research, AUX brings automated user testing out of the lab and into practical UX research, marketing, and product design.
</p>

---

## ✨ What It Does

Instead of waiting weeks for human focus groups, **AUX** allows you to test your digital products against a synthetic panel of AI personas in minutes.

1. **Scrape & Synthesize:** Enter any URL. AUX bypasses bot-detection (Cloudflare, reCAPTCHA), scrapes the content, and generates a diverse panel of realistic user personas tailored to the site's target audience.
2. **Real-World Grounding:** Optionally pull live social signals from Reddit, Hacker News, GitHub, Bluesky, and TikTok to shape the personas' opinions based on what real people are saying *today*.
3. **Multi-Modal Simulation:**
   - **Mode 1 (Content):** Personas react to social media copy and messaging.
   - **Mode 2 (Browser):** Personas navigate the live website using headless browsers, recording their clicks, scrolls, and frustrations on video.
   - **Mode 3 (Visual):** Personas critique brand imagery and UI screenshots using Vision LLMs.
4. **Automated UX Reporting:** The system runs a heuristic accessibility scan, generates AI-driven HTML redesigns for identified issues, and compiles everything into a presentation-ready PDF slide deck powered by **Quarkdown**.

---

## 🏗️ Architecture & Pipeline

The application is built on a modern, asynchronous Python stack with a Gradio web interface.

### 1. The CAPTCHA Guard & Stealth Browsing
Standard headless browsers are immediately blocked by modern CDNs. AUX uses **CloakBrowser**, a custom-patched Chromium binary that alters canvas, WebGL, and CDP fingerprints at the C++ level. This allows the app to silently bypass Cloudflare Turnstile and reCAPTCHA v3 without relying on slow, paid solving services.

### 2. Parallel Simulation Engine
All LLM calls and browser sessions run on a persistent background `asyncio` event loop. Simulation modes (Content, Browser, Visual) execute concurrently. Within Mode 2, up to `MAX_BROWSER_SESSIONS` personas navigate the target website simultaneously, each using a deterministic browser fingerprint seed.

### 3. Quarkdown Slide Report Pipeline
The final UX report is generated using **Quarkdown** with reveal.js output. Users can choose from **38 design kits**, including presets from **frontend-slides** and brand kits from **designkits.sh** (Stripe, Vercel, Apple, etc.). The resulting HTML is captured by Playwright into a landscape PDF presentation.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Java 17 (for the Quarkdown slide engine)
- Ubuntu/Debian Linux or macOS (Windows users should use WSL2)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Greene-ctrl/aux.git
   cd aux
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r ux_sim_app/requirements.txt
   ```

3. **Install browser binaries:**
   ```bash
   playwright install chromium
   python -m cloakbrowser install
   ```

### Configuration

Copy the example environment file and add your API keys:

```bash
cp ux_sim_app/.env.example ux_sim_app/.env
```

At a minimum, you must provide an `OPENAI_API_KEY`.

### Running the App

Start the Gradio web server:

```bash
python -m ux_sim_app --port 7860
```

Open `http://localhost:7860` in your browser.

---

## 🧩 Open Source Projects & References

AUX builds upon and adapts several outstanding open-source projects.

### Browser Automation & Stealth

| Project | Role in AUX | License | Repository |
|---|---|---|---|
| **Playwright** | Headless browser automation for Mode 2 persona simulations, UX screenshots, and PDF export. | Apache 2.0 | [microsoft/playwright](https://github.com/microsoft/playwright) |
| **CloakBrowser** | Stealth Chromium binary that bypasses bot detection. | Proprietary (free tier) | [CloakHQ/CloakBrowser](https://github.com/CloakHQ/CloakBrowser) |
| **Vercel Agent Browser** | Inspired the stealth browser automation and visual testing approach. | MIT | [vercel-labs/agent-browser](https://github.com/vercel-labs/agent-browser) |

### Slide & Report Generation

| Project | Role in AUX | License | Repository |
|---|---|---|---|
| **Quarkdown** | Markdown-to-HTML slide rendering engine with reveal.js output. | MIT | [iamgio/quarkdown](https://github.com/iamgio/quarkdown) |
| **frontend-slides** | Curated visual styles and layout presets integrated into the design chooser. | MIT | [zarazhangrui/frontend-slides](https://github.com/zarazhangrui/frontend-slides) |
| **designkits.sh** | Brand-inspired design kits (Stripe, Vercel, Notion) for slide theming. | Apache 2.0 / MIT | [designkits.sh](https://designkits.sh) |
| **slides-in-markdown** | Inspired the initial approach to converting Markdown ASTs into slides. | MIT | [technopagan/slides-in-markdown](https://github.com/technopagan/slides-in-markdown) |

### Web UI & Core

| Project | Role in AUX | License | Repository |
|---|---|---|---|
| **Gradio** | Interactive web UI framework handling all tabs and state management. | Apache 2.0 | [gradio-app/gradio](https://github.com/gradio-app/gradio) |
| **httpx** | Async HTTP client used for scraping and API calls. | BSD 3-Clause | [encode/httpx](https://github.com/encode/httpx) |
| **Beautiful Soup 4** | HTML parsing and structured content extraction. | MIT | [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) |

### Inspiration & Related Work

| Project | Relationship |
|---|---|
| **Agent-Lens** | Core inspiration for the automated heuristic scanning and AI-driven redesign pipeline. | [reynoldw/Agent-Lens](https://github.com/reynoldw/Agent-Lens) |
| **CAMEL-AI / OASIS** | The original social simulation research framework that AUX extends for practical UX testing. | [camel-ai/oasis](https://github.com/camel-ai/oasis) |
| **last30days** | Inspired the Real World Data tab architecture for social signal gathering. | [mvanhorn/last30days-skill](https://github.com/mvanhorn/last30days-skill) |

---

## 🧪 Testing & Development

Run the Python pytest suite:
```bash
python -m pytest tests/
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 🖺 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
