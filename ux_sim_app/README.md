# OASIS UX Simulation App

A no-code Gradio application for automated website UX auditing and persona-driven simulation testing.

## What it does

1. **Scrapes** any website automatically to extract content, navigation, and images
2. **Generates** a realistic customer focus group (AI personas) based on the website + your business context
3. **Runs three simulation modes:**
   - **Mode 1 – Content Simulation:** Personas react to your social media copy and marketing text
   - **Mode 2 – Browser Usability:** Personas navigate the live website and report usability issues
   - **Mode 3 – Visual Branding:** Personas analyse your brand imagery and give resonance scores
4. **Runs a full UX audit** (eyeson-style) with multi-viewport screenshots, heuristic checks, and GPT-4o vision critique across 6 UX dimensions
5. **Generates a rich HTML report** where persona feedback is highlighted and used to prioritise UX findings
6. **Delivers the report** via email, direct download, or an in-browser HTML editor

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Configure credentials
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Launch the app
python -m ux_sim_app
# → Open http://localhost:7860
```

## Usage (no coding required)

1. **Tab 1 – Setup & Personas:** Enter the website URL and optional business context → click "Scrape & Generate Personas"
2. **Tab 2 – Content to Test:** Paste your marketing copy (separate items with `---`)
3. **Tab 3 – Run Simulations:** Click "Run Simulations" to run all selected modes
4. **Tab 4 – UX Scan:** Click "Run UX Scan" for the full eyeson-style audit
5. **Tab 5 – Report:** Click "Generate Report" → download, email, or preview/edit the HTML report
6. **Tab 6 – Settings:** Override API keys and SMTP settings without editing files

## Environment Variables

| Variable | Required | Description |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `DEFAULT_MODEL` | No | Chat model (default: `gpt-4o-mini`) |
| `VISION_MODEL` | No | Vision model (default: `gpt-4o`) |
| `BROWSERBASE_API_KEY` | No | Browserbase fallback for Mode 2 |
| `SMTP_HOST/PORT/USER/PASS` | No | Email delivery settings |
| `MAX_BROWSER_SESSIONS` | No | Parallel browser sessions (default: 3) |

## Architecture

```
ux_sim_app/
├── core/
│   ├── config.py          # All settings and env vars
│   ├── llm.py             # Async OpenAI wrapper
│   ├── scraper.py         # Website scraper (httpx + BeautifulSoup)
│   └── personas.py        # AI persona generator
├── modes/
│   └── runner.py          # Mode 1, 2, 3 simulation engines
├── ux/
│   └── scanner.py         # UX scanner (screenshots + heuristics + AI critique)
├── report/
│   └── generator.py       # HTML report builder + email delivery
├── ui/
│   └── app.py             # Gradio interface
└── data/
    ├── reports/           # Generated HTML reports
    └── screenshots/       # Website screenshots
```
