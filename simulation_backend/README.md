# OASIS Multi-Mode Simulation Backend

A Python server-side application that wraps the [OASIS](https://github.com/camel-ai/oasis) persona simulation framework and exposes three switchable simulation modes through both a **Gradio testing interface** and a **REST API** compatible with the [UserSyncInterface](https://huggingface.co/spaces/AUXteam/UserSyncInterface) frontend.

---

## Architecture Overview

```
simulation_backend/
├── app.py                  ← FastAPI entry point + Gradio mount
├── __main__.py             ← python -m simulation_backend
├── requirements.txt
├── .env.example
│
├── core/
│   ├── settings.py         ← Pydantic-settings config (env vars / .env)
│   ├── models.py           ← Shared Pydantic models & schemas
│   ├── persona_manager.py  ← LLM-based persona generation + OASIS AgentGraph builder
│   ├── prompt_adapter.py   ← Mode-specific system prompt generation
│   ├── job_store.py        ← In-memory async job tracker
│   └── orchestrator.py     ← Routes jobs to the correct mode handler
│
├── modes/
│   ├── mode1_content.py    ← Mode 1: Social-media content feedback
│   ├── mode2_browser.py    ← Mode 2: Website interaction (Playwright + Browserbase MCP)
│   ├── browser_session.py  ← Playwright session pool + RSS anti-bot fallback
│   ├── browserbase_client.py ← Browserbase MCP JSON-RPC client
│   └── mode3_visual.py     ← Mode 3: Visual-input analysis (VLM)
│
├── api/
│   └── routers/
│       ├── personas.py     ← /api/v1/personas endpoints
│       └── simulations.py  ← /api/v1/simulations endpoints
│
├── gradio_ui/
│   └── interface.py        ← Three-tab Gradio Blocks app
│
└── data/
    └── focus_groups/       ← Persisted focus group JSON files
```

---

## The Three Simulation Modes

| Mode | Name | Description | Driver |
|------|------|-------------|--------|
| **1** | Content Simulation | Personas react to social media copy, links, or articles using standard OASIS social actions (like, comment, repost, etc.) | OpenAI-compatible LLM |
| **2** | Browser-Action Simulation | Personas navigate a live website step-by-step, driven by the LLM choosing browser actions | **Primary:** Playwright (local, parallel) · **Fallback:** Browserbase MCP |
| **3** | Visual Input Simulation | Personas analyse images (ads, mockups, brand assets) and provide structured in-character feedback | Vision-Language Model (GPT-4o or compatible) |

### Switching Modes

Set `parameters.mode` in the simulation request:

```json
{
  "focus_group_id": "...",
  "content_payload": "Check out our new eco-friendly collection!",
  "parameters": { "mode": 1 }
}
```

---

## Quick Start

### 1. Install dependencies

```bash
cd simulation_backend
pip install -r requirements.txt

# Mode 2 also requires Playwright browsers:
playwright install chromium
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set at minimum: OPENAI_API_KEY
```

### 3. Start the server

```bash
# From the repo root:
python -m simulation_backend

# Or with uvicorn directly:
uvicorn simulation_backend.app:app --host 0.0.0.0 --port 7860 --reload
```

The server starts on **http://localhost:7860**.

| Interface | URL |
|-----------|-----|
| Gradio UI | http://localhost:7860/gradio |
| REST API docs (Swagger) | http://localhost:7860/docs |
| REST API docs (ReDoc) | http://localhost:7860/redoc |

---

## REST API Reference

### Personas

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/personas/generate` | Generate a new focus group via LLM |
| `GET`  | `/api/v1/personas` | List all focus groups |
| `GET`  | `/api/v1/personas/{id}` | Get a focus group with full persona details |
| `DELETE` | `/api/v1/personas/{id}` | Delete a focus group |

**Generate request body:**
```json
{
  "business_description": "A sustainable fashion brand…",
  "customer_profile": "18-28 year olds interested in eco-friendly clothing",
  "num_personas": 5
}
```

### Simulations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/simulations` | Start a simulation (returns `job_id` immediately) |
| `GET`  | `/api/v1/simulations` | List all jobs |
| `GET`  | `/api/v1/simulations/{job_id}` | Poll job status & results |
| `DELETE` | `/api/v1/simulations/{job_id}` | Cancel a running job |

**Start simulation request body:**
```json
{
  "focus_group_id": "uuid",
  "content_type": "text",
  "content_payload": "Your content here",
  "parameters": {
    "mode": 1,
    "target_url": null,
    "image_urls": null,
    "use_browserbase_fallback": true,
    "max_rounds": 1
  }
}
```

**Poll response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "mode": 1,
  "progress_percentage": 100.0,
  "message": "Simulation complete.",
  "results": { ... }
}
```

---

## Persona Adaptation

Each persona is generated with a rich profile (demographics, MBTI, bio, narrative description). When a simulation starts, the `PromptAdapter` rewrites the persona's system prompt for the selected mode:

- **Mode 1:** Standard social-media user prompt, emphasising platform behaviour and social actions.
- **Mode 2:** Web-browsing user prompt, emphasising browser tool use and in-character navigation goals.
- **Mode 3:** Visual analyst prompt, emphasising subjective aesthetic and emotional reactions.

The `PersonaManager.build_agent_graph()` method converts any `FocusGroup` into an OASIS `AgentGraph`, enabling direct use of the OASIS simulation loop when the `oasis-sim` package is installed.

---

## Mode 2: Browser Parallelism

Mode 2 runs up to `MAX_BROWSER_SESSIONS` (default: 8) Playwright sessions concurrently. Each persona gets its own isolated browser context. If a page is blocked by Cloudflare or similar anti-bot measures, the session automatically attempts RSS feed discovery and falls back to serving article content from the feed.

If Playwright is unavailable or a session fails after retries, the request is routed to the **Browserbase MCP** client (requires `BROWSERBASE_API_KEY`).

---

## Frontend Integration

The API is designed to satisfy the `GradioService.ts` contract in the UserSyncInterface frontend. Set the backend URL in the frontend environment:

```
VITE_BACKEND_URL=http://localhost:7860
```

The frontend calls:
- `POST /api/v1/personas/generate` → to create focus groups
- `GET  /api/v1/personas` → to populate the focus group dropdown
- `POST /api/v1/simulations` → to start a simulation
- `GET  /api/v1/simulations/{job_id}` → to poll for results

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | API key for the LLM provider |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `DEFAULT_MODEL` | `gpt-4o-mini` | Model for Modes 1 & 2 |
| `VISION_MODEL` | `gpt-4o` | Model for Mode 3 (must support vision) |
| `BROWSERBASE_API_KEY` | *(optional)* | Enables Browserbase MCP fallback in Mode 2 |
| `MAX_BROWSER_SESSIONS` | `8` | Max parallel Playwright sessions |
| `OASIS_SEMAPHORE` | `64` | Max concurrent LLM calls (Modes 1 & 3) |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `7860` | Server port |
| `RELOAD` | `false` | Enable uvicorn hot-reload |
