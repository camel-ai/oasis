"""
BrowserbaseMCPClient – fallback browser driver for Mode 2.

Communicates with the Browserbase MCP server via JSON-RPC over HTTP/SSE.
Adapted from feat/browserbase-mcp-integration-7406594295122990252 with
improved session-id handling and error recovery.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx

log = logging.getLogger("simulation_backend.browserbase_client")

_MCP_BASE = "https://mcp.browserbase.com/mcp"


class BrowserbaseMCPClient:
    """
    Thin async client for the Browserbase MCP tool server.

    Usage::

        async with BrowserbaseMCPClient(api_key="...") as client:
            await client.navigate("https://example.com")
            result = await client.observe()
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=90.0)
        self.session_id: Optional[str] = None
        self._req_id = 1

    # ── Context manager ────────────────────────────────────────────────────

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ── Session lifecycle ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Perform MCP handshake and obtain a session ID."""
        url = f"{_MCP_BASE}?browserbaseApiKey={self.api_key}"
        payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "oasis-sim-backend", "version": "1.0"},
            },
            "id": self._next_id(),
        }
        resp = await self._http.post(
            url,
            json=payload,
            headers={"Accept": "application/json, text/event-stream"},
        )
        resp.raise_for_status()
        self.session_id = resp.headers.get("mcp-session-id")
        if not self.session_id:
            raise RuntimeError(
                "Browserbase MCP: no mcp-session-id returned during initialize"
            )
        log.debug("Browserbase MCP session initialised: %s", self.session_id)

    async def close(self) -> None:
        await self._http.aclose()

    # ── Tool calls ─────────────────────────────────────────────────────────

    async def start(self) -> Dict[str, Any]:
        return await self._call("start", {})

    async def navigate(self, url: str) -> Dict[str, Any]:
        return await self._call("navigate", {"url": url})

    async def observe(self) -> Dict[str, Any]:
        return await self._call("observe", {})

    async def act(self, action: str) -> Dict[str, Any]:
        return await self._call("act", {"action": action})

    async def screenshot(self) -> Dict[str, Any]:
        return await self._call("screenshot", {})

    # ── Internal ───────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        rid = self._req_id
        self._req_id += 1
        return rid

    async def _call(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.session_id:
            await self.initialize()

        url = f"{_MCP_BASE}?browserbaseApiKey={self.api_key}"
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": self._next_id(),
        }
        headers = {
            "mcp-session-id": self.session_id,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        try:
            resp = await self._http.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            text = resp.text
            # SSE format: "data: {json}"
            data_line = next(
                (l for l in text.split("\n") if l.startswith("data: ")), None
            )
            if data_line:
                return json.loads(data_line[6:])
            return {"raw": text}
        except httpx.HTTPStatusError as exc:
            log.error("Browserbase MCP HTTP error: %s", exc)
            return {"error": str(exc)}
        except Exception as exc:
            log.error("Browserbase MCP unexpected error: %s", exc)
            return {"error": str(exc)}
