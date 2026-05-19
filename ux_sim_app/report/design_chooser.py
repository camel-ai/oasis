"""
design_chooser.py
=================
Unified design-kit registry for the OASIS UX Simulation App slide reports.

Sources:
  1. OASIS Default — built-in teal/linen theme (no external dependency)
  2. frontend-slides presets (zarazhangrui/frontend-slides, MIT)
     12 curated visual styles: Bold Signal, Electric Studio, Creative Voltage,
     Dark Botanical, Notebook Tabs, Pastel Geometry, Split Pastel,
     Vintage Editorial, Neon Cyber, Terminal Green, Swiss Modern, Paper & Ink
  3. designkits.sh official kits (dk/* — Apache 2.0)
     6 official kits: Classic, Heritage, Modern Minimal, Bold Tech, Atelier, Neon Grid
  4. designkits.sh community kits (getdesign/* — MIT via VoltAgent/awesome-design-md)
     58 brand-inspired kits: Stripe, Vercel, Linear, Notion, Supabase, etc.

Usage:
    from ux_sim_app.report.design_chooser import list_kits, get_kit, KIT_CHOICES

    # Get all kit names for a Gradio dropdown
    choices = KIT_CHOICES  # list of (display_name, kit_id) tuples

    # Get a DesignKit by ID
    kit = get_kit("bold-signal")
    kit = get_kit("stripe")
    kit = get_kit("dk/classic")

    # Use with the Quarkdown generator
    from ux_sim_app.report.quarkdown_generator import render_html_quarkdown
    html, compiled_dir = render_html_quarkdown(report_data, kit=kit)
"""

from __future__ import annotations

from typing import Optional
from ux_sim_app.report.quarkdown_generator import DesignKit

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _gfont(*families: str) -> list[str]:
    """Build a Google Fonts @import URL for one or more font families."""
    query = "&family=".join(
        f.replace(" ", "+") for f in families
    )
    return [f"https://fonts.googleapis.com/css2?family={query}&display=swap"]


# ---------------------------------------------------------------------------
# 1. OASIS Default
# ---------------------------------------------------------------------------

_OASIS_DEFAULT = DesignKit(
    id="oasis-default",
    name="OASIS Default",
    source="oasis-default",
    vibe="Clean, professional, teal-accented",
    css_vars={
        "--oasis-bg":          "#FAFAF8",
        "--oasis-bg-dark":     "#1A2332",
        "--oasis-accent":      "#00897B",
        "--oasis-accent-2":    "#00695C",
        "--oasis-text":        "#1A1A1A",
        "--oasis-text-muted":  "#555555",
        "--oasis-border":      "#E0E0E0",
        "--oasis-card":        "#FFFFFF",
        "--oasis-font-display":"'Inter', 'Helvetica Neue', sans-serif",
        "--oasis-font-body":   "'Inter', 'Helvetica Neue', sans-serif",
        "--oasis-font-mono":   "'JetBrains Mono', monospace",
        "--oasis-radius":      "8px",
        "--oasis-shadow":      "0 2px 8px rgba(0,0,0,0.08)",
    },
    font_imports=_gfont("Inter:wght@300;400;500;600;700"),
)


# ---------------------------------------------------------------------------
# 2. frontend-slides presets (zarazhangrui/frontend-slides, MIT)
# ---------------------------------------------------------------------------

_FRONTEND_SLIDES: list[DesignKit] = [

    DesignKit(
        id="bold-signal",
        name="Bold Signal",
        source="frontend-slides",
        vibe="Confident, bold, modern, high-impact",
        css_vars={
            "--oasis-bg":          "#1a1a1a",
            "--oasis-bg-dark":     "#0d0d0d",
            "--oasis-accent":      "#FF5722",
            "--oasis-accent-2":    "#E64A19",
            "--oasis-text":        "#ffffff",
            "--oasis-text-muted":  "#aaaaaa",
            "--oasis-border":      "#333333",
            "--oasis-card":        "#FF5722",
            "--oasis-font-display":"'Archivo Black', sans-serif",
            "--oasis-font-body":   "'Space Grotesk', sans-serif",
            "--oasis-radius":      "4px",
            "--oasis-shadow":      "0 4px 16px rgba(255,87,34,0.25)",
        },
        font_imports=_gfont("Archivo+Black", "Space+Grotesk:wght@400;500"),
    ),

    DesignKit(
        id="electric-studio",
        name="Electric Studio",
        source="frontend-slides",
        vibe="Bold, clean, professional, high contrast",
        css_vars={
            "--oasis-bg":          "#ffffff",
            "--oasis-bg-dark":     "#0a0a0a",
            "--oasis-accent":      "#4361ee",
            "--oasis-accent-2":    "#3451d1",
            "--oasis-text":        "#0a0a0a",
            "--oasis-text-muted":  "#555555",
            "--oasis-border":      "#e0e0e0",
            "--oasis-card":        "#ffffff",
            "--oasis-font-display":"'Manrope', sans-serif",
            "--oasis-font-body":   "'Manrope', sans-serif",
            "--oasis-radius":      "0px",
            "--oasis-shadow":      "none",
        },
        font_imports=_gfont("Manrope:wght@400;500;800"),
    ),

    DesignKit(
        id="creative-voltage",
        name="Creative Voltage",
        source="frontend-slides",
        vibe="Bold, creative, energetic, retro-modern",
        css_vars={
            "--oasis-bg":          "#0066ff",
            "--oasis-bg-dark":     "#1a1a2e",
            "--oasis-accent":      "#d4ff00",
            "--oasis-accent-2":    "#b8e000",
            "--oasis-text":        "#ffffff",
            "--oasis-text-muted":  "#cccccc",
            "--oasis-border":      "#0055dd",
            "--oasis-card":        "#1a1a2e",
            "--oasis-font-display":"'Syne', sans-serif",
            "--oasis-font-body":   "'Space Mono', monospace",
            "--oasis-font-mono":   "'Space Mono', monospace",
            "--oasis-radius":      "2px",
            "--oasis-shadow":      "0 0 20px rgba(212,255,0,0.2)",
        },
        font_imports=_gfont("Syne:wght@700;800", "Space+Mono:wght@400;700"),
    ),

    DesignKit(
        id="dark-botanical",
        name="Dark Botanical",
        source="frontend-slides",
        vibe="Elegant, sophisticated, artistic, premium",
        css_vars={
            "--oasis-bg":          "#0f0f0f",
            "--oasis-bg-dark":     "#080808",
            "--oasis-accent":      "#d4a574",
            "--oasis-accent-2":    "#c9b896",
            "--oasis-text":        "#e8e4df",
            "--oasis-text-muted":  "#9a9590",
            "--oasis-border":      "#2a2a2a",
            "--oasis-card":        "#1a1a1a",
            "--oasis-font-display":"'Cormorant Garamond', serif",
            "--oasis-font-body":   "'IBM Plex Sans', sans-serif",
            "--oasis-radius":      "4px",
            "--oasis-shadow":      "0 4px 24px rgba(0,0,0,0.4)",
        },
        font_imports=_gfont("Cormorant+Garamond:wght@400;600", "IBM+Plex+Sans:wght@300;400"),
    ),

    DesignKit(
        id="notebook-tabs",
        name="Notebook Tabs",
        source="frontend-slides",
        vibe="Editorial, organized, elegant, tactile",
        css_vars={
            "--oasis-bg":          "#f8f6f1",
            "--oasis-bg-dark":     "#2d2d2d",
            "--oasis-accent":      "#98d4bb",
            "--oasis-accent-2":    "#c7b8ea",
            "--oasis-text":        "#1a1a1a",
            "--oasis-text-muted":  "#555555",
            "--oasis-border":      "#ddd8ce",
            "--oasis-card":        "#f8f6f1",
            "--oasis-font-display":"'Bodoni Moda', serif",
            "--oasis-font-body":   "'DM Sans', sans-serif",
            "--oasis-radius":      "4px",
            "--oasis-shadow":      "0 2px 12px rgba(0,0,0,0.1)",
        },
        font_imports=_gfont("Bodoni+Moda:wght@400;700", "DM+Sans:wght@400;500"),
    ),

    DesignKit(
        id="pastel-geometry",
        name="Pastel Geometry",
        source="frontend-slides",
        vibe="Friendly, organized, modern, approachable",
        css_vars={
            "--oasis-bg":          "#c8d9e6",
            "--oasis-bg-dark":     "#8fa8bc",
            "--oasis-accent":      "#9b8dc4",
            "--oasis-accent-2":    "#7c6aad",
            "--oasis-text":        "#1a1a1a",
            "--oasis-text-muted":  "#555555",
            "--oasis-border":      "#b0c4d4",
            "--oasis-card":        "#faf9f7",
            "--oasis-font-display":"'Plus Jakarta Sans', sans-serif",
            "--oasis-font-body":   "'Plus Jakarta Sans', sans-serif",
            "--oasis-radius":      "16px",
            "--oasis-shadow":      "0 4px 16px rgba(0,0,0,0.08)",
        },
        font_imports=_gfont("Plus+Jakarta+Sans:wght@400;500;700;800"),
    ),

    DesignKit(
        id="split-pastel",
        name="Split Pastel",
        source="frontend-slides",
        vibe="Playful, modern, friendly, creative",
        css_vars={
            "--oasis-bg":          "#f5e6dc",
            "--oasis-bg-dark":     "#e4dff0",
            "--oasis-accent":      "#c8f0d8",
            "--oasis-accent-2":    "#f0d4e0",
            "--oasis-text":        "#1a1a1a",
            "--oasis-text-muted":  "#555555",
            "--oasis-border":      "#e0d8d0",
            "--oasis-card":        "#ffffff",
            "--oasis-font-display":"'Outfit', sans-serif",
            "--oasis-font-body":   "'Outfit', sans-serif",
            "--oasis-radius":      "12px",
            "--oasis-shadow":      "0 2px 8px rgba(0,0,0,0.06)",
        },
        font_imports=_gfont("Outfit:wght@400;500;700;800"),
    ),

    DesignKit(
        id="vintage-editorial",
        name="Vintage Editorial",
        source="frontend-slides",
        vibe="Witty, confident, editorial, personality-driven",
        css_vars={
            "--oasis-bg":          "#f5f3ee",
            "--oasis-bg-dark":     "#1a1a1a",
            "--oasis-accent":      "#c41e3a",
            "--oasis-accent-2":    "#a01830",
            "--oasis-text":        "#1a1a1a",
            "--oasis-text-muted":  "#555555",
            "--oasis-border":      "#e8d4c0",
            "--oasis-card":        "#ffffff",
            "--oasis-font-display":"'Fraunces', serif",
            "--oasis-font-body":   "'Work Sans', sans-serif",
            "--oasis-radius":      "2px",
            "--oasis-shadow":      "0 2px 8px rgba(0,0,0,0.08)",
        },
        font_imports=_gfont("Fraunces:wght@700;900", "Work+Sans:wght@400;500"),
    ),

    DesignKit(
        id="neon-cyber",
        name="Neon Cyber",
        source="frontend-slides",
        vibe="Futuristic, techy, confident",
        css_vars={
            "--oasis-bg":          "#0a0f1c",
            "--oasis-bg-dark":     "#050810",
            "--oasis-accent":      "#00ffcc",
            "--oasis-accent-2":    "#ff00aa",
            "--oasis-text":        "#e0e8ff",
            "--oasis-text-muted":  "#8899bb",
            "--oasis-border":      "#1a2540",
            "--oasis-card":        "#0d1428",
            "--oasis-font-display":"'Syne', sans-serif",
            "--oasis-font-body":   "'Space Grotesk', sans-serif",
            "--oasis-font-mono":   "'Space Mono', monospace",
            "--oasis-radius":      "0px",
            "--oasis-shadow":      "0 0 20px rgba(0,255,204,0.15)",
        },
        font_imports=_gfont("Syne:wght@700;800", "Space+Grotesk:wght@400;500", "Space+Mono"),
    ),

    DesignKit(
        id="terminal-green",
        name="Terminal Green",
        source="frontend-slides",
        vibe="Developer-focused, hacker aesthetic",
        css_vars={
            "--oasis-bg":          "#0d1117",
            "--oasis-bg-dark":     "#010409",
            "--oasis-accent":      "#39d353",
            "--oasis-accent-2":    "#26a641",
            "--oasis-text":        "#c9d1d9",
            "--oasis-text-muted":  "#8b949e",
            "--oasis-border":      "#21262d",
            "--oasis-card":        "#161b22",
            "--oasis-font-display":"'JetBrains Mono', monospace",
            "--oasis-font-body":   "'JetBrains Mono', monospace",
            "--oasis-font-mono":   "'JetBrains Mono', monospace",
            "--oasis-radius":      "4px",
            "--oasis-shadow":      "0 0 12px rgba(57,211,83,0.1)",
        },
        font_imports=["https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap"],
    ),

    DesignKit(
        id="swiss-modern",
        name="Swiss Modern",
        source="frontend-slides",
        vibe="Clean, precise, Bauhaus-inspired",
        css_vars={
            "--oasis-bg":          "#ffffff",
            "--oasis-bg-dark":     "#000000",
            "--oasis-accent":      "#ff3300",
            "--oasis-accent-2":    "#cc2200",
            "--oasis-text":        "#000000",
            "--oasis-text-muted":  "#555555",
            "--oasis-border":      "#000000",
            "--oasis-card":        "#ffffff",
            "--oasis-font-display":"'Archivo', sans-serif",
            "--oasis-font-body":   "'Nunito', sans-serif",
            "--oasis-radius":      "0px",
            "--oasis-shadow":      "none",
        },
        font_imports=_gfont("Archivo:wght@800", "Nunito:wght@400"),
    ),

    DesignKit(
        id="paper-and-ink",
        name="Paper & Ink",
        source="frontend-slides",
        vibe="Editorial, literary, thoughtful",
        css_vars={
            "--oasis-bg":          "#faf9f7",
            "--oasis-bg-dark":     "#1a1a1a",
            "--oasis-accent":      "#c41e3a",
            "--oasis-accent-2":    "#a01830",
            "--oasis-text":        "#1a1a1a",
            "--oasis-text-muted":  "#555555",
            "--oasis-border":      "#d4c8b8",
            "--oasis-card":        "#ffffff",
            "--oasis-font-display":"'Cormorant Garamond', serif",
            "--oasis-font-body":   "'Source Serif 4', serif",
            "--oasis-radius":      "2px",
            "--oasis-shadow":      "0 1px 4px rgba(0,0,0,0.06)",
        },
        font_imports=_gfont("Cormorant+Garamond:wght@400;700", "Source+Serif+4:wght@400;600"),
    ),
]


# ---------------------------------------------------------------------------
# 3. designkits.sh official kits (dk/*)
# ---------------------------------------------------------------------------

_DK_OFFICIAL: list[DesignKit] = [

    DesignKit(
        id="dk/classic",
        name="DK Classic",
        source="designkits.sh",
        vibe="Deep slate canvas, terracotta accent, Inter UI — Slack-flavoured workspace",
        css_vars={
            "--oasis-bg":          "#1B2333",
            "--oasis-bg-dark":     "#0F1621",
            "--oasis-accent":      "#E07B54",
            "--oasis-accent-2":    "#C86840",
            "--oasis-text":        "#F0EDE8",
            "--oasis-text-muted":  "#8A9BB0",
            "--oasis-border":      "#2D3D52",
            "--oasis-card":        "#243040",
            "--oasis-font-display":"'Inter', sans-serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "6px",
            "--oasis-shadow":      "0 2px 8px rgba(0,0,0,0.3)",
        },
        font_imports=_gfont("Inter:wght@300;400;500;600;700"),
    ),

    DesignKit(
        id="dk/heritage",
        name="DK Heritage",
        source="designkits.sh",
        vibe="Journalistic high-contrast neutrals, deep red accent, serif headings",
        css_vars={
            "--oasis-bg":          "#F5F2EC",
            "--oasis-bg-dark":     "#1A1A1A",
            "--oasis-accent":      "#8B1A1A",
            "--oasis-accent-2":    "#6B1414",
            "--oasis-text":        "#1A1A1A",
            "--oasis-text-muted":  "#555555",
            "--oasis-border":      "#D4C8B8",
            "--oasis-card":        "#FFFFFF",
            "--oasis-font-display":"'Playfair Display', serif",
            "--oasis-font-body":   "'Source Serif 4', serif",
            "--oasis-radius":      "0px",
            "--oasis-shadow":      "0 1px 4px rgba(0,0,0,0.08)",
        },
        font_imports=_gfont("Playfair+Display:wght@400;700", "Source+Serif+4:wght@400;600"),
    ),

    DesignKit(
        id="dk/modern-minimal",
        name="DK Modern Minimal",
        source="designkits.sh",
        vibe="Clean sans-serif, generous whitespace, neutral grays, single blue accent",
        css_vars={
            "--oasis-bg":          "#FFFFFF",
            "--oasis-bg-dark":     "#F5F5F5",
            "--oasis-accent":      "#2563EB",
            "--oasis-accent-2":    "#1D4ED8",
            "--oasis-text":        "#111827",
            "--oasis-text-muted":  "#6B7280",
            "--oasis-border":      "#E5E7EB",
            "--oasis-card":        "#FFFFFF",
            "--oasis-font-display":"'Inter', sans-serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "8px",
            "--oasis-shadow":      "0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04)",
        },
        font_imports=_gfont("Inter:wght@300;400;500;600;700"),
    ),

    DesignKit(
        id="dk/bold-tech",
        name="DK Bold Tech",
        source="designkits.sh",
        vibe="Dark-mode-first, saturated purple/cyan accents, monospace code treatment",
        css_vars={
            "--oasis-bg":          "#0A0A0F",
            "--oasis-bg-dark":     "#050508",
            "--oasis-accent":      "#7C3AED",
            "--oasis-accent-2":    "#06B6D4",
            "--oasis-text":        "#F1F5F9",
            "--oasis-text-muted":  "#94A3B8",
            "--oasis-border":      "#1E1B4B",
            "--oasis-card":        "#0F0F1A",
            "--oasis-font-display":"'Space Grotesk', sans-serif",
            "--oasis-font-body":   "'Space Grotesk', sans-serif",
            "--oasis-font-mono":   "'JetBrains Mono', monospace",
            "--oasis-radius":      "6px",
            "--oasis-shadow":      "0 0 20px rgba(124,58,237,0.15)",
        },
        font_imports=_gfont("Space+Grotesk:wght@400;500;700", "JetBrains+Mono:wght@400;700"),
    ),

    DesignKit(
        id="dk/atelier",
        name="DK Atelier",
        source="designkits.sh",
        vibe="Warm hand-made feel — creamy off-whites, terracotta, sage, humanist serif",
        css_vars={
            "--oasis-bg":          "#FAF7F2",
            "--oasis-bg-dark":     "#2C2420",
            "--oasis-accent":      "#C4714A",
            "--oasis-accent-2":    "#7A9E7E",
            "--oasis-text":        "#2C2420",
            "--oasis-text-muted":  "#7A6E68",
            "--oasis-border":      "#E8DDD4",
            "--oasis-card":        "#FFFFFF",
            "--oasis-font-display":"'Lora', serif",
            "--oasis-font-body":   "'Nunito', sans-serif",
            "--oasis-radius":      "12px",
            "--oasis-shadow":      "0 2px 12px rgba(44,36,32,0.08)",
        },
        font_imports=_gfont("Lora:wght@400;600;700", "Nunito:wght@400;500;600"),
    ),

    DesignKit(
        id="dk/neon-grid",
        name="DK Neon Grid",
        source="designkits.sh",
        vibe="Cyberpunk UI — true-black, magenta + cyan neons, sharp corners, monospace",
        css_vars={
            "--oasis-bg":          "#000000",
            "--oasis-bg-dark":     "#000000",
            "--oasis-accent":      "#FF00FF",
            "--oasis-accent-2":    "#00FFFF",
            "--oasis-text":        "#FFFFFF",
            "--oasis-text-muted":  "#AAAAAA",
            "--oasis-border":      "#333333",
            "--oasis-card":        "#0A0A0A",
            "--oasis-font-display":"'Share Tech Mono', monospace",
            "--oasis-font-body":   "'Share Tech Mono', monospace",
            "--oasis-font-mono":   "'Share Tech Mono', monospace",
            "--oasis-radius":      "0px",
            "--oasis-shadow":      "0 0 10px #FF00FF, 0 0 20px #00FFFF",
        },
        font_imports=_gfont("Share+Tech+Mono"),
    ),
]


# ---------------------------------------------------------------------------
# 4. designkits.sh community kits (getdesign/* — brand-inspired)
#    CSS tokens derived from the brand descriptions in the designkits.sh catalog.
# ---------------------------------------------------------------------------

_DK_COMMUNITY: list[DesignKit] = [

    DesignKit(
        id="getdesign/stripe",
        name="Stripe",
        source="designkits.sh",
        vibe="Fintech gold standard — weight-300 headlines, deep navy, saturated purple accent",
        css_vars={
            "--oasis-bg":          "#FFFFFF",
            "--oasis-bg-dark":     "#0A2540",
            "--oasis-accent":      "#635BFF",
            "--oasis-accent-2":    "#0A2540",
            "--oasis-text":        "#0A2540",
            "--oasis-text-muted":  "#425466",
            "--oasis-border":      "#E6EBF1",
            "--oasis-card":        "#FFFFFF",
            "--oasis-font-display":"'Sohne', 'Inter', sans-serif",
            "--oasis-font-body":   "'Sohne', 'Inter', sans-serif",
            "--oasis-radius":      "6px",
            "--oasis-shadow":      "0 2px 5px rgba(10,37,64,0.08), 0 8px 24px rgba(10,37,64,0.06)",
        },
        font_imports=_gfont("Inter:wght@300;400;500;600"),
    ),

    DesignKit(
        id="getdesign/vercel",
        name="Vercel",
        source="designkits.sh",
        vibe="Engineered minimalism with Geist — aggressive negative tracking, shadow-as-border",
        css_vars={
            "--oasis-bg":          "#000000",
            "--oasis-bg-dark":     "#000000",
            "--oasis-accent":      "#FFFFFF",
            "--oasis-accent-2":    "#888888",
            "--oasis-text":        "#FFFFFF",
            "--oasis-text-muted":  "#888888",
            "--oasis-border":      "#333333",
            "--oasis-card":        "#111111",
            "--oasis-font-display":"'Geist', 'Inter', sans-serif",
            "--oasis-font-body":   "'Geist', 'Inter', sans-serif",
            "--oasis-font-mono":   "'Geist Mono', 'JetBrains Mono', monospace",
            "--oasis-radius":      "8px",
            "--oasis-shadow":      "0 0 0 1px rgba(255,255,255,0.1)",
        },
        font_imports=_gfont("Inter:wght@400;500;700"),
    ),

    DesignKit(
        id="getdesign/linear.app",
        name="Linear",
        source="designkits.sh",
        vibe="Dark-mode-native near-black, translucent surfaces, Inter Variable weight 510",
        css_vars={
            "--oasis-bg":          "#0F0F11",
            "--oasis-bg-dark":     "#080809",
            "--oasis-accent":      "#5E6AD2",
            "--oasis-accent-2":    "#4B56C0",
            "--oasis-text":        "#F2F2F2",
            "--oasis-text-muted":  "#8A8A9A",
            "--oasis-border":      "#1E1E24",
            "--oasis-card":        "#16161A",
            "--oasis-font-display":"'Inter', sans-serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "8px",
            "--oasis-shadow":      "0 1px 2px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.3)",
        },
        font_imports=_gfont("Inter:wght@300;400;500;600;700"),
    ),

    DesignKit(
        id="getdesign/notion",
        name="Notion",
        source="designkits.sh",
        vibe="Warm neutral minimalism, paper-like canvas, yellow-tinted grays, Notion Blue",
        css_vars={
            "--oasis-bg":          "#FFFFFF",
            "--oasis-bg-dark":     "#191919",
            "--oasis-accent":      "#2383E2",
            "--oasis-accent-2":    "#1A6EC7",
            "--oasis-text":        "#37352F",
            "--oasis-text-muted":  "#9B9A97",
            "--oasis-border":      "#E9E9E7",
            "--oasis-card":        "#FFFFFF",
            "--oasis-font-display":"'Inter', sans-serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "4px",
            "--oasis-shadow":      "rgba(15,15,15,0.05) 0 0 0 1px, rgba(15,15,15,0.1) 0 3px 6px",
        },
        font_imports=_gfont("Inter:wght@400;500;600;700"),
    ),

    DesignKit(
        id="getdesign/supabase",
        name="Supabase",
        source="designkits.sh",
        vibe="Dark-mode-native developer platform, emerald green accents, Circular font",
        css_vars={
            "--oasis-bg":          "#1C1C1C",
            "--oasis-bg-dark":     "#141414",
            "--oasis-accent":      "#3ECF8E",
            "--oasis-accent-2":    "#2EBF7E",
            "--oasis-text":        "#EDEDED",
            "--oasis-text-muted":  "#8B8B8B",
            "--oasis-border":      "#2A2A2A",
            "--oasis-card":        "#242424",
            "--oasis-font-display":"'Inter', sans-serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "8px",
            "--oasis-shadow":      "0 0 0 1px rgba(62,207,142,0.1), 0 4px 16px rgba(0,0,0,0.3)",
        },
        font_imports=_gfont("Inter:wght@400;500;600;700"),
    ),

    DesignKit(
        id="getdesign/apple",
        name="Apple",
        source="designkits.sh",
        vibe="Cinematic minimalism — pure black/light gray, SF Pro typography, Apple Blue",
        css_vars={
            "--oasis-bg":          "#FFFFFF",
            "--oasis-bg-dark":     "#000000",
            "--oasis-accent":      "#0071E3",
            "--oasis-accent-2":    "#0077ED",
            "--oasis-text":        "#1D1D1F",
            "--oasis-text-muted":  "#6E6E73",
            "--oasis-border":      "#D2D2D7",
            "--oasis-card":        "#FBFBFD",
            "--oasis-font-display":"'SF Pro Display', 'Inter', sans-serif",
            "--oasis-font-body":   "'SF Pro Text', 'Inter', sans-serif",
            "--oasis-radius":      "12px",
            "--oasis-shadow":      "0 2px 8px rgba(0,0,0,0.06)",
        },
        font_imports=_gfont("Inter:wght@300;400;500;600"),
    ),

    DesignKit(
        id="getdesign/framer",
        name="Framer",
        source="designkits.sh",
        vibe="Pure-black void canvas, extreme negative letter-spacing, electric blue accent",
        css_vars={
            "--oasis-bg":          "#000000",
            "--oasis-bg-dark":     "#000000",
            "--oasis-accent":      "#0055FF",
            "--oasis-accent-2":    "#0044CC",
            "--oasis-text":        "#FFFFFF",
            "--oasis-text-muted":  "#888888",
            "--oasis-border":      "#222222",
            "--oasis-card":        "#111111",
            "--oasis-font-display":"'Inter', sans-serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "999px",
            "--oasis-shadow":      "none",
        },
        font_imports=_gfont("Inter:wght@400;500;700;900"),
        extra_css=".reveal h1, .reveal h2 { letter-spacing: -0.05em; }",
    ),

    DesignKit(
        id="getdesign/spotify",
        name="Spotify",
        source="designkits.sh",
        vibe="Dark immersive music player, near-black cocoon, Spotify Green, pill geometry",
        css_vars={
            "--oasis-bg":          "#121212",
            "--oasis-bg-dark":     "#000000",
            "--oasis-accent":      "#1DB954",
            "--oasis-accent-2":    "#1AA34A",
            "--oasis-text":        "#FFFFFF",
            "--oasis-text-muted":  "#B3B3B3",
            "--oasis-border":      "#282828",
            "--oasis-card":        "#181818",
            "--oasis-font-display":"'Circular', 'Inter', sans-serif",
            "--oasis-font-body":   "'Circular', 'Inter', sans-serif",
            "--oasis-radius":      "500px",
            "--oasis-shadow":      "0 8px 24px rgba(0,0,0,0.5)",
        },
        font_imports=_gfont("Inter:wght@400;500;700;900"),
    ),

    DesignKit(
        id="getdesign/figma",
        name="Figma",
        source="designkits.sh",
        vibe="Strictly black-and-white interface, variable sans font, color only in gradients",
        css_vars={
            "--oasis-bg":          "#FFFFFF",
            "--oasis-bg-dark":     "#1E1E1E",
            "--oasis-accent":      "#0ACF83",
            "--oasis-accent-2":    "#F24E1E",
            "--oasis-text":        "#1E1E1E",
            "--oasis-text-muted":  "#757575",
            "--oasis-border":      "#E5E5E5",
            "--oasis-card":        "#FFFFFF",
            "--oasis-font-display":"'Inter', sans-serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "6px",
            "--oasis-shadow":      "0 1px 2px rgba(0,0,0,0.1)",
        },
        font_imports=_gfont("Inter:wght@300;400;500;600;700"),
    ),

    DesignKit(
        id="getdesign/ibm",
        name="IBM",
        source="designkits.sh",
        vibe="Carbon Design System — white canvas, IBM Blue, rectangular zero-radius, IBM Plex Sans",
        css_vars={
            "--oasis-bg":          "#FFFFFF",
            "--oasis-bg-dark":     "#161616",
            "--oasis-accent":      "#0F62FE",
            "--oasis-accent-2":    "#0043CE",
            "--oasis-text":        "#161616",
            "--oasis-text-muted":  "#6F6F6F",
            "--oasis-border":      "#E0E0E0",
            "--oasis-card":        "#F4F4F4",
            "--oasis-font-display":"'IBM Plex Sans', sans-serif",
            "--oasis-font-body":   "'IBM Plex Sans', sans-serif",
            "--oasis-font-mono":   "'IBM Plex Mono', monospace",
            "--oasis-radius":      "0px",
            "--oasis-shadow":      "none",
        },
        font_imports=_gfont("IBM+Plex+Sans:wght@300;400;500;600", "IBM+Plex+Mono:wght@400"),
    ),

    DesignKit(
        id="getdesign/airbnb",
        name="Airbnb",
        source="designkits.sh",
        vibe="Warm photography-forward marketplace, Rausch Red accent, rounded Nunito type",
        css_vars={
            "--oasis-bg":          "#FFFFFF",
            "--oasis-bg-dark":     "#222222",
            "--oasis-accent":      "#FF5A5F",
            "--oasis-accent-2":    "#E04347",
            "--oasis-text":        "#222222",
            "--oasis-text-muted":  "#717171",
            "--oasis-border":      "#DDDDDD",
            "--oasis-card":        "#FFFFFF",
            "--oasis-font-display":"'Nunito', sans-serif",
            "--oasis-font-body":   "'Nunito', sans-serif",
            "--oasis-radius":      "12px",
            "--oasis-shadow":      "0 2px 4px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.05)",
        },
        font_imports=_gfont("Nunito:wght@400;500;600;700;800"),
    ),

    DesignKit(
        id="getdesign/mistral.ai",
        name="Mistral AI",
        source="designkits.sh",
        vibe="Warm golden-amber universe, billboard-scale display, sharp architectural corners",
        css_vars={
            "--oasis-bg":          "#FFF8F0",
            "--oasis-bg-dark":     "#1A1200",
            "--oasis-accent":      "#F7A600",
            "--oasis-accent-2":    "#E09500",
            "--oasis-text":        "#1A1200",
            "--oasis-text-muted":  "#7A6A40",
            "--oasis-border":      "#F0E0C0",
            "--oasis-card":        "#FFFFFF",
            "--oasis-font-display":"'Syne', sans-serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "0px",
            "--oasis-shadow":      "4px 4px 0 #F7A600",
        },
        font_imports=_gfont("Syne:wght@400;700;800", "Inter:wght@400;500"),
    ),

    DesignKit(
        id="getdesign/claude",
        name="Claude (Anthropic)",
        source="designkits.sh",
        vibe="Warm parchment canvas, Fraunces-like serif headlines, terracotta accents",
        css_vars={
            "--oasis-bg":          "#FAF7F2",
            "--oasis-bg-dark":     "#2C2420",
            "--oasis-accent":      "#D97757",
            "--oasis-accent-2":    "#C06644",
            "--oasis-text":        "#2C2420",
            "--oasis-text-muted":  "#7A6E68",
            "--oasis-border":      "#E8DDD4",
            "--oasis-card":        "#FFFFFF",
            "--oasis-font-display":"'Fraunces', serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "8px",
            "--oasis-shadow":      "0 1px 4px rgba(44,36,32,0.08)",
        },
        font_imports=_gfont("Fraunces:wght@400;700", "Inter:wght@400;500"),
    ),

    DesignKit(
        id="getdesign/opencode.ai",
        name="OpenCode",
        source="designkits.sh",
        vibe="Terminal-native monospace-first aesthetic, warm near-black, Berkeley Mono",
        css_vars={
            "--oasis-bg":          "#1A1814",
            "--oasis-bg-dark":     "#0F0E0B",
            "--oasis-accent":      "#E8D5A3",
            "--oasis-accent-2":    "#D4C090",
            "--oasis-text":        "#E8D5A3",
            "--oasis-text-muted":  "#8A7E6A",
            "--oasis-border":      "#2A2820",
            "--oasis-card":        "#221F1A",
            "--oasis-font-display":"'JetBrains Mono', monospace",
            "--oasis-font-body":   "'JetBrains Mono', monospace",
            "--oasis-font-mono":   "'JetBrains Mono', monospace",
            "--oasis-radius":      "4px",
            "--oasis-shadow":      "none",
        },
        font_imports=_gfont("JetBrains+Mono:wght@400;700"),
    ),

    DesignKit(
        id="getdesign/raycast",
        name="Raycast",
        source="designkits.sh",
        vibe="Near-black blue-tinted canvas, macOS-native shadows, Raycast Red punctuation",
        css_vars={
            "--oasis-bg":          "#1A1B1E",
            "--oasis-bg-dark":     "#0E0F11",
            "--oasis-accent":      "#FF6363",
            "--oasis-accent-2":    "#E84E4E",
            "--oasis-text":        "#EEEEEF",
            "--oasis-text-muted":  "#8E8E93",
            "--oasis-border":      "#2A2B2E",
            "--oasis-card":        "#222326",
            "--oasis-font-display":"'Inter', sans-serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "8px",
            "--oasis-shadow":      "0 0 0 0.5px rgba(0,0,0,0.3), 0 2px 8px rgba(0,0,0,0.4)",
        },
        font_imports=_gfont("Inter:wght@400;500;600;700"),
    ),

    DesignKit(
        id="getdesign/resend",
        name="Resend",
        source="designkits.sh",
        vibe="Dark cinematic canvas, pure black void, three-font editorial hierarchy",
        css_vars={
            "--oasis-bg":          "#000000",
            "--oasis-bg-dark":     "#000000",
            "--oasis-accent":      "#5EEAD4",
            "--oasis-accent-2":    "#F97316",
            "--oasis-text":        "#FFFFFF",
            "--oasis-text-muted":  "#888888",
            "--oasis-border":      "#1A1A1A",
            "--oasis-card":        "#0A0A0A",
            "--oasis-font-display":"'Inter', sans-serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "6px",
            "--oasis-shadow":      "0 0 0 1px rgba(255,255,255,0.05)",
        },
        font_imports=_gfont("Inter:wght@300;400;500;600;700"),
    ),

    DesignKit(
        id="getdesign/sentry",
        name="Sentry",
        source="designkits.sh",
        vibe="Dark-mode-first developer tool, deep purple-black, warm purple + lime-green",
        css_vars={
            "--oasis-bg":          "#1A1625",
            "--oasis-bg-dark":     "#0F0C1A",
            "--oasis-accent":      "#A78BFA",
            "--oasis-accent-2":    "#84CC16",
            "--oasis-text":        "#F8F4FF",
            "--oasis-text-muted":  "#9B8EC4",
            "--oasis-border":      "#2D2540",
            "--oasis-card":        "#221D30",
            "--oasis-font-display":"'Inter', sans-serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "6px",
            "--oasis-shadow":      "0 2px 8px rgba(0,0,0,0.4)",
        },
        font_imports=_gfont("Inter:wght@400;500;600;700"),
    ),

    DesignKit(
        id="getdesign/mongodb",
        name="MongoDB",
        source="designkits.sh",
        vibe="Forest-dark teal-black canvas, neon MongoDB Green accents, serif hero headlines",
        css_vars={
            "--oasis-bg":          "#001E2B",
            "--oasis-bg-dark":     "#00111A",
            "--oasis-accent":      "#00ED64",
            "--oasis-accent-2":    "#00C853",
            "--oasis-text":        "#FFFFFF",
            "--oasis-text-muted":  "#89979B",
            "--oasis-border":      "#023430",
            "--oasis-card":        "#00293D",
            "--oasis-font-display":"'Source Serif 4', serif",
            "--oasis-font-body":   "'Inter', sans-serif",
            "--oasis-radius":      "8px",
            "--oasis-shadow":      "0 4px 16px rgba(0,0,0,0.4)",
        },
        font_imports=_gfont("Source+Serif+4:wght@400;700", "Inter:wght@400;500"),
    ),

    DesignKit(
        id="getdesign/nvidia",
        name="NVIDIA",
        source="designkits.sh",
        vibe="High-contrast technology-forward, signature electric green, industrial typography",
        css_vars={
            "--oasis-bg":          "#FFFFFF",
            "--oasis-bg-dark":     "#000000",
            "--oasis-accent":      "#76B900",
            "--oasis-accent-2":    "#5A8C00",
            "--oasis-text":        "#000000",
            "--oasis-text-muted":  "#555555",
            "--oasis-border":      "#CCCCCC",
            "--oasis-card":        "#F5F5F5",
            "--oasis-font-display":"'Barlow', sans-serif",
            "--oasis-font-body":   "'Barlow', sans-serif",
            "--oasis-radius":      "0px",
            "--oasis-shadow":      "none",
        },
        font_imports=_gfont("Barlow:wght@400;600;700;800"),
    ),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL_KITS: list[DesignKit] = (
    [_OASIS_DEFAULT]
    + _FRONTEND_SLIDES
    + _DK_OFFICIAL
    + _DK_COMMUNITY
)

_KIT_INDEX: dict[str, DesignKit] = {k.id: k for k in _ALL_KITS}


def list_kits() -> list[DesignKit]:
    """Return all available design kits."""
    return list(_ALL_KITS)


def get_kit(kit_id: str) -> DesignKit:
    """Return a DesignKit by ID, falling back to OASIS Default if not found."""
    return _KIT_INDEX.get(kit_id, _OASIS_DEFAULT)


def kits_by_source(source: str) -> list[DesignKit]:
    """Return all kits from a given source."""
    return [k for k in _ALL_KITS if k.source == source]


# ---------------------------------------------------------------------------
# Gradio dropdown choices
# ---------------------------------------------------------------------------

def _source_label(source: str) -> str:
    labels = {
        "oasis-default":  "OASIS",
        "frontend-slides": "Frontend Slides",
        "designkits.sh":   "designkits.sh",
    }
    return labels.get(source, source)


# List of (display_label, kit_id) tuples for Gradio gr.Dropdown
KIT_CHOICES: list[tuple[str, str]] = [
    (f"[{_source_label(k.source)}] {k.name} — {k.vibe[:50]}", k.id)
    for k in _ALL_KITS
]
