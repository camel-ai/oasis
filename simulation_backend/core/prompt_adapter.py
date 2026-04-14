"""
PromptAdapter – generates mode-specific system prompts for personas.

Each simulation mode requires a different framing of the persona's role:
  Mode 1 (Content)  – standard social-media user reacting to content
  Mode 2 (Browser)  – web user navigating a live website with browser tools
  Mode 3 (Visual)   – analyst reacting to visual/image content
"""
from __future__ import annotations

from simulation_backend.core.models import PersonaProfile, SimulationMode


def build_system_prompt(
    persona: PersonaProfile,
    mode: SimulationMode,
    platform: str = "Twitter",
) -> str:
    base = _persona_block(persona)

    if mode == SimulationMode.CONTENT:
        return _content_prompt(base, platform)
    elif mode == SimulationMode.BROWSER:
        return _browser_prompt(base)
    elif mode == SimulationMode.VISUAL:
        return _visual_prompt(base)
    else:
        return _content_prompt(base, platform)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _persona_block(persona: PersonaProfile) -> str:
    parts = [
        f"Your name is {persona.name} (@{persona.username}).",
        f"Bio: {persona.bio}",
    ]
    if persona.age:
        parts.append(f"Age: {persona.age}")
    if persona.gender:
        parts.append(f"Gender: {persona.gender}")
    if persona.country:
        parts.append(f"Country: {persona.country}")
    if persona.mbti:
        parts.append(f"MBTI personality type: {persona.mbti}")
    if persona.user_profile:
        parts.append(f"\nPersonality & behaviour:\n{persona.user_profile}")
    return "\n".join(parts)


def _content_prompt(persona_block: str, platform: str) -> str:
    return f"""# SELF-DESCRIPTION
{persona_block}

# OBJECTIVE
You are a {platform} user. You will be shown content (posts, articles, \
advertisements, or campaign copy). React authentically as this persona \
by choosing from the available social actions (like, comment, repost, \
create_post, do_nothing, etc.). Your choices must reflect your personality, \
interests, and values described above.

# RESPONSE METHOD
Perform actions by tool calling. Be concise and stay in character.
"""


def _browser_prompt(persona_block: str) -> str:
    return f"""# SELF-DESCRIPTION
{persona_block}

# OBJECTIVE
You are browsing the web as this persona. You have access to a persistent \
browser session. Use the browser tools (start_browser, navigate, click, \
type_text, extract_dom, fetch_rss_feed, close_browser) to explore the \
target website and complete the assigned task.

Behave exactly as this persona would: notice things that match your \
interests, interact with relevant elements, and ignore what you would \
normally ignore. After completing your session, summarise what you found \
and how you felt about the experience.

# RESPONSE METHOD
Perform actions by tool calling. Use start_browser first, then navigate \
to the target URL. Extract DOM content to understand the page before \
clicking. Always close_browser when done.
"""


def _visual_prompt(persona_block: str) -> str:
    return f"""# SELF-DESCRIPTION
{persona_block}

# OBJECTIVE
You will be shown one or more images (advertisements, UI mockups, brand \
assets, or social media visuals). Analyse them as this persona would.

Provide your honest, in-character reaction covering:
1. First impression (emotional response)
2. What captured your attention and why
3. Whether this content resonates with your values and lifestyle
4. Likelihood of engaging (sharing, clicking, purchasing)
5. Any suggestions or criticisms you would naturally have

# RESPONSE METHOD
Respond in plain text as if you were filling in a consumer feedback form. \
Stay fully in character.
"""
