#!/usr/bin/env python
from __future__ import annotations

import json
import math
import re
import os
from typing import Any, Dict

import httpx
from openai import OpenAI
from dotenv import load_dotenv

# ─────────────────── Load API key from .env ───────────────────────────────────
# .env file should contain:
# API_KEY=your-key-here
load_dotenv()
API_KEY = os.getenv("AALTO_OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Missing API_KEY. Please create a .env file with:\nAPI_KEY=your-key-here"
    )

# ─────────────────── Azure OpenAI setup ───────────────────────────────────────
DEPLOYMENT = "gpt-4o-2024-08-06"
BASE_URL = "https://aalto-openai-apigw.azure-api.net"
ENDPOINT_PATH = f"/v1/openai/deployments/{DEPLOYMENT}/chat/completions"


def _update_base_url(request: httpx.Request) -> None:
    if request.url.path == "/chat/completions":
        request.url = request.url.copy_with(path=ENDPOINT_PATH)


client = OpenAI(
    base_url=BASE_URL,
    api_key=False,  # using subscription key in headers
    default_headers={"Ocp-Apim-Subscription-Key": API_KEY},
    http_client=httpx.Client(event_hooks={"request": [_update_base_url]}),
)

# ─────────────────── Duolingo Design Task ─────────────────────────────────────
DESIGN_TASK = (
    "Duolingo is a language-learning app with millions of users worldwide. "
    "It offers bite-sized lessons, progress tracking, and gamified challenges to help people learn languages on the go. "
    "The brand is friendly, playful, and highly accessible, appealing to learners of all ages and backgrounds. "
    "It uses bright colors and a cheerful tone to make learning fun and motivating.\n\n"
    "Goal: Design a toggle for Duolingo's settings screen where users can enable or disable daily reminder notifications. "
    "Choose the toggle design that fits naturally into Duolingo's settings screen and supports a clear, inviting, and "
    "easy-to-use experience for managing daily reminders."
)

# ─────────────────── Evaluator Function ───────────────────────────────────────


def evaluate_toggle_design(params: Dict[str, Any]):
    """
    LLM-driven toggle design evaluator for Duolingo's Daily Reminder toggle.
    Returns a float between 0 and 1 (1 = most preferable, 0 = worst).
    """

    def _messages(p: Dict[str, Any]):
        # ---- Mapping tables (must match frontend) ----
        COLOR_SCHEMES = [
            {"index": 0, "name": "modern",     "semantics": "primary blue track on light gray; white thumb"},
            {"index": 1, "name": "nature",     "semantics": "green track on pale green; white thumb"},
            {"index": 2, "name": "sunset",     "semantics": "orange/amber track on warm cream; white thumb"},
            {"index": 3, "name": "ocean",      "semantics": "cyan/teal track on very light cyan; white thumb"},
            {"index": 4, "name": "dark",       "semantics": "dark gray track; white thumb"},
            {"index": 5, "name": "neon",       "semantics": "vivid purple track on light purple; white thumb"},
            {"index": 6, "name": "enterprise", "semantics": "neutral gray/blue track on light gray; white thumb"},
        ]
        VISUAL_STYLES = [
            {"index": 0, "name": "flat",     "thumb": "flat",     "track": "solid"},
            {"index": 1, "name": "shadow",   "thumb": "shadow",   "track": "gradient"},
            {"index": 2, "name": "glow",     "thumb": "glow",     "track": "glass"},
            {"index": 3, "name": "outline",  "thumb": "outline",  "track": "outlined"},
            {"index": 4, "name": "elevated", "thumb": "elevated", "track": "textured"},
        ]
        EASING = [
            {"index": 0, "name": "linear"},
            {"index": 1, "name": "ease"},
            {"index": 2, "name": "ease-in"},
            {"index": 3, "name": "ease-out"},
            {"index": 4, "name": "ease-in-out"},
        ]
        
        THUMB_SHAPES = [
            {"index": 0, "name": "circle",   "semantics": "Perfect round; friendly and brand-appropriate"},
            {"index": 1, "name": "square",   "semantics": "Sharp corners; precise but can feel rigid"},
            {"index": 2, "name": "diamond",  "semantics": "Rotated square; edgy look, careful with legibility"},
            {"index": 3, "name": "hexagon",  "semantics": "Six-sided; geometric, slightly technical vibe"},
            {"index": 4, "name": "star",     "semantics": "Five-pointed; playful but busy at small sizes"},
            {"index": 5, "name": "triangle", "semantics": "Equilateral; directional, can imply imbalance"},
            {"index": 6, "name": "teardrop", "semantics": "Pointed drop; dynamic, modern accent"},
            {"index": 7, "name": "bean",     "semantics": "Organic blob; whimsical and soft"},
        ]

        # Helper: decode from index OR name string (robust to either input form)
        def decode_choice(lst, value):
            try:
                i = int(value)
                i = max(0, min(i, len(lst) - 1))
                return lst[i]
            except Exception:
                # match by name if given a string like "modern" or "circle"
                val = str(value).strip().lower()
                for item in lst:
                    if item["name"].lower() == val:
                        return item
                # fallback to index 0
                return lst[0]

        sys_txt = (
            "You are a senior UI designer evaluating a toggle switch for Duolingo.\n"
            "Context — Design Task:\n"
            f"{DESIGN_TASK}\n\n"
            "Output format: return ONLY a JSON object: {\"score\": <number 0..1>}.\n\n"
            "How to interpret parameters (VERY IMPORTANT):\n"
            "• Some fields are integer INDICES that must be decoded before judging.\n"
            "  - colorScheme ∈ {0..6} → use this lookup table:\n"
            + "\n".join([f"    {c['index']}: {c['name']} — {c['semantics']}" for c in COLOR_SCHEMES]) + "\n"
            "  - visualStyle ∈ {0..4} → use this lookup table:\n"
            + "\n".join([f"    {v['index']}: {v['name']} (thumb: {v['thumb']}, track: {v['track']})" for v in VISUAL_STYLES]) + "\n"
            "  - easing ∈ {0..4} → use this lookup table:\n"
            + "\n".join([f"    {e['index']}: {e['name']}" for e in EASING]) + "\n"
            "  - thumbShape ∈ {0..7} → use this lookup table:\n"
            + "\n".join([f"    {s['index']}: {s['name']} — {s['semantics']}" for s in THUMB_SHAPES]) + "\n"
            "• If an index is out of range, clamp to the nearest valid index.\n"
            "• Non-index fields:\n"
            "  - duration (ms): 200–400 ideal; >600 feels sluggish; <150 abrupt.\n"
            "  - borderRadius (px): 10–35 modern; >45 may feel bloated.\n"
            "  - thumbRatio (0..1): 0.70–0.80 balances affordance and elegance.\n"
            "  - toggleSize (px): affects legibility and target size (140–200 typical in this task).\n\n"
            "Scoring guidelines (brand + UX):\n"
            "• Brand fit: playful, friendly, high-contrast; avoid muddy/low-contrast combos.\n"
            "• Clarity: on/off state obvious; good track/thumb contrast; readable outline.\n"
            "• Motion: easing consistent with mobile UI norms; duration per guidance above.\n"
            "• Effects: minimal ‘glow’; avoid gimmicks that hurt readability; outline acceptable if contrast is strong.\n"
            "Return ONLY JSON with the numeric score. "
            "IMPORTANT: Scores for different designs MUST be different unless the parameters are exactly identical."
        )

        # Provide both raw params and a decoded view so the model can reason reliably.
        decoded = {
            "colorScheme_decoded": decode_choice(COLOR_SCHEMES, p.get("colorScheme", 0)),
            "visualStyle_decoded": decode_choice(VISUAL_STYLES,  p.get("visualStyle", 0)),
            "easing_decoded":      decode_choice(EASING,        p.get("easing", 1)),
            "thumbShape_decoded":  decode_choice(THUMB_SHAPES,  p.get("thumbShape", 0)),
        }

        user_txt = (
            "Evaluate this toggle design for Duolingo. Decode any index parameters using the tables above, "
            "then score the design. Return ONLY JSON with the field \"score\".\n\n"
            "Raw parameters:\n"
            f"{json.dumps(p, ensure_ascii=False)}\n\n"
            "Decoded indices (for your reasoning only):\n"
            f"{json.dumps(decoded, ensure_ascii=False)}"
        )

        return [
            {"role": "system", "content": sys_txt},
            {"role": "user", "content": user_txt},
        ]

    def _call_once(p: Dict[str, Any]) -> float:
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=_messages(p),
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=250,
        )
        content = resp.choices[0].message.content.strip()

        try:
            data = json.loads(content)
            score = float(data.get("score"))
        except Exception:
            m = re.search(r"(-?\d+(\.\d+)?)", content)
            if not m:
                raise ValueError(f"Invalid LLM output: {content!r}")
            score = float(m.group(1))

        if math.isnan(score) or math.isinf(score):
            raise ValueError(f"Invalid score from LLM: {score!r}")

        return max(0.0, min(1.0, score))

    last_err = None
    for _ in range(3):
        try:
            return _call_once(params)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to get valid LLM score after 3 attempts: {last_err}")



# ─────────────────── Example usage ───────────────────────────────────────────
if __name__ == "__main__":
    example_params = {
        "duration": 300,
        "borderRadius": 24,
        "thumbRatio": 0.75,
        "thumbShape": 0,      # circle
        "visualStyle": 1,     # shadow
        "colorScheme": 0,     # modern
        "easing": 3,          # ease-out
        "toggleSize": 180,
    }
    print(evaluate_toggle_design(example_params))
