"""
agent.py — ResearchAgent: single proactive agent (Anthropic backend)
Dùng requests gọi Anthropic Messages API trực tiếp.
"""
import os
import sys
import json
import requests
from pathlib import Path
from src.tools import TOOL_DEFINITIONS, execute_tool

# Load .env manually (python-dotenv may not be installed)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"

# ── System prompt: proactive + completion criteria ────────────────────────
SYSTEM_PROMPT = """You are ResearchAgent — a proactive research assistant.

Your job: given a research topic, autonomously gather information and produce
a comprehensive, well-structured research summary.

## How you work
1. Search for the topic using search_web
2. Read the most promising sources using fetch_page
3. If content is long, summarize_text before keeping in context
4. Save your final summary using save_note
5. Return a well-formatted report to the user

## Completion criteria (self-check before stopping)
Before returning your final answer, verify:
- [ ] Searched at least 2 different angles of the topic
- [ ] Read at least 1 source in depth with fetch_page
- [ ] Saved a note with save_note
- [ ] Your final answer includes: Overview, Key Findings, Sources used

When all criteria are met, return your report. Do NOT wait for the user to ask
for more — be proactive and complete the full research cycle autonomously.

## Output format
Return your final report in this structure:
**Research Report: [Topic]**
**Overview:** [2-3 sentences]
**Key Findings:**
- Finding 1
- Finding 2
- Finding 3
**Sources consulted:** [list URLs]
**Note saved at:** [file path]"""


def _call_anthropic(messages: list) -> dict:
    """Call Anthropic Messages API. Returns the response dict."""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 4096,
        "system": SYSTEM_PROMPT,
        "tools": TOOL_DEFINITIONS,
        "messages": messages,
    }
    resp = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── Core agentic loop ─────────────────────────────────────────────────────
def run_research_agent(topic: str, max_iterations: int = 10) -> str:
    """
    Run the research agent on a topic.
    Returns the final research report as a string.
    """
    # Anthropic: system is a top-level param, messages start with user
    messages = [
        {"role": "user", "content": f"Please research this topic thoroughly: {topic}"},
    ]

    print(f"\n🔍 ResearchAgent starting: '{topic}'")
    print("─" * 50)

    for iteration in range(max_iterations):
        # ① Call Anthropic
        data = _call_anthropic(messages)
        stop_reason = data["stop_reason"]
        content_blocks = data["content"]  # list of blocks: text | tool_use

        # ② Append assistant message (Anthropic keeps content as block list)
        messages.append({"role": "assistant", "content": content_blocks})

        # ③ Check stop reason
        if stop_reason == "end_turn":
            final = next(
                (b["text"] for b in content_blocks if b["type"] == "text"),
                "[Agent returned no text]"
            )
            print(f"\n✅ Agent completed in {iteration + 1} iteration(s)")
            return final

        # ④ Execute all tool_use blocks; results go into a single user message
        if stop_reason == "tool_use":
            tool_results = []
            for block in content_blocks:
                if block["type"] != "tool_use":
                    continue
                name = block["name"]
                input_dict = block["input"]  # already a dict, not JSON string

                print(f"  → Tool: {name}({list(input_dict.keys())})")
                result = execute_tool(name, input_dict)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": result,
                })

            # Anthropic: all tool results in ONE user message
            messages.append({"role": "user", "content": tool_results})

    return f"[Agent stopped after {max_iterations} iterations without completing]"


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI agents and multi-agent systems"

    report = run_research_agent(topic)
    print("\n" + "═" * 50)
    print(report)
