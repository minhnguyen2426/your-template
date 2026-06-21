"""
agent.py — ResearchAgent: single proactive agent (DeepSeek backend)
Dùng requests gọi DeepSeek REST API (OpenAI-compatible endpoint).
"""
import os
import sys
import json
import requests
from dotenv import load_dotenv
from src.tools import TOOL_DEFINITIONS, execute_tool  # Bài 1

load_dotenv()

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

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


def _call_deepseek(messages: list) -> dict:
    """Call DeepSeek API. Returns the response dict."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 4096,
        "tools": TOOL_DEFINITIONS,
        "messages": messages,
    }
    resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── Core agentic loop ─────────────────────────────────────────────────────
def run_research_agent(topic: str, max_iterations: int = 10) -> str:
    """
    Run the research agent on a topic.
    Returns the final research report as a string.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Please research this topic thoroughly: {topic}"},
    ]

    print(f"\n🔍 ResearchAgent starting: '{topic}'")
    print("─" * 50)

    for iteration in range(max_iterations):
        # ① Call DeepSeek
        data = _call_deepseek(messages)
        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice["finish_reason"]

        # ② Append assistant message to history
        assistant_msg = {"role": "assistant", "content": message.get("content")}
        tool_calls = message.get("tool_calls")
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # ③ Check stop reason
        if finish_reason == "stop":
            final = message.get("content") or "[Agent returned no text]"
            print(f"\n✅ Agent completed in {iteration + 1} iteration(s)")
            return final

        # ④ Execute all tool calls; each result is a separate "tool" message
        if tool_calls:
            for tc in tool_calls:
                name = tc["function"]["name"]
                input_dict = json.loads(tc["function"]["arguments"])

                print(f"  → Tool: {name}({list(input_dict.keys())})")
                result = execute_tool(name, input_dict)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

    return f"[Agent stopped after {max_iterations} iterations without completing]"


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI agents and multi-agent systems"

    report = run_research_agent(topic)
    print("\n" + "═" * 50)
    print(report)
