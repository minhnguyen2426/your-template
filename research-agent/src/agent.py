"""
agent.py — ResearchAgent: single proactive agent
Dùng tools từ tools.py, chạy vòng lặp agentic đến khi task hoàn chỉnh.
"""
import os
import sys
import anthropic
from dotenv import load_dotenv
from src.tools import TOOL_DEFINITIONS, execute_tool  # Bài 1

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

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

# ── Core agentic loop ─────────────────────────────────────────────────────
def run_research_agent(topic: str, max_iterations: int = 10) -> str:
    """
    Run the research agent on a topic.
    Returns the final research report as a string.
    """
    messages = [{
        "role": "user",
        "content": f"Please research this topic thoroughly: {topic}"
    }]

    print(f"\n🔍 ResearchAgent starting: '{topic}'")
    print("─" * 50)

    for iteration in range(max_iterations):
        # ① Call Claude
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Haiku: fast + cheap for research
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # ② Append Claude's response to history
        messages.append({"role": "assistant", "content": response.content})

        # ③ Check stop reason
        if response.stop_reason == "end_turn":
            # Extract final text answer
            final = next(
                (b.text for b in response.content if b.type == "text"),
                "[Agent returned no text]"
            )
            print(f"\n✅ Agent completed in {iteration + 1} iteration(s)")
            return final

        # ④ Execute all tool calls (parallel support)
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"  → Tool: {block.name}({list(block.input.keys())})")
            result = execute_tool(block.name, block.input)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,  # bắt buộc — matching với request
                "content": result,
            })

        # ⑤ Send all results back in one message
        messages.append({"role": "user", "content": tool_results})

    # Guard: max iterations reached
    return f"[Agent stopped after {max_iterations} iterations without completing]"


# ── Message anatomy helper ────────────────────────────────────────────────
def explain_message_anatomy(messages: list) -> None:
    """Print message structure for learning — không dùng trong production."""
    print("\n📋 Message History Anatomy:")
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            print(f"  [{i}] {role}: text({len(content)} chars)")
        elif isinstance(content, list):
            types = [b.get("type", b.__class__.__name__) if isinstance(b, dict) else b.type for b in content]
            print(f"  [{i}] {role}: {types}")


if __name__ == "__main__":
    # Chạy từ command line: python src/agent.py "AI agents 2024"
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI agents and multi-agent systems"

    report = run_research_agent(topic)
    print("\n" + "═" * 50)
    print(report)
