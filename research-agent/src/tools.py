"""
tools.py — ResearchAgent tool suite
Nguyên tắc ACI: tên rõ nghĩa, description đầy đủ, error là string, không crash.
"""
import os
import re
import json
from pathlib import Path
from datetime import datetime

# ── Mock data (thay bằng API thật trong production) ──────────────────────────
MOCK_SEARCH_DB = {
    "AI agents": [
        {"title": "Building Effective Agents — Anthropic",
         "url": "https://anthropic.com/engineering/building-effective-agents",
         "snippet": "Agents are systems where LLMs dynamically direct their own processes and tool usage."},
        {"title": "OpenAI Agents SDK",
         "url": "https://openai.com/agents",
         "snippet": "Agents plan, call tools, collaborate across specialists to complete multi-step work."},
        {"title": "What is an AI Agent? — IBM",
         "url": "https://ibm.com/topics/ai-agents",
         "snippet": "AI agents perceive their environment, make decisions, and take actions to achieve goals."},
    ],
    "prompt caching": [
        {"title": "Prompt Caching — Anthropic Docs",
         "url": "https://docs.anthropic.com/prompt-caching",
         "snippet": "Cache frequently used context to reduce latency and costs by up to 90%."},
    ],
    "multi-agent systems": [
        {"title": "How we built our multi-agent research system",
         "url": "https://anthropic.com/engineering/multi-agent-research-system",
         "snippet": "Parallelism and specialization are the two key advantages of multi-agent systems."},
        {"title": "Orchestrator-Workers Pattern",
         "url": "https://anthropic.com/patterns/orchestrator",
         "snippet": "A central LLM dynamically breaks tasks and delegates to worker LLMs."},
    ],
}

MOCK_PAGES = {
    "https://anthropic.com/engineering/building-effective-agents": """
Building Effective Agents

The most important distinction in agentic systems: workflows vs agents.
Workflows: LLMs follow predefined code paths.
Agents: LLMs dynamically direct their own processes and tool usage.

Five key patterns:
1. Prompt Chaining — sequential steps with quality gates
2. Routing — classify input, direct to specialist
3. Parallelization — run subtasks simultaneously
4. Orchestrator-Workers — dynamic task breakdown
5. Evaluator-Optimizer — generate, evaluate, refine loop

Key insight: Start simple. Add complexity only when it demonstrably improves outcomes.
""",
}

NOTES_DIR = Path(__file__).parent.parent / "notes"

# ── Tool definitions (JSON Schema for Claude) ─────────────────────────────
TOOL_DEFINITIONS = [
    {
        "name": "search_web",
        "description": """Search the web for information on a topic.

Use this when you need current information, facts, articles, or data about a topic.
Returns a list of search results with title, URL, and snippet.

When to use: Starting research on a new topic, finding sources.
When NOT to use: When you already have the URL (use fetch_page instead),
or when the information is already in context.

Tip: Be specific. 'Claude AI tool use 2024' is better than 'AI'.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Be specific and focused."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_page",
        "description": """Fetch and return the text content of a web page.

Use this when you have a specific URL and want to read its full content.
Returns the page text (truncated to 2000 chars if too long).

When to use: After search_web gives you a URL you want to read deeply.
When NOT to use: When you don't have a URL yet (use search_web first).""",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL starting with https://"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "summarize_text",
        "description": """Summarize a long text into key points.

Use this to condense lengthy content before saving or presenting.
Returns a concise bullet-point summary.

When to use: After fetching a long page, before saving notes.
When NOT to use: For short texts under 200 words (summarize inline instead).""",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize"
                },
                "max_words": {
                    "type": "integer",
                    "description": "Maximum words in summary. Default: 150",
                    "default": 150
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "save_note",
        "description": """Save a research note to disk for later reference.

Use this to persist important findings, summaries, or conclusions.
Notes are saved to the notes/ directory as markdown files.
Returns the absolute path of the saved file.

When to use: After researching a topic and finding key insights to keep.
When NOT to use: For temporary data (keep in context instead).""",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Note title (used as filename). Keep it short and descriptive."
                },
                "content": {
                    "type": "string",
                    "description": "Note content in markdown format"
                }
            },
            "required": ["title", "content"]
        }
    },
]

# ── Tool implementations ──────────────────────────────────────────────────
def search_web(query: str) -> str:
    """Search web (mock). Returns JSON string of results."""
    if not query or not query.strip():
        return "Error: query cannot be empty."

    # Poka-yoke: truncate overly long queries
    query = query.strip()[:200]

    # Match against mock database
    results = []
    query_lower = query.lower()
    for key, items in MOCK_SEARCH_DB.items():
        if any(word in query_lower for word in key.split()):
            results.extend(items)

    if not results:
        results = [{
            "title": f"Search results for: {query}",
            "url": "https://example.com/results",
            "snippet": f"Found 0 mock results for '{query}'. In production, this calls a real search API."
        }]

    output = {"query": query, "results": results[:3]}
    return json.dumps(output, ensure_ascii=False, indent=2)


def fetch_page(url: str) -> str:
    """Fetch page content (mock). Returns text content."""
    if not url.startswith("http"):
        return f"Error: Invalid URL '{url}'. URL must start with http:// or https://"

    content = MOCK_PAGES.get(url)
    if not content:
        content = f"[Mock page content for: {url}]\n\nThis URL is not in the mock database.\nIn production, this fetches real web content.\nKey information would appear here..."

    # Poka-yoke: truncate very long pages
    if len(content) > 2000:
        content = content[:2000] + "\n\n[Content truncated at 2000 chars. Use summarize_text if needed.]"

    return content.strip()


def summarize_text(text: str, max_words: int = 150) -> str:
    """Summarize text using simple extraction (mock). Real version calls Claude."""
    if not text or not text.strip():
        return "Error: text cannot be empty."

    if max_words < 10:
        return "Error: max_words must be at least 10."

    # Simple mock: take first N words
    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    truncated = " ".join(words[:max_words])
    return f"[Summary, {max_words} words]:\n{truncated}...\n\n(In production: Claude summarizes with full understanding)"


def save_note(title: str, content: str) -> str:
    """Save note to disk. Returns absolute path."""
    if not title or not title.strip():
        return "Error: title cannot be empty."
    if not content or not content.strip():
        return "Error: content cannot be empty."

    try:
        NOTES_DIR.mkdir(parents=True, exist_ok=True)

        # Poka-yoke: sanitize filename (no special chars)
        safe_name = re.sub(r'[^\w\s-]', '', title.strip())
        safe_name = re.sub(r'\s+', '-', safe_name).lower()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{safe_name}.md"

        # Absolute path (poka-yoke: never relative)
        filepath = NOTES_DIR / filename
        filepath.write_text(f"# {title}\n\n{content}", encoding="utf-8")

        return f"Note saved successfully.\nTitle: {title}\nPath: {filepath.absolute()}"

    except Exception as e:
        return f"Error saving note: {str(e)}"


# ── Tool dispatcher ───────────────────────────────────────────────────────
TOOL_MAP = {
    "search_web": search_web,
    "fetch_page": fetch_page,
    "summarize_text": summarize_text,
    "save_note": save_note,
}

def execute_tool(name: str, input_dict: dict) -> str:
    """Execute a tool by name. Always returns string — never raises."""
    if name not in TOOL_MAP:
        return f"Error: Unknown tool '{name}'. Available: {list(TOOL_MAP.keys())}"
    try:
        return str(TOOL_MAP[name](**input_dict))
    except Exception as e:
        return f"Tool '{name}' error: {str(e)}"


if __name__ == "__main__":
    # Quick test
    print("=== Tool Tests ===")
    print("\n[search_web]")
    print(search_web("AI agents"))
    print("\n[fetch_page]")
    print(fetch_page("https://anthropic.com/engineering/building-effective-agents"))
    print("\n[summarize_text]")
    print(summarize_text("Agents are systems where LLMs dynamically direct their own processes. " * 20, max_words=30))
    print("\n[save_note]")
    print(save_note("Test Note", "This is a test note from Bài 1."))
