from dotenv import load_dotenv
import os

load_dotenv()

# Test Anthropic
from anthropic import Anthropic
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=50,
    messages=[{"role": "user", "content": "Say 'Phase 2 ready' and nothing else."}]
)
print(response.content[0].text)

# Test LangGraph
from langgraph.graph import StateGraph
from typing import TypedDict
print("LangGraph imported successfully")

# Test LangSmith
print(f"LangSmith tracing: {os.environ.get('LANGCHAIN_TRACING_V2', 'not set')}")