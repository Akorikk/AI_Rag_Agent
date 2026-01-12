import os
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

from app.rag import rag_pipeline

load_dotenv()

# OpenAI / Azure OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -------------------------------
# Simple in-memory session memory
# -------------------------------
SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a session"""
    return SESSION_MEMORY.get(session_id, [])


def update_session_history(session_id: str, role: str, content: str):
    """Update session memory"""
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []
    SESSION_MEMORY[session_id].append({"role": role, "content": content})


# -------------------------------
# Agent decision prompt
# -------------------------------
DECISION_PROMPT = """
You are an AI agent.

Your task is to decide whether the user's question:
1. Can be answered using general knowledge, OR
2. Requires searching internal company documents.

Internal documents include:
- Company overview
- Leave policy
- Security policy

Respond ONLY in valid JSON:
{
  "decision": "direct" or "rag",
  "reason": "short explanation"
}
"""


def decide_route(query: str) -> str:
    """Decide whether to use RAG or direct LLM answer"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DECISION_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0
    )

    decision_json = response.choices[0].message.content.lower()

    if "rag" in decision_json:
        return "rag"
    return "direct"


# -------------------------------
# Tool: RAG search
# -------------------------------
def rag_tool(query: str):
    """Tool that retrieves context from documents"""
    context, sources = rag_pipeline.retrieve(query)
    return context, sources


# -------------------------------
# Main agent entry point
# -------------------------------
def run_agent(query: str, session_id: str = "default") -> Dict:
    """
    Main agent function
    """
    # Initialize RAG index once
    if rag_pipeline.index is None:
        rag_pipeline.build_index()

    # Store user query in memory
    update_session_history(session_id, "user", query)

    # Decide route
    route = decide_route(query)

    # Get conversation history
    history = get_session_history(session_id)

    if route == "direct":
        # Direct LLM response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history + [
                {"role": "system", "content": "Answer clearly and concisely."}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        sources = []

    else:
        # RAG-based response
        context, sources = rag_tool(query)

        rag_prompt = f"""
You are an AI assistant answering questions using internal company documents.

Use ONLY the context below to answer.
If the answer is not found, say so clearly.

Context:
{context}

Question:
{query}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": rag_prompt}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

    # Store assistant response
    update_session_history(session_id, "assistant", answer)

    return {
        "answer": answer,
        "source": sources
    }
