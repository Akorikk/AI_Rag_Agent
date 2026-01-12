import os
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

from app.rag import rag_pipeline

load_dotenv()

# ================= OpenAI Client =================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ================= Session Memory =================
# Lightweight in-memory session memory (OK for demo)
SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    return SESSION_MEMORY.get(session_id, [])


def update_session_history(session_id: str, role: str, content: str):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []
    SESSION_MEMORY[session_id].append(
        {"role": role, "content": content}
    )


# ================= Decision Prompt =================
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
    """
    Decide whether to answer directly or use RAG
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DECISION_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0
    )

    content = response.choices[0].message.content.lower()

    if "rag" in content:
        return "rag"
    return "direct"


# ================= RAG Tool =================
def rag_tool(query: str):
    """
    Retrieve context from internal documents
    (Index is built lazily inside rag_pipeline)
    """
    context, sources = rag_pipeline.retrieve(query)
    return context, sources


# ================= Main Agent =================
def run_agent(query: str, session_id: str = "default") -> Dict:
    """
    Main agent entry point
    """

    # Store user query
    update_session_history(session_id, "user", query)

    # Decide route
    route = decide_route(query)

    # Get conversation history
    history = get_session_history(session_id)

    # -------- DIRECT ANSWER --------
    if route == "direct":
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history + [
                {"role": "system", "content": "Answer clearly and concisely."}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        sources = []

    # -------- RAG ANSWER --------
    else:
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

    # Store assistant reply
    update_session_history(session_id, "assistant", answer)

    return {
        "answer": answer,
        "source": sources
    }