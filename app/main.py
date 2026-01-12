from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import os

from app.agent import run_agent

# ================= FastAPI App =================
app = FastAPI(
    title="AI Agent with RAG",
    description="AI agent that answers questions using direct LLM or RAG over internal documents",
    version="1.0.0"
)

# ================= Schemas =================
class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"


class AskResponse(BaseModel):
    answer: str
    source: List[str]


# ================= API Endpoint =================
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Accepts a user query and optional session_id.
    The agent decides whether to answer directly or use RAG.
    """
    result = run_agent(
        query=request.query,
        session_id=request.session_id
    )

    return AskResponse(
        answer=result["answer"],
        source=result["source"]
    )


# ================= Health Check (IMPORTANT for Render) =================
@app.get("/")
def health():
    return {"status": "ok", "service": "AI RAG Agent"}


# ================= Entry Point =================
# NOTE:

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
