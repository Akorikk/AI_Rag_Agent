from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from app.agent import run_agent


# FastAPI app initialization

app = FastAPI(
    title="AI Agent with RAG",
    description="AI agent that answers questions using direct LLM or RAG over internal documents",
    version="1.0.0"
)


# Request / Response schemas

class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"


class AskResponse(BaseModel):
    answer: str
    source: List[str]



# API endpoint

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


# Local development entrypoint

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
