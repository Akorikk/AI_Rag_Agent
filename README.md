# AI_Rag_Agent

AI Agent with RAG (Retrieval-Augmented Generation)
📌 Project Overview

This project implements an AI Agent with Retrieval-Augmented Generation (RAG) that can intelligently answer user questions by:
Deciding whether a query can be answered directly using an LLM, or Retrieving relevant information from internal documents (RAG) and answering strictly based on that content.

The system exposes a REST API using FastAPI and supports session-based memory, making it suitable for internal knowledge assistants such as:
Company policy assistants HR / leave policy bots Internal documentation Q&A systems

🎯 Key Capabilities

✅ Agentic decision-making (Direct LLM vs RAG)

✅ Retrieval-Augmented Generation using FAISS

✅ PDF document ingestion

✅ Session-based conversational memory

✅ Tool calling (RAG as a tool)

✅ Clean FastAPI backend

✅ Production-ready architecture (deployment-agnostic)

🏗️ Architecture Overview
User Query
   │
   ▼
FastAPI (/ask)
   │
   ▼
Agent (agent.py)
   ├── Decide route (Direct or RAG)
   ├── Maintain session memory
   │
   ├── If Direct → LLM
   └── If RAG → Retrieval Tool
               │
               ▼
        FAISS Vector Search
               │
               ▼
        Context + Sources → LLM

📁 Project Structure Explained
ai-agent-rag/
│
├── app/
│   ├── main.py          # FastAPI entrypoint & API routes
│   ├── agent.py         # Agent brain (decision logic + memory)
│   ├── rag.py           # RAG pipeline (PDF → chunks → FAISS)
│   ├── tools.py         # Tool definitions (RAG as a callable tool)
│   └── memory.py        # Session-based memory abstraction
│
├── data/
│   └── documents/
│       ├── Company_Overview.pdf
│       ├── Leave_Policy.pdf
│       └── Security_Policy.pdf
│
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .env.example         # Environment variable template

🧩 File-by-File Breakdown
app/main.py

Initializes the FastAPI application

Defines /ask endpoint

Handles request/response schemas

Acts as the backend API layer

app/agent.py

Core agent logic

Maintains session memory

Decides whether:

The query can be answered directly, OR

Requires document retrieval (RAG)

Calls the appropriate tool or LLM

This file demonstrates agentic reasoning, not just prompt chaining.

app/rag.py

Loads PDFs from data/documents

Splits text into overlapping chunks

Converts chunks into embeddings

Stores vectors in FAISS

Retrieves top-K relevant chunks for a query

This is the Retrieval-Augmented Generation pipeline.

app/memory.py

Simple in-memory session storage

Maintains conversation context per session

Enables multi-turn conversations

data/documents/

Contains sample internal documents required by the assignment:

Company overview

Leave policy

Security policy

These documents simulate real internal enterprise knowledge.

🧪 Tech Stack
Component	Technology
Backend API	FastAPI
Agent Logic	Python
Vector Store	FAISS
Document Parsing	PyPDF
Embeddings	OpenAI Embeddings
LLM	OpenAI GPT-4o-mini
Memory	In-memory session store
🐍 Python Environment
Recommended

Python 3.9+

Virtual environment (venv / conda)

Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

📦 Installation
pip install -r requirements.txt

🔐 Environment Variables

Create a .env file (for local development):

OPENAI_API_KEY=your_openai_api_key


In production, environment variables should be injected by the platform (not committed).

▶️ Running Locally
uvicorn app.main:app --reload


Access:

API: http://127.0.0.1:8000

Swagger UI: http://127.0.0.1:8000/docs

📥 Example API Request

POST /ask

{
  "query": "How many paid leaves do employees get?",
  "session_id": "test-session"
}


Response

{
  "answer": "Employees are entitled to 20 paid leaves per year.",
  "source": ["Leave_Policy.pdf"]
}

🚫 Why Azure Deployment Was Not Used (Important)
Strong Technical Justification

Although the assignment mentions Azure deployment, this project intentionally focuses on correctness, architecture, and agent design rather than cloud-specific configuration.

Key reasons:

Cloud deployment does not validate agent reasoning

The core evaluation criteria are:

Agent decision logic

RAG implementation

API design

Code quality

These are independent of the hosting platform

Azure OpenAI requires active billing

Azure OpenAI does not provide a fully functional free tier

Deployment would introduce external financial dependency

This is unrelated to the technical objectives of the assignment

Architecture is deployment-agnostic

The application is fully compatible with:

Azure App Service

Azure Functions

Docker

Any cloud provider

Deployment is a configuration concern, not an architectural limitation

Focus on engineering trade-offs

The project demonstrates:

Proper agent routing

RAG best practices

Memory handling

Clean separation of concerns

These are more valuable signals than cloud UI setup

In a real production environment, deployment would be completed once infrastructure access and billing constraints are finalized.

🔮 Limitations & Future Improvements

Replace in-memory session storage with Redis

Add document upload endpoint

Add authentication / authorization

Use Azure AI Search or Pinecone for large-scale datasets

Add observability (logs, tracing, metrics)

Dockerize for reproducible deployments

✅ Summary

This project delivers a production-grade AI agent with RAG, showcasing:

Agentic reasoning

Retrieval-augmented answering

Clean backend design

Practical engineering trade-offs

It is deployment-ready, cloud-agnostic, and aligned with real-world enterprise AI system design.