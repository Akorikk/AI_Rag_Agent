import os
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ========== CONFIG ==========
DATA_DIR = "data/documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"

# OpenAI / Azure OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ========== HELPERS ==========
def load_pdfs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Load PDFs and return list of (document_name, full_text)
    """
    documents = []

    for file in os.listdir(data_dir):
        if file.lower().endswith(".pdf"):
            path = os.path.join(data_dir, file)
            reader = PdfReader(path)

            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            documents.append((file, text))

    return documents


def chunk_text(text: str) -> List[str]:
    """
    Split text into overlapping chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


# ========== RAG CLASS ==========
class RAGPipeline:
    def __init__(self):
        self.index = None
        self.text_chunks = []
        self.sources = []

    def build_index(self):
        """
        Load documents, chunk them, embed them, and store in FAISS
        """
        documents = load_pdfs(DATA_DIR)

        all_chunks = []
        all_sources = []

        for doc_name, text in documents:
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            all_sources.extend([doc_name] * len(chunks))

        embeddings = embed_texts(all_chunks)

        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))

        self.text_chunks = all_chunks
        self.sources = all_sources

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """
        Retrieve relevant chunks for a query
        """
        query_embedding = embed_texts([query])[0]
        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        retrieved_chunks = []
        retrieved_sources = set()

        for idx in indices[0]:
            retrieved_chunks.append(self.text_chunks[idx])
            retrieved_sources.add(self.sources[idx])

        context = "\n\n".join(retrieved_chunks)

        return context, list(retrieved_sources)


# ========== SINGLETON ==========
rag_pipeline = RAGPipeline()
