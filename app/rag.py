import os
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ========== CONFIG ==========
DATA_DIR = "data/documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Local embedding model (FREE, no API key)
EMBEDDING_DIMENSION = 384
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


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


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings locally using SentenceTransformers
    """
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings.astype("float32")


# ========== RAG CLASS ==========
class RAGPipeline:
    def __init__(self):
        self.index = None
        self.text_chunks: List[str] = []
        self.sources: List[str] = []
        self.built = False

    def build_index(self):
        """
        Load documents, chunk them, embed them, and store in FAISS
        """
        if self.built:
            return

        documents = load_pdfs(DATA_DIR)

        all_chunks = []
        all_sources = []

        for doc_name, text in documents:
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            all_sources.extend([doc_name] * len(chunks))

        embeddings = embed_texts(all_chunks)

        self.index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        self.index.add(embeddings)

        self.text_chunks = all_chunks
        self.sources = all_sources
        self.built = True

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """
        Retrieve relevant chunks for a query
        """
        if not self.built:
            self.build_index()

        query_embedding = embed_texts([query])
        distances, indices = self.index.search(query_embedding, top_k)

        retrieved_chunks = []
        retrieved_sources = set()

        for idx in indices[0]:
            if idx < len(self.text_chunks):
                retrieved_chunks.append(self.text_chunks[idx])
                retrieved_sources.add(self.sources[idx])

        context = "\n\n".join(retrieved_chunks)

        return context, list(retrieved_sources)


rag_pipeline = RAGPipeline()