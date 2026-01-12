import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pypdf import PdfReader

# ================= CONFIG =================
DATA_DIR = "data/documents"
CHUNK_SIZE = 400           # ↓ smaller chunks = less RAM
CHUNK_OVERLAP = 50
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # ✅ light & free

# Load embedding model lazily
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedding_model


# ================= HELPERS =================
def load_pdfs(data_dir: str) -> List[Tuple[str, str]]:
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(data_dir, file))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            documents.append((file, text))
    return documents


def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False)


# ================= RAG PIPELINE =================
class RAGPipeline:
    def __init__(self):
        self.index = None
        self.text_chunks = []
        self.sources = []
        self.is_built = False

    def build_index(self):
        if self.is_built:
            return

        documents = load_pdfs(DATA_DIR)

        all_chunks = []
        all_sources = []

        for doc_name, text in documents:
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            all_sources.extend([doc_name] * len(chunks))

        embeddings = embed_texts(all_chunks)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype("float32"))

        self.text_chunks = all_chunks
        self.sources = all_sources
        self.is_built = True

    def retrieve(self, query: str, top_k: int = 3):
        if not self.is_built:
            self.build_index()

        query_vec = embed_texts([query]).astype("float32")
        _, indices = self.index.search(query_vec, top_k)

        context = []
        sources = set()

        for idx in indices[0]:
            context.append(self.text_chunks[idx])
            sources.add(self.sources[idx])

        return "\n\n".join(context), list(sources)


# Singleton
rag_pipeline = RAGPipeline()
