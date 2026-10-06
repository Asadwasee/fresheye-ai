"""
core/rag/ingest.py
===================
Builds (or rebuilds) the ChromaDB vector store from the knowledge base in
core/rag/knowledge/. This demonstrates an "advanced" ingestion pipeline:

  1. ADVANCED PARSERS
     - UnstructuredMarkdownLoader / TextLoader for .md files
     - PyPDFLoader for any .pdf files dropped into the knowledge folder
     - DirectoryLoader auto-discovers and routes files to the right loader
     This means you can drop in real product spec-sheets, PDFs of food
     safety regulations, etc. and they'll be ingested automatically.

  2. ADVANCED / HIERARCHICAL TEXT SPLITTING
     - Step 1: MarkdownHeaderTextSplitter splits each document along its
       "##" headers first, preserving section structure as metadata
       (this is far better than naive fixed-size chunking because chunks
       stay topically coherent - e.g. "Signs of Spoilage" never gets
       merged with an unrelated "Storage Advice" section).
     - Step 2: RecursiveCharacterTextSplitter further splits any
       still-oversized section using a language-aware separator hierarchy
       (paragraph -> sentence -> word), with overlap so context isn't
       lost at chunk boundaries.

  3. EMBEDDINGS
     - Open-source sentence-transformers model (all-MiniLM-L6-v2),
       runs 100% locally/offline via HuggingFaceEmbeddings.

  4. VECTOR DB
     - ChromaDB, persisted to disk (Config.CHROMA_PERSIST_DIR) so the
       index survives app restarts and doesn't need to be rebuilt on
       every boot.

Run standalone to (re)build the index:
    python -m core.rag.ingest
"""
import os
import glob

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import Config
from core.llm import get_embeddings


MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "doc_title"),
    ("##", "section"),
]


def _load_raw_documents(knowledge_dir: str) -> list[Document]:
    """Advanced, multi-format loading: routes each file extension to the
    parser best suited for it."""
    docs: list[Document] = []

    for path in glob.glob(os.path.join(knowledge_dir, "**", "*.md"), recursive=True):
        try:
            loader = UnstructuredMarkdownLoader(path)
            docs.extend(loader.load())
        except Exception:
            # Fall back to a plain text loader if unstructured has trouble
            docs.extend(TextLoader(path, encoding="utf-8").load())

    for path in glob.glob(os.path.join(knowledge_dir, "**", "*.pdf"), recursive=True):
        docs.extend(PyPDFLoader(path).load())

    for path in glob.glob(os.path.join(knowledge_dir, "**", "*.txt"), recursive=True):
        docs.extend(TextLoader(path, encoding="utf-8").load())

    return docs


def _split_documents(raw_docs: list[Document]) -> list[Document]:
    """Hierarchical splitting: markdown-header-aware first pass, then a
    recursive character splitter as a safety net for long sections."""
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_chunks: list[Document] = []
    for doc in raw_docs:
        source = doc.metadata.get("source", "unknown")
        try:
            header_chunks = md_splitter.split_text(doc.page_content)
        except Exception:
            header_chunks = [doc]

        for chunk in header_chunks:
            chunk.metadata["source"] = source
            if len(chunk.page_content) > Config.CHUNK_SIZE:
                sub_chunks = recursive_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

    return final_chunks


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Builds (or loads, if already persisted) the Chroma vector store."""
    embeddings = get_embeddings()

    already_built = os.path.isdir(Config.CHROMA_PERSIST_DIR) and os.listdir(Config.CHROMA_PERSIST_DIR)

    if already_built and not force_rebuild:
        print("[rag.ingest] Existing ChromaDB index found - loading it.")
        return Chroma(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR,
        )

    print(f"[rag.ingest] Building vector store from {Config.KNOWLEDGE_DIR} ...")
    raw_docs = _load_raw_documents(Config.KNOWLEDGE_DIR)
    print(f"[rag.ingest] Loaded {len(raw_docs)} raw documents.")

    chunks = _split_documents(raw_docs)
    print(f"[rag.ingest] Split into {len(chunks)} chunks.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=Config.CHROMA_COLLECTION_NAME,
        persist_directory=Config.CHROMA_PERSIST_DIR,
    )
    print("[rag.ingest] Vector store built and persisted.")
    return vectorstore


def get_vectorstore() -> Chroma:
    """Convenience accessor used by the rest of the app - loads existing
    index or builds it on first call."""
    return build_vectorstore(force_rebuild=False)


if __name__ == "__main__":
    build_vectorstore(force_rebuild=True)
