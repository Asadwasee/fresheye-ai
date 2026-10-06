"""
core/rag/retriever.py
======================
An "advanced retriever" pipeline, composed of three layers so retrieval
quality is noticeably better than a single naive similarity search:

  1. BASE RETRIEVER - Chroma similarity search using Maximal Marginal
     Relevance (MMR), which balances relevance with diversity so we don't
     get 6 near-duplicate chunks about the same sentence.

  2. QUERY EXPANSION - MultiQueryRetriever uses the LLM to rewrite the
     user's/agent's query into several different phrasings and unions the
     retrieved results. This meaningfully helps recall when the
     classification label ("rottenbanana") doesn't lexically match the
     knowledge base wording ("spoiled banana").

  3. CONTEXTUAL COMPRESSION - EmbeddingsFilter drops any retrieved chunk
     whose similarity to the query falls below a threshold, so the LLM's
     context window isn't polluted with irrelevant chunks pulled in only
     because of query expansion.

get_advanced_retriever(llm) returns a single object with the standard
LangChain retriever interface (`.invoke(query)` / `.get_relevant_documents`).
"""
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

from config import Config
from core.llm import get_embeddings
from core.rag.ingest import get_vectorstore


def get_advanced_retriever(llm):
    vectorstore = get_vectorstore()

    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": Config.RETRIEVER_TOP_K,
            "fetch_k": Config.RETRIEVER_FETCH_K,
            "lambda_mult": 0.6,
        },
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    embeddings_filter = EmbeddingsFilter(
        embeddings=get_embeddings(),
        similarity_threshold=Config.RETRIEVER_SIMILARITY_THRESHOLD,
    )

    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=multi_query_retriever,
    )

    return compressed_retriever


def format_docs_for_prompt(docs) -> str:
    """Render retrieved chunks into a compact, citeable context block."""
    if not docs:
        return "No specific reference material was found for this query."
    blocks = []
    for i, d in enumerate(docs, 1):
        section = d.metadata.get("section", "")
        source = d.metadata.get("source", "knowledge base")
        label = f"[{i}] {section}".strip() if section else f"[{i}]"
        blocks.append(f"{label} (source: {source}):\n{d.page_content.strip()}")
    return "\n\n".join(blocks)
