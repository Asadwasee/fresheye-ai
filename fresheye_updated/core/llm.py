"""
core/llm.py
===========
Thin factory around the open-source LLM used across the RAG pipeline and
the agent. Two providers are supported out of the box:

  * "groq"   - hosted, very low-latency inference of OPEN-SOURCE weight
               models (Meta's Llama 3.x family). Needs an API key
               (see core/api_keys.py + .env). This is the default because
               it needs no local GPU and is production-friendly.
  * "ollama" - a 100% local, open-source model served by Ollama. No API
               key, no external calls - good for on-prem / offline demos.

Swapping providers is a one-line .env change (LLM_PROVIDER=...); nothing
else in the codebase needs to change because both return a LangChain
`BaseChatModel`.
"""
from functools import lru_cache

from config import Config
from core.api_keys import LLM_PROVIDER, get_groq_api_key, get_ollama_base_url, validate_llm_config


@lru_cache(maxsize=1)
def get_llm():
    """Returns a singleton LangChain chat-model instance for the
    configured open-source LLM provider."""
    ok, message = validate_llm_config()
    print(f"[llm] provider={LLM_PROVIDER} | {message}")

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        api_key = get_groq_api_key()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(see .env.example), or set LLM_PROVIDER=ollama to run a "
                "fully local open-source model instead."
            )
        return ChatGroq(
            api_key=api_key,
            model=Config.GROQ_MODEL,
            temperature=Config.LLM_TEMPERATURE,
        )

    elif LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=get_ollama_base_url(),
            model=Config.OLLAMA_MODEL,
            temperature=Config.LLM_TEMPERATURE,
        )

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


@lru_cache(maxsize=1)
def get_embeddings():
    """Open-source sentence-transformer embeddings, run 100% locally -
    no API key needed. Used to embed the knowledge base into ChromaDB
    and to embed queries at retrieval time."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
