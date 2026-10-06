"""
core/rag/chain.py
==================
The generative half of the RAG pipeline: takes (a) the YOLO classification
result and (b) retrieved knowledge-base context, and produces a structured,
professional freshness recommendation using the open-source LLM - built
with LangChain Expression Language (LCEL).

This module also implements short-term conversational memory for the chat
assistant, using the standard two-step "conversational RAG" pattern:

  1. CONTEXTUALIZE - if there's prior chat history, ask the LLM to rewrite
     the user's latest message into a fully standalone question (so "is it
     safe to eat?" becomes "is a rotten orange safe to eat?" using the
     previous turn). This standalone question is what's used to search the
     vector store - searching on the raw pronoun-laden message would
     retrieve poor/irrelevant results.
  2. ANSWER - the final answer is generated from the *original* message
     (so the reply still reads naturally), the retrieved context, AND the
     recent chat history, so the model can resolve references itself too.

Chat history is kept client-side (sent with each request) rather than
server-side sessions, keeping the backend stateless and easy to scale
horizontally - the frontend just replays the last few turns each time.

Two generation chains are exposed:
  * get_advice_chain()  -> turns a classification result into a
                            recommendation (eat / use soon / discard) with
                            reasoning, storage tips and an estimated
                            remaining shelf life. Always a single-turn,
                            standalone call - no history needed.
  * get_chat_chain()    -> general-purpose, history-aware RAG Q&A chain
                            for the chat assistant and the agent's
                            knowledge-search tool.
"""
import re
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm
from core.rag.retriever import get_advanced_retriever, format_docs_for_prompt

# How many past messages (user + bot, combined) to carry into both the
# contextualization step and the final answer. Keeps prompts small and
# avoids the model getting confused by very old, no-longer-relevant turns.
MAX_HISTORY_MESSAGES = 8


# Safety net: even with an instruction not to mention the knowledge base,
# LLMs occasionally slip in a meta-commentary opener anyway. Strip any
# leading sentence like this before it reaches the user.
_META_COMMENTARY_PREFIX = re.compile(
    r"^\s*(i\s+(couldn'?t|could\s+not|didn'?t|did\s+not)\s+find\s+.{0,120}?"
    r"(reference material|knowledge base|source|document)s?.{0,80}?[.!]\s*"
    r"(in general,?\s*)?)",
    re.IGNORECASE,
)


def _strip_meta_commentary(text: str) -> str:
    cleaned = _META_COMMENTARY_PREFIX.sub("", text).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or text


def format_chat_history(history: list[dict] | None) -> str:
    """Turns a list of {"role": "user"|"bot", "content": str} into a plain
    transcript for prompt injection. Accepts None/empty gracefully. Only
    the most recent MAX_HISTORY_MESSAGES entries are kept."""
    if not history:
        return ""
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    lines = []
    for turn in trimmed:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


ADVICE_SYSTEM_PROMPT = """You are FreshEye AI, a professional food-safety and \
produce-freshness advisor embedded in an industrial fruit inspection system.

You are given:
- The output of a computer-vision classifier (predicted class + confidence)
- Reference material retrieved from a food-safety knowledge base

Your job is to produce a short, clear, professional recommendation for the \
end user (a shopper, or a QA operator on a packing line). Always ground your \
reasoning in the retrieved reference material when it's relevant, and be \
explicit about your confidence.

Respond in this exact structure (plain text, no markdown headers):

VERDICT: <one of: SAFE TO EAT | USE SOON | DO NOT EAT>
CONFIDENCE NOTE: <one sentence about how much to trust the classifier's confidence score>
REASONING: <2-3 sentences explaining why, grounded in the reference material>
RECOMMENDATION: <1-2 concrete, actionable sentences - e.g. storage tips, or how to use it if it's borderline, or a food-safety warning if rotten>
"""

ADVICE_USER_PROMPT = """Classifier prediction: "{label}" (confidence: {confidence:.1%})
Freshness heuristic from label: {is_fresh_hint}

Reference material:
{context}

Write the structured recommendation now."""


CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user \
message, rewrite the latest message into a fully standalone question that \
makes sense without the history - resolve any pronouns or vague references \
("it", "that one", "the fruit we talked about") into the specific thing \
being discussed. Do NOT answer the question. Do NOT add new information. \
If the latest message is already standalone, or there is no relevant \
history, return it unchanged. Output ONLY the rewritten question, nothing else."""

CONTEXTUALIZE_USER_PROMPT = """Chat history:
{chat_history}

Latest message: {question}

Standalone question:"""


CHAT_SYSTEM_PROMPT = """You are FreshEye AI's knowledge assistant. Answer the \
user's question about fruit/vegetable freshness, storage, or food safety \
directly and confidently, grounding your answer in the provided reference \
material whenever it's relevant. Draw on general food-safety best practice \
to fill any gaps, but do so seamlessly - never mention the reference \
material, a knowledge base, or whether something "was covered" or "was \
found"; the user should just get a clear, practical answer. If chat history \
is provided, use it to understand what the user is referring to (e.g. "it", \
"that fruit", "the one I mentioned") and answer as a natural continuation of \
the conversation - never ask the user to clarify something the history \
already makes clear. Keep answers concise (3-5 sentences) and practical."""

CHAT_USER_PROMPT = """Chat history:
{chat_history}

Reference material:
{context}

Question: {question}"""


@lru_cache(maxsize=1)
def _cached_retriever():
    return get_advanced_retriever(get_llm())


def get_advice_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", ADVICE_SYSTEM_PROMPT),
        ("human", ADVICE_USER_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


@lru_cache(maxsize=1)
def get_contextualize_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        ("human", CONTEXTUALIZE_USER_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


@lru_cache(maxsize=1)
def get_chat_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_SYSTEM_PROMPT),
        ("human", CHAT_USER_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


def generate_freshness_advice(label: str, confidence: float, is_fresh: bool | None) -> dict:
    """High-level helper used by the /api/predict routes: retrieves
    relevant knowledge for this label, then generates structured advice.
    This is always a single-turn call - each image is judged on its own.
    """
    retriever = _cached_retriever()
    query = f"{label} freshness signs storage advice food safety"
    docs = retriever.invoke(query)
    context = format_docs_for_prompt(docs)

    hint = {True: "likely fresh", False: "likely rotten/spoiled", None: "unclear"}[is_fresh]

    chain = get_advice_chain()
    raw_text = chain.invoke({
        "label": label,
        "confidence": confidence,
        "is_fresh_hint": hint,
        "context": context,
    })

    return {
        "advice_text": _strip_meta_commentary(raw_text.strip()),
        "sources": sorted({d.metadata.get("source", "knowledge base").split("/")[-1] for d in docs}),
    }


def answer_chat_question(question: str, history: list[dict] | None = None) -> dict:
    """History-aware RAG Q&A. `history` is a list of {"role", "content"}
    dicts (oldest first) representing the recent conversation - pass the
    prior turns only, not the current `question`."""
    history_text = format_chat_history(history)

    # Step 1: resolve pronouns/references into a standalone search query.
    if history_text:
        standalone_question = get_contextualize_chain().invoke({
            "chat_history": history_text,
            "question": question,
        }).strip()
        if not standalone_question:
            standalone_question = question
    else:
        standalone_question = question

    # Step 2: retrieve using the standalone question, but answer using the
    # original question + history so the reply stays natural.
    retriever = _cached_retriever()
    docs = retriever.invoke(standalone_question)
    context = format_docs_for_prompt(docs)

    chain = get_chat_chain()
    answer = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": history_text or "(none)",
    })

    return {
        "answer": _strip_meta_commentary(answer.strip()),
        "sources": sorted({d.metadata.get("source", "knowledge base").split("/")[-1] for d in docs}),
    }
