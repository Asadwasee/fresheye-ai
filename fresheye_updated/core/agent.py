"""
core/agent.py
=============
A LangChain tool-calling AGENT that orchestrates the full inspection flow:

    image/label  --> [classify_fruit tool]        --> raw prediction
    prediction   --> [search_knowledge_base tool]  --> reference material
    everything   --> [freshness_advisor tool]      --> final structured advice

Rather than hard-coding this pipeline in Python, the agent is given these
three tools and a system prompt describing the flow, and the LLM decides
which tools to call and in what order (typically: classify -> search ->
advise, but it can also just answer a follow-up chat question using only
the knowledge-base tool, or re-run advice generation if the user asks
"are you sure?").

CONVERSATIONAL MEMORY: the agent's prompt includes a `chat_history`
messages placeholder (LangChain's standard pattern for giving an agent
short-term memory). Callers pass in the recent conversation as a list of
{"role": "user"|"bot", "content": str} dicts; this module converts them to
proper HumanMessage/AIMessage objects so the agent can resolve references
like "it" or "that one" to whatever was discussed a few turns earlier -
without needing a server-side session store, since the frontend simply
replays recent turns with each request.

This is used by the AI Agent page for grounded, multi-turn follow-up
conversation about a single classified item, and can also be used as the
single entry point for the whole predict-and-advise flow.
"""
from functools import lru_cache

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from core.llm import get_llm
from core.classifier import get_classifier
from core.rag.chain import generate_freshness_advice, answer_chat_question

# How many past messages to replay into the agent's memory per request.
MAX_AGENT_HISTORY_MESSAGES = 8


AGENT_SYSTEM_PROMPT = """You are FreshEye AI, an autonomous produce-inspection \
agent for an industrial/consumer fruit freshness detection system.

You have three tools:
1. classify_fruit_image - runs the computer-vision model on an uploaded image and returns the predicted class + confidence.
2. search_freshness_knowledge - searches a food-safety knowledge base for relevant guidance.
3. generate_structured_advice - turns a classification result into a final VERDICT / REASONING / RECOMMENDATION.

Standard flow when given an image path: call classify_fruit_image first, \
then call generate_structured_advice with its output (that tool internally \
also searches the knowledge base, so you usually don't need to call \
search_freshness_knowledge separately unless the user asks a follow-up \
question that isn't about a specific classification).

For general questions ("how do I store bananas?") just use \
search_freshness_knowledge and answer directly - no need to classify anything.

You will often be given the recent conversation history. Use it to resolve \
references like "it", "that one", or "the fruit we talked about" to the \
specific item being discussed - never ask the user to clarify something the \
history already makes clear.

Always be concise, professional, and food-safety conscious. Never tell a \
user something is safe to eat if the evidence suggests otherwise."""


def _to_lc_messages(history: list[dict] | None) -> list[BaseMessage]:
    """Converts [{"role": "user"|"bot", "content": str}, ...] into
    LangChain message objects, trimmed to the most recent N entries."""
    if not history:
        return []
    trimmed = history[-MAX_AGENT_HISTORY_MESSAGES:]
    messages: list[BaseMessage] = []
    for turn in trimmed:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


@tool
def classify_fruit_image(image_path: str) -> str:
    """Run the fruit freshness YOLO classifier on an image file path and
    return the predicted label, confidence, and a fresh/rotten hint."""
    clf = get_classifier()
    result = clf.predict_image(image_path)
    return (
        f"label={result.label} confidence={result.confidence:.4f} "
        f"is_fresh={result.is_fresh}"
    )


@tool
def search_freshness_knowledge(query: str) -> str:
    """Search the food-safety / fruit-freshness knowledge base for
    information relevant to the given query and return a short answer."""
    result = answer_chat_question(query)
    return result["answer"]


@tool
def generate_structured_advice(label: str, confidence: float, is_fresh_hint: str) -> str:
    """Generate the final structured VERDICT/REASONING/RECOMMENDATION advice
    for a given classification. is_fresh_hint should be 'true', 'false' or
    'unknown'."""
    hint_map = {"true": True, "false": False}
    is_fresh = hint_map.get(is_fresh_hint.lower(), None)
    result = generate_freshness_advice(label, confidence, is_fresh)
    return result["advice_text"]


TOOLS = [classify_fruit_image, search_freshness_knowledge, generate_structured_advice]


@lru_cache(maxsize=1)
def get_agent_executor() -> AgentExecutor:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=True, max_iterations=6)


def run_agent_on_image(image_path: str, user_message: str | None = None,
                        chat_history: list[dict] | None = None) -> dict:
    """Runs the full agentic flow on an already-uploaded image."""
    executor = get_agent_executor()
    instruction = (
        f"Classify the fruit image at path '{image_path}', then generate "
        f"structured freshness advice for it."
    )
    if user_message:
        instruction += f" The user also asked: {user_message}"

    result = executor.invoke({
        "input": instruction,
        "chat_history": _to_lc_messages(chat_history),
    })
    return {"agent_output": result["output"]}


def run_agent_chat(user_message: str, context: str | None = None,
                    chat_history: list[dict] | None = None) -> dict:
    """Runs the agent for a free-form chat question. `context` is an
    optional block of already-known information (e.g. "this item was just
    classified as X with Y% confidence, verdict Z") so the agent can answer
    contextual follow-up questions about a specific item without needing to
    re-run the vision classifier. `chat_history` is the recent conversation
    (see _to_lc_messages) - together these give the agent both grounding
    (what item is this?) and memory (what did we already discuss?)."""
    executor = get_agent_executor()
    instruction = user_message
    if context:
        instruction = (
            f"Known context about the item currently being discussed:\n{context}\n\n"
            f"User's follow-up question: {user_message}"
        )
    result = executor.invoke({
        "input": instruction,
        "chat_history": _to_lc_messages(chat_history),
    })
    return {"agent_output": result["output"]}
