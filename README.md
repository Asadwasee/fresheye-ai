# FreshEye AI 

**AI-powered fruit freshness detection and advisory system.**

FreshEye AI is a full-stack web application that detects fruit in an uploaded photo, classifies its freshness using a trained vision model, and generates grounded, source-backed advice using a Retrieval-Augmented Generation (RAG) pipeline — with an AI agent for natural follow-up conversation.

---

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Building the Knowledge Base](#-building-the-knowledge-base)
- [Running the App](#-running-the-app)
- [API Reference](#-api-reference)
- [How It Works](#-how-it-works)
- [Known Limitations](#-known-limitations)
- [Future Work](#-future-work)
- [Team](#-team)
- [License](#-license)

---

## Overview

Roughly a third of all food produced for human consumption is lost or wasted every year, much of it to spoilage that goes unnoticed or sound produce that gets needlessly thrown out. Manual visual inspection is the usual method of checking freshness, but it's subjective — people disagree most right at the boundary that matters most (is this still fresh, or not?).

FreshEye AI replaces that guesswork with:

1. **A vision model** (YOLO) that detects and classifies fruit freshness from a photo, with a confidence score.
2. **A RAG pipeline** (LangChain + ChromaDB + Llama 3) that turns that classification into clear, factual storage and food-safety advice — grounded in a curated knowledge base, not guessed by the LLM.
3. **An AI agent** that handles natural follow-up questions about the same item, deciding for itself which tool to use next.

---

## Features

- **Single & batch image inspection** — upload one photo or many; each gets its own independent verdict.
- **Fresh / Unripe / Rotten classification** with a calibrated confidence score.
- **Non-fruit detection** — the system honestly declines to classify an image that isn't a fruit, instead of fabricating a verdict.
- **Grounded AI advice** — every recommendation is generated from a curated, source-attributed knowledge base, not invented by the LLM.
- **Standalone knowledge-base chat** ("Ask FreshEye") for general storage/food-safety questions.
- **AI Agent page** — upload a photo once, then keep chatting about that same item; the agent decides on its own whether to re-classify, search the knowledge base, or just answer.
- **Conversational memory** — follow-up questions like *"is it safe to eat?"* are resolved against what was already discussed.
- **Pluggable LLM backend** — switch between Groq (hosted, fast) and Ollama (fully local, offline, zero cost) with a single `.env` flag.
- **Health-check endpoint** for quick diagnostics (model loaded? LLM key valid?).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vision Model | YOLO (Ultralytics) — classification |
| Backend | Flask (REST API) |
| RAG / Orchestration | LangChain |
| Vector Database | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, open-source) |
| LLM | Llama 3 — via **Groq** (hosted) or **Ollama** (local) |
| Frontend | HTML5, CSS3, Vanilla JavaScript (no framework, no build step) |

---

## System Architecture

FreshEye AI follows a simple three-layer design; each layer only talks to its immediate neighbour.

```
┌──────────────────────────────┐
│      Presentation Layer       │   Browser UI — upload, result cards, chat
│   (HTML5 / CSS3 / Vanilla JS) │
└───────────────┬───────────────┘
                 │  REST calls
┌───────────────▼───────────────┐
│       Application Layer        │   Flask — validation, routing, JSON
│   (app.py + routes/ blueprints)│   Model & pipeline loaded once at start-up
└───────────────┬───────────────┘
                 │
     ┌───────────┴────────────┐
     ▼                        ▼
┌─────────────┐      ┌──────────────────┐
│Vision Pipeline│      │   RAG Pipeline    │
│ YOLO classify │      │ ChromaDB retrieval │
│ (core/classifier.py)│ + LangChain + LLM  │
└─────────────┘      │ (core/rag/*, core/agent.py) │
                      └──────────────────┘
```

The backend is **stateless** — conversation history is replayed by the client on every request rather than stored server-side, so it scales horizontally with no session cleanup required.

---

## Project Structure

```
.
├── app.py                      # Flask application entry point
├── config.py                   # All settings, loaded from .env
├── requirements.txt
├── .env.example                 # Environment variable template
│
├── core/
│   ├── api_keys.py             # Centralised LLM API key loading/validation
│   ├── llm.py                  # LLM + embeddings factory (Groq / Ollama)
│   ├── classifier.py           # YOLO model wrapper (thread-safe singleton)
│   ├── agent.py                # LangChain tool-calling agent
│   └── rag/
│       ├── ingest.py           # Knowledge base loading + chunking + Chroma indexing
│       ├── retriever.py        # Advanced retriever (MMR + multi-query + compression)
│       ├── chain.py            # LCEL chains — freshness advice + chat Q&A
│       └── knowledge/          # Markdown/PDF knowledge base (edit freely)
│
├── routes/
│   ├── predict.py              # /api/predict, /api/predict/batch
│   └── chat.py                 # /api/chat, /api/agent/chat
│
├── templates/
│   ├── base.html               # Shared layout + result-card template
│   ├── inspect.html            # Page 1 — single/batch inspection
│   ├── agent.html              # Page 2 — AI agent conversation
│   └── chat.html               # Page 3 — standalone knowledge chat
│
├── static/
│   ├── css/style.css
│   └── js/
│       ├── common.js           # Shared DOM helpers + health check
│       ├── results.js          # Shared result-card rendering
│       ├── inspect.js
│       ├── agent.js
│       └── chat.js
│
└── models/
    └── best.pt                 # Trained YOLO classification weights
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- `pip`
- A [Groq API key](https://console.groq.com/keys) (free tier available) **or** [Ollama](https://ollama.com) installed locally

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Copy the template and fill in your own values:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|---|---|---|
| `FLASK_SECRET_KEY` | Flask session secret | — (set your own) |
| `PORT` | Server port | `5000` |
| `LLM_PROVIDER` | `groq` or `ollama` | `groq` |
| `GROQ_API_KEY` | Your Groq API key (required if `LLM_PROVIDER=groq`) | — |
| `GROQ_MODEL` | Groq-hosted model name | `llama-3.3-70b-versatile` |
| `OLLAMA_BASE_URL` | Local Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Local model name | `llama3.1` |
| `EMBEDDING_MODEL` | Local embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `YOLO_WEIGHTS_PATH` | Path to trained weights | `models/best.pt` |
| `YOLO_CONFIDENCE_THRESHOLD` | Minimum confidence to accept a prediction | `0.35` |
| `MAX_CONTENT_LENGTH_MB` | Max upload size | `100` |

> **Never commit your real `.env` file.** It is already excluded via `.gitignore`.

---

## Building the Knowledge Base

Before first run, build the vector index from the markdown knowledge base:

```bash
python -m core.rag.ingest
```

This parses every file in `core/rag/knowledge/` (`.md`, `.pdf`, `.txt` are all supported), splits it using a hierarchical header-aware + recursive splitter, embeds it locally, and persists it to `chroma_db/` (auto-created, git-ignored). The app will also build this automatically on first request if the step is skipped. Re-run it any time the knowledge base content changes.

---

## Running the App

```bash
python app.py
```

Visit **http://localhost:5000**.

| Page | Route | Purpose |
|---|---|---|
| Inspect | `/` | Single or batch photo freshness inspection |
| AI Agent | `/agent` | Upload once, then chat about that item |
| Ask FreshEye | `/chat` | Standalone knowledge-base Q&A |
| Health Check | `/api/health` | Confirms model weights loaded + LLM key valid |

---

##  API Reference

| Endpoint | Method | Body | Purpose |
|---|---|---|---|
| `/api/predict` | `POST` | multipart, field `image` | Classify one image + generate advice |
| `/api/predict/batch` | `POST` | multipart, field `images` (repeat per file) | Classify multiple images; each gets its own result. Capped at `MAX_BATCH_IMAGES`. |
| `/api/chat` | `POST` | JSON `{ message, history? }` | History-aware RAG Q&A over the knowledge base |
| `/api/agent/chat` | `POST` | JSON `{ message, context?, history?, image_path? }` | Agentic follow-up chat about the current item |
| `/api/health` | `GET` | — | System status |

**Example — `/api/predict` response:**

```json
{
  "filename": "apple1.jpg",
  "image_url": "/static/uploads/images/apple1.jpg",
  "classification": {
    "label": "freshapple",
    "confidence": 0.97,
    "all_probs": { "...": 0.0 },
    "is_fresh": true
  },
  "advice": {
    "advice_text": "VERDICT: SAFE TO EAT\n...",
    "verdict": "SAFE TO EAT",
    "not_applicable": false
  }
}
```

If the model predicts a non-fruit class, `advice.verdict` becomes `"UNCLASSIFIED"` and `advice.not_applicable` is `true` — the RAG advisor is skipped entirely, and a fixed, honest message is returned instead of a fabricated verdict.

---

## How It Works

1. **Detection & Classification** — an uploaded image is passed through the YOLO classifier, which outputs a predicted class (e.g. `rottenbanana`) and a confidence score. A simple keyword rule (`Config.FRESH_KEYWORDS` / `ROTTEN_KEYWORDS`) maps that class to a Fresh/Rotten verdict.
2. **Retrieval** — the predicted label is turned into a search query, which runs through a three-layer retriever over the ChromaDB knowledge base:
   - **MMR search** — relevant *and* diverse results, avoiding duplicate chunks.
   - **Multi-query expansion** — the query is rewritten several ways by the LLM to catch wording the raw label alone would miss.
   - **Contextual compression** — chunks below a similarity threshold are filtered out.
3. **Generation** — the retrieved, source-attributed context is passed to the LLM, which produces a structured `VERDICT / CONFIDENCE NOTE / REASONING / RECOMMENDATION` response — grounded in real documents, not guessed.
4. **Agent** — for the `/agent` page, a LangChain tool-calling agent decides for itself, turn by turn, whether to classify an image, search the knowledge base, or generate advice, based on the conversation so far.

---

## Known Limitations

- **Surface-only assessment** — the model sees what the camera sees; internal decay with no visible external sign cannot be detected.
- **Fresh / Unripe boundary** — this transition is genuinely continuous in nature, and is the hardest case for the classifier.
- **Static knowledge base** — the RAG corpus is a set of text files; it needs periodic human curation, with no automatic refresh.
- **Client-side conversation memory** — chat history lives in the browser tab and resets on page reload (by design, to keep the backend stateless and scalable).

---

## Future Work

- Broader fruit coverage beyond the current classes.
- On-device mobile inference (TensorFlow Lite / Core ML export).
- Fine-tuned domain LLM for common questions, with RAG retained for the long tail.
- Shelf-life estimation (regression instead of discrete classes).
- Inventory / ERP integration — a rotten detection could trigger a waste record or reorder automatically.

---

## Team

| Name | Role |
|---|---|
| Asad Waseem | AI, Backend & RAG Lead |
| Abdul Samad | Computer Vision & Data Lead |
| Sharif Sarwar | Frontend, Integration & QA |

Final Year Project — Department of Software Engineering, University of Science and Technology, Lahore.
Supervisor: Sir Tamour Ali Khan, Lecturer.

---

## License

This project is submitted as a Final Year Project (FYP) for academic purposes.
