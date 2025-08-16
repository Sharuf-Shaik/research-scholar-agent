# Research Scholar Agent (Prototype)

An AI-powered academic assistant that performs advanced academic searches (arXiv, Crossref), analyzes recent publications, synthesizes findings across disciplines, and writes well-structured academic reports with proper citations.

## Features
- Async multi-source academic search (arXiv, Crossref)
- Query expansion and deduplication
- Lightweight ranking by relevance and recency
- LLM-powered synthesis with inline numeric citations [1], [2], ...
- CLI, REST API (FastAPI), and simple Streamlit UI
- Pluggable LLM backends: OpenAI, Together, Ollama (auto-detected)

## Quickstart

1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Set an LLM provider:
- OpenAI: export `OPENAI_API_KEY=...`
- Together: export `TOGETHER_API_KEY=...`
- Ollama (local): ensure `ollama` is running (default at `http://localhost:11434`).

3. Run a research job from the CLI:
```bash
python -m research_scholar_agent.cli research "large language models for healthcare" --top-k 8 --out report.md
```

4. Start the API server:
```bash
python -m research_scholar_agent.cli serve --host 0.0.0.0 --port 8000
# Then POST to http://localhost:8000/research
```

5. Launch the Streamlit UI:
```bash
python -m research_scholar_agent.cli ui
```

## Environment Variables
- `DEFAULT_LLM_PROVIDER` = `openai` | `together` | `ollama` (auto if unset)
- `OPENAI_API_KEY` (when using OpenAI)
- `TOGETHER_API_KEY` (when using Together)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `CROSSREF_MAILTO` (recommended for polite Crossref usage)

## Notes
- The prototype prefers abstracts and metadata; PDF retrieval is intentionally omitted for reliability and speed.
- Rate limits may apply depending on provider policies.