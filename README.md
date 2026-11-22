# Fact-Check Agent — Automated PDF Truth Layer

A Streamlit web application that automates claim verification for marketing reports, whitepapers, and business PDFs. Upload any PDF to extract concrete factual claims (statistics, dates, financial and technical figures), verify each claim against live web search data, flag inaccuracies, and surface the verified "real" facts with clickable source links.

## Supported LLM Providers & Web Grounding

Select your provider directly in the Streamlit sidebar:

1. **Google Gemini API (`gemini-2.5-flash`)**
   - Uses `google-genai` structured JSON output for claim extraction.
   - Uses native **Google Search Grounding** (`google_search` tool) for live verification and clickable grounding citations.
2. **OpenRouter (`google/gemini-2.5-flash`)**
   - Uses OpenRouter's Chat Completions API for claim extraction and verification.
   - Enables live web grounding via OpenRouter's web search plugin (`plugins: [{"id": "web"}]`, with fallback to the `openrouter:web_search` server tool), which routes Google models through native Google Search (or Exa fallback) and returns standardized `url_citation` annotations.
   - **Grounding Guardrail:** If no live web citations are returned for a claim, the app never marks an ungrounded response as `Verified`; it flags the claim with a clear low-confidence notice.

## Project Structure

```text
fact-check-agent/
├── app.py                        # Streamlit frontend & results dashboard
├── core/
│   ├── __init__.py
│   ├── config.py                 # Secure .env & Streamlit Secrets loader
│   ├── pdf_parser.py             # PyMuPDF text & page extraction
│   ├── claim_extractor.py        # Structured claim extraction (Gemini & OpenRouter)
│   └── fact_verifier.py          # Parallel live web verification & citations
├── sample_trap_report.pdf        # Built-in Trap Document (true, outdated & false claims)
├── .env.example                  # Environment variable template (safe to commit)
├── .gitignore                    # Excludes .env, .env.*, and .streamlit/secrets.toml
├── requirements.txt              # Python dependencies
├── .streamlit/
│   ├── config.toml               # UI theme configuration
│   └── secrets.toml.example      # Streamlit Cloud Secrets template
└── README.md
```

## Local Setup & Run

1. **Create a virtual environment and install dependencies:**
   ```bash
   cd fact-check-agent
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Create your local `.env` file from the template:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and add your API key(s):
   ```env
   GEMINI_API_KEY=your-google-gemini-api-key
   OPENROUTER_API_KEY=your-openrouter-api-key
   LLM_PROVIDER=gemini
   GEMINI_MODEL=gemini-2.5-flash
   OPENROUTER_MODEL=google/gemini-2.5-flash
   ```
   *(Never commit `.env` or `.streamlit/secrets.toml` to Git; both are ignored by `.gitignore`.)*

3. **Launch the application locally:**
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud

1. Push the `fact-check-agent/` directory to a public GitHub repository.
2. Open [share.streamlit.io](https://share.streamlit.io/), click **Create app**, and select your repository, branch (`main`), and main file path (`app.py`).
3. Under **Advanced settings → Secrets**, add your keys:
   ```toml
   GEMINI_API_KEY = "your-google-gemini-api-key"
   OPENROUTER_API_KEY = "your-openrouter-api-key"
   LLM_PROVIDER = "gemini"
   GEMINI_MODEL = "gemini-2.5-flash"
   OPENROUTER_MODEL = "google/gemini-2.5-flash"
   ```
4. Click **Deploy** and verify both providers using the built-in **Try Sample Trap PDF** button.
