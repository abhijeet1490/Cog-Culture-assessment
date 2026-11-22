"""Fact-Check Agent: Automated PDF Claim Verification Web App."""

from pathlib import Path
from typing import List, Optional
import hashlib
import html

import pandas as pd
import streamlit as st

from core.config import (
    get_default_provider,
    get_missing_key_message,
    get_model_for_provider,
    init_env,
    resolve_provider_api_key,
)
from core.pdf_parser import extract_text_from_pdf, format_pages_for_llm
from core.claim_extractor import extract_claims
from core.fact_verifier import VerificationResult, verify_claims_parallel

# Load local .env variables (if present) without overriding existing env vars
init_env()

# -----------------------------------------------------------------------------
# Page Config & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fact-Check Agent | PDF Truth Layer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1120px;
    }
    .hero-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #FFFFFF;
        padding: 1.75rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #2563EB;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #CBD5E1;
        margin: 0;
        line-height: 1.5;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        text-align: left;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .kpi-val {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .claim-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 0.9rem;
    }
    .claim-card.verified { border-left: 5px solid #059669; }
    .claim-card.inaccurate { border-left: 5px solid #D97706; }
    .claim-card.false { border-left: 5px solid #DC2626; }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-verified { background: #D1FAE5; color: #065F46; }
    .badge-inaccurate { background: #FEF3C7; color: #92400E; }
    .badge-false { background: #FEE2E2; color: #991B1B; }
    .meta-tag {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        background: #F1F5F9;
        color: #475569;
        margin-left: 0.4rem;
    }
    .claim-title {
        font-size: 1.03rem;
        font-weight: 600;
        color: #0F172A;
        margin: 0.55rem 0 0.35rem 0;
        line-height: 1.45;
    }
    .quote-box {
        font-size: 0.86rem;
        color: #475569;
        background: #F8FAFC;
        border-left: 3px solid #CBD5E1;
        padding: 0.45rem 0.75rem;
        border-radius: 0 6px 6px 0;
        margin: 0.45rem 0 0.65rem 0;
        font-style: italic;
    }
    .fact-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 0.65rem 0.9rem;
        margin-top: 0.5rem;
        font-size: 0.92rem;
        color: #1E3A8A;
    }
    .fact-box.verified {
        background: #ECFDF5;
        border-color: #A7F3D0;
        color: #065F46;
    }
    .source-link {
        display: inline-block;
        font-size: 0.8rem;
        color: #2563EB !important;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.18rem 0.55rem;
        margin: 0.25rem 0.35rem 0 0;
        text-decoration: none;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SAMPLE_PDF_PATH = Path(__file__).parent / "sample_trap_report.pdf"

PROVIDER_LABELS = {
    "gemini": "Google Gemini API",
    "openrouter": "OpenRouter",
}
GROUNDING_NOTES = {
    "gemini": "Live Google Search Grounding (native Gemini tool)",
    "openrouter": "OpenRouter Live Web Search Grounding (native Google Search / Exa fallback)",
}


def build_markdown_report(filename: str, results: List[VerificationResult]) -> str:
    """Generate a clean Markdown fact-check report for download."""
    total = len(results)
    verified = sum(1 for r in results if r.status == "Verified")
    inaccurate = sum(1 for r in results if r.status == "Inaccurate")
    false_cnt = sum(1 for r in results if r.status == "False")

    lines = [
        f"# Fact-Check Report: {filename}",
        "",
        f"- **Total Claims Checked:** {total}",
        f"- **Verified:** {verified}",
        f"- **Inaccurate (Outdated / Distorted):** {inaccurate}",
        f"- **False (Fabricated / No Evidence):** {false_cnt}",
        "",
        "---",
        "",
    ]
    for r in results:
        lines.append(f"## Claim #{r.claim_id} — [{r.status.upper()}] (Page {r.page_number} | {r.category})")
        lines.append(f"- **Claim:** {r.claim_text}")
        lines.append(f"- **PDF Quote:** \"{r.exact_quote}\"")
        lines.append(f"- **Assessment:** {r.explanation}")
        lines.append(f"- **Verified Real Fact:** {r.correct_fact}")
        if r.sources:
            src_md = ", ".join(f"[{s.title}]({s.url})" for s in r.sources)
            lines.append(f"- **Live Sources:** {src_md}")
        lines.append("")
    return "\n".join(lines)


def build_csv_bytes(results: List[VerificationResult]) -> bytes:
    """Serialize verification results to UTF-8 CSV bytes."""
    rows = []
    for r in results:
        rows.append(
            {
                "Claim #": r.claim_id,
                "Page": r.page_number,
                "Category": r.category,
                "Status": r.status,
                "Confidence": r.confidence,
                "Claim": r.claim_text,
                "PDF Quote": r.exact_quote,
                "Explanation": r.explanation,
                "Verified Real Fact": r.correct_fact,
                "Sources": " | ".join(s.url for s in r.sources),
            }
        )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def render_claim_card(r: VerificationResult) -> None:
    """Render a single styled claim verification card with safe HTML escaping."""
    status_slug = r.status.lower()
    badge_cls = f"badge-{status_slug}"
    fact_cls = "fact-box verified" if status_slug == "verified" else "fact-box"
    fact_label = "Confirming Evidence" if status_slug == "verified" else "Correct Real Fact"

    safe_status = html.escape(r.status)
    safe_category = html.escape(r.category)
    safe_confidence = html.escape(r.confidence)
    safe_claim = html.escape(r.claim_text)
    safe_quote = html.escape(r.exact_quote)
    safe_explanation = html.escape(r.explanation)
    safe_fact = html.escape(r.correct_fact)

    sources_html = ""
    if r.sources:
        links = "".join(
            f'<a class="source-link" href="{html.escape(s.url, quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">{html.escape(s.title[:52])}</a>'
            for s in r.sources
            if s.url.startswith(("http://", "https://"))
        )
        if links:
            sources_html = f'<div style="margin-top:0.55rem;">{links}</div>'

    card_html = f"""
    <div class="claim-card {status_slug}">
        <div>
            <span class="badge {badge_cls}">{safe_status}</span>
            <span class="meta-tag">Page {r.page_number}</span>
            <span class="meta-tag">{safe_category}</span>
            <span class="meta-tag">Confidence: {safe_confidence}</span>
        </div>
        <div class="claim-title">#{r.claim_id}. {safe_claim}</div>
        <div class="quote-box">“{safe_quote}”</div>
        <div style="font-size:0.92rem; color:#334155; line-height:1.5;">{safe_explanation}</div>
        <div class="{fact_cls}"><strong>{fact_label}:</strong> {safe_fact}</div>
        {sources_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
default_provider = get_default_provider(secrets=st.secrets)
provider_keys = ["gemini", "openrouter"]
default_idx = provider_keys.index(default_provider) if default_provider in provider_keys else 0

with st.sidebar:
    st.markdown("### Configuration")
    selected_provider = st.selectbox(
        "LLM Provider",
        options=provider_keys,
        index=default_idx,
        format_func=lambda k: PROVIDER_LABELS[k],
        help="Switch between direct Google Gemini API and OpenRouter.",
    )

    configured_model = get_model_for_provider(selected_provider, secrets=st.secrets)
    model_name = st.selectbox(
        "Verification Model",
        options=[configured_model],
        index=0,
        help=GROUNDING_NOTES[selected_provider],
    )

    key_label = (
        "Gemini API Key (Optional Override)"
        if selected_provider == "gemini"
        else "OpenRouter API Key (Optional Override)"
    )
    user_api_key = st.text_input(
        key_label,
        type="password",
        help="Uses Streamlit Secrets or local .env by default. Enter a key here only to override.",
    )

    max_claims = st.slider(
        "Max claims to verify",
        min_value=5,
        max_value=25,
        value=15,
        step=1,
        help="Caps the number of extracted claims to keep response latency fast.",
    )

    st.caption(f"Grounding: {GROUNDING_NOTES[selected_provider]}")
    st.markdown("---")
    st.markdown(
        "**Verdict Legend**\n"
        "- **Verified:** Matches live authoritative web data.\n"
        "- **Inaccurate:** Outdated, distorted, or misleading stat.\n"
        "- **False:** Fabricated claim, contradicted, or no live web evidence."
    )

# -----------------------------------------------------------------------------
# Main Header & Upload Flow
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Fact-Check Agent — Automated PDF Truth Layer</div>
        <p class="hero-sub">
            Upload any marketing report, whitepaper, or pitch deck PDF. The agent extracts factual claims
            (statistics, dates, financial and technical figures), verifies each against live web search,
            flags inaccuracies, and returns the verified real facts with sources.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_upload, col_sample = st.columns([3, 1.3], gap="medium")
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a PDF document to fact-check",
        type=["pdf"],
        help="Text-based PDFs up to 30 pages.",
    )
with col_sample:
    st.markdown("<div style='height:1.7rem;'></div>", unsafe_allow_html=True)
    use_sample = st.button(
        "Try Sample Trap PDF",
        use_container_width=True,
        type="secondary",
        disabled=not SAMPLE_PDF_PATH.exists(),
        help="Loads a built-in test PDF containing intentional lies and outdated stats.",
    )

if use_sample and SAMPLE_PDF_PATH.exists():
    st.session_state["active_pdf_bytes"] = SAMPLE_PDF_PATH.read_bytes()
    st.session_state["active_pdf_name"] = SAMPLE_PDF_PATH.name
    st.session_state.pop("last_results", None)
elif uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("active_pdf_hash") != file_hash:
        st.session_state["active_pdf_bytes"] = file_bytes
        st.session_state["active_pdf_name"] = uploaded_file.name
        st.session_state["active_pdf_hash"] = file_hash
        st.session_state.pop("last_results", None)

active_bytes: Optional[bytes] = st.session_state.get("active_pdf_bytes")
active_name: str = st.session_state.get("active_pdf_name", "")

if active_bytes:
    st.info(
        f"Loaded document: **{active_name}** ({len(active_bytes) / 1024:.1f} KB)  |  "
        f"Provider: **{PROVIDER_LABELS[selected_provider]}** (`{model_name}`)"
    )
    run_btn = st.button("Run Automated Fact-Check", type="primary", use_container_width=False)

    if run_btn:
        api_key = resolve_provider_api_key(
            provider=selected_provider,
            user_override=user_api_key,
            secrets=st.secrets,
        )
        if not api_key:
            st.error(get_missing_key_message(selected_provider))
            st.stop()

        cache_key = (
            f"{hashlib.sha256(active_bytes).hexdigest()}_"
            f"{selected_provider}_{model_name}_{max_claims}"
        )
        cached = st.session_state.get("results_cache", {}).get(cache_key)

        if cached:
            st.session_state["last_results"] = cached
        else:
            try:
                with st.status("Running Truth Layer pipeline...", expanded=True) as status_ui:
                    st.write("1. Extracting text and page structure from PDF...")
                    pages = extract_text_from_pdf(active_bytes)
                    doc_text = format_pages_for_llm(pages)
                    st.write(
                        f"Parsed {len(pages)} page(s). Extracting claims via "
                        f"{PROVIDER_LABELS[selected_provider]}..."
                    )

                    claims = extract_claims(
                        document_text=doc_text,
                        api_key=api_key,
                        model_name=model_name,
                        max_claims=max_claims,
                        provider=selected_provider,
                    )
                    if not claims:
                        status_ui.update(label="No verifiable factual claims found.", state="complete")
                        st.warning("No objective statistics, dates, or factual claims were detected in this PDF.")
                        st.stop()

                    st.write(f"2. Found {len(claims)} claims. Verifying against live web search...")
                    progress_bar = st.progress(0.0)

                    def on_progress(done: int, total: int) -> None:
                        progress_bar.progress(done / total)

                    results = verify_claims_parallel(
                        claims=claims,
                        api_key=api_key,
                        model_name=model_name,
                        max_workers=4,
                        progress_callback=on_progress,
                        provider=selected_provider,
                    )
                    status_ui.update(
                        label=f"Verification complete ({len(results)} claims analyzed).",
                        state="complete",
                    )

                st.session_state.setdefault("results_cache", {})[cache_key] = results
                st.session_state["last_results"] = results
            except Exception as exc:
                st.error(f"Could not process PDF: {exc}")
                st.stop()

# -----------------------------------------------------------------------------
# Results Dashboard
# -----------------------------------------------------------------------------
results: Optional[List[VerificationResult]] = st.session_state.get("last_results")
if results:
    total = len(results)
    verified_cnt = sum(1 for r in results if r.status == "Verified")
    inaccurate_cnt = sum(1 for r in results if r.status == "Inaccurate")
    false_cnt = sum(1 for r in results if r.status == "False")
    trust_score = round((verified_cnt / total) * 100) if total else 0

    st.markdown("### Verification Summary")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Claims Checked</div><div class="kpi-val">{total}</div></div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Verified</div><div class="kpi-val" style="color:#059669;">{verified_cnt}</div></div>',
        unsafe_allow_html=True,
    )
    k3.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Inaccurate</div><div class="kpi-val" style="color:#D97706;">{inaccurate_cnt}</div></div>',
        unsafe_allow_html=True,
    )
    k4.markdown(
        f'<div class="kpi-card"><div class="kpi-label">False</div><div class="kpi-val" style="color:#DC2626;">{false_cnt}</div></div>',
        unsafe_allow_html=True,
    )
    k5.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Trust Score</div><div class="kpi-val" style="color:#2563EB;">{trust_score}%</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    # Filter & Downloads
    f_col, d1_col, d2_col = st.columns([2.5, 1, 1])
    with f_col:
        selected_filter = st.radio(
            "Filter by Verdict",
            options=["All", "False", "Inaccurate", "Verified"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with d1_col:
        st.download_button(
            "Download CSV",
            data=build_csv_bytes(results),
            file_name=f"fact_check_{Path(active_name).stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2_col:
        st.download_button(
            "Download Report (.md)",
            data=build_markdown_report(active_name, results).encode("utf-8"),
            file_name=f"fact_check_{Path(active_name).stem}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    filtered = results if selected_filter == "All" else [r for r in results if r.status == selected_filter]
    if not filtered:
        st.info(f"No claims matched the '{selected_filter}' filter.")
    else:
        for res in filtered:
            render_claim_card(res)
