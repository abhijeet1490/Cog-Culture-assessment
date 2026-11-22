"""Live web fact-verification engine using Gemini Search Grounding or OpenRouter Web Search."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple
import json
import re

import httpx
from google import genai
from google.genai import types

from core.claim_extractor import ExtractedClaim

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class VerificationSource:
    title: str
    url: str


@dataclass
class VerificationResult:
    claim_id: int
    page_number: int
    category: str
    claim_text: str
    exact_quote: str
    status: str  # "Verified", "Inaccurate", or "False"
    confidence: str  # "High", "Medium", "Low"
    explanation: str
    correct_fact: str
    sources: List[VerificationSource] = field(default_factory=list)


VERIFICATION_PROMPT_TEMPLATE = """Today's date is {today_str}.
You are an investigative fact-checker verifying a claim extracted from a marketing/business PDF against live web search results.

Claim to verify:
"{claim_text}"

Original quote from PDF (Page {page_number}):
"{exact_quote}"

Instructions:
1. Search the live web thoroughly to verify whether this claim is accurate as of {today_str}.
2. Classify the claim into EXACTLY ONE of these three statuses:
   - "Verified": The claim accurately matches current, authoritative live web evidence (minor rounding within ~5% is acceptable if directionally accurate and not outdated).
   - "Inaccurate": The claim is partially wrong, misleading, numerically distorted, or relies on significantly outdated statistics presented as current facts.
   - "False": The claim is fabricated, contradicted outright by authoritative sources, describes an event/acquisition/milestone that never happened, or has no credible supporting evidence.
3. Whenever the status is "Inaccurate" or "False", you MUST provide the accurate, up-to-date "real" fact in `correct_fact` (including the real number, date, or correction). If "Verified", briefly state the confirming fact in `correct_fact`.
4. Return ONLY a valid JSON object with no markdown fences and no extra prose, using this exact schema:
{{
  "status": "Verified" | "Inaccurate" | "False",
  "confidence": "High" | "Medium" | "Low",
  "explanation": "1-2 concise sentences explaining why the claim received this verdict based on live web evidence.",
  "correct_fact": "The verified, up-to-date real fact (with exact figure/date) correcting or confirming the claim."
}}
"""


def _extract_json_object(text: str) -> Dict[str, str]:
    """Robustly parse a JSON object from model output even if wrapped in code fences."""
    cleaned = (text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(0)
    return json.loads(cleaned)


def _extract_grounding_sources(response) -> List[VerificationSource]:
    """Pull real clickable URLs and titles from Gemini Google Search grounding metadata."""
    sources: List[VerificationSource] = []
    seen_urls = set()
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return sources
        metadata = getattr(candidates[0], "grounding_metadata", None)
        chunks = getattr(metadata, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if not web:
                continue
            url = (getattr(web, "uri", None) or "").strip()
            title = (getattr(web, "title", None) or url).strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(VerificationSource(title=title, url=url))
    except Exception:
        pass
    return sources[:5]


def _extract_openrouter_sources(message: dict) -> List[VerificationSource]:
    """Extract standardized url_citation annotations returned by OpenRouter web search."""
    sources: List[VerificationSource] = []
    seen_urls = set()
    annotations = message.get("annotations") or []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        if ann.get("type") != "url_citation":
            continue
        cite = ann.get("url_citation") or {}
        url = (cite.get("url") or "").strip()
        title = (cite.get("title") or url).strip()
        if url.startswith(("http://", "https://")) and url not in seen_urls:
            seen_urls.add(url)
            sources.append(VerificationSource(title=title, url=url))
    return sources[:5]


def _normalize_status(raw_status: str) -> str:
    s = (raw_status or "").strip().lower()
    if "verif" in s and "unverif" not in s:
        return "Verified"
    if "inacc" in s or "outdat" in s or "partial" in s or "mislead" in s:
        return "Inaccurate"
    return "False"


def _verify_via_gemini(
    prompt: str,
    api_key: str,
    model_name: str,
) -> Tuple[Dict[str, str], List[VerificationSource]]:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    data = _extract_json_object(response.text or "")
    sources = _extract_grounding_sources(response)
    return data, sources


def _verify_via_openrouter(
    prompt: str,
    api_key: str,
    model_name: str,
) -> Tuple[Dict[str, str], List[VerificationSource]]:
    """
    Verify a claim through OpenRouter with live web search enabled.
    Uses OpenRouter's web plugin (native Google Search for Google models, or Exa fallback),
    falling back to the openrouter:web_search server tool if required.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Fact-Check Agent",
    }
    base_messages = [{"role": "user", "content": prompt}]
    payload_attempts = [
        {
            "model": model_name,
            "temperature": 0.0,
            "plugins": [{"id": "web", "max_results": 5}],
            "messages": base_messages,
        },
        {
            "model": model_name,
            "temperature": 0.0,
            "tools": [{"type": "openrouter:web_search"}],
            "messages": base_messages,
        },
    ]

    last_status = None
    data_json = None
    with httpx.Client(timeout=60.0) as client:
        for payload in payload_attempts:
            resp = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            if resp.status_code < 400:
                data_json = resp.json()
                break
            last_status = resp.status_code
            # Auth or credit errors should fail immediately without retrying another payload
            if resp.status_code in {401, 402, 403, 429}:
                break

    if data_json is None:
        raise RuntimeError(
            f"OpenRouter web-grounded verification failed (HTTP {last_status}). "
            "Check your OpenRouter API key, account credits, and web search availability."
        )

    choices = data_json.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content") or ""
    parsed = _extract_json_object(content)
    sources = _extract_openrouter_sources(message)
    return parsed, sources


def verify_single_claim(
    claim: ExtractedClaim,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    provider: str = "gemini",
) -> VerificationResult:
    """Verify a single extracted claim against live web search via the selected provider."""
    today_str = date.today().strftime("%B %d, %Y")
    prompt = VERIFICATION_PROMPT_TEMPLATE.format(
        today_str=today_str,
        claim_text=claim.claim_text,
        page_number=claim.page_number,
        exact_quote=claim.exact_quote,
    )
    norm_provider = (provider or "gemini").strip().lower()

    try:
        if norm_provider == "gemini":
            data, sources = _verify_via_gemini(prompt, api_key=api_key, model_name=model_name)
        elif norm_provider == "openrouter":
            data, sources = _verify_via_openrouter(prompt, api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        status = _normalize_status(data.get("status", "False"))
        confidence = (data.get("confidence") or "Medium").strip().capitalize()
        if confidence not in {"High", "Medium", "Low"}:
            confidence = "Medium"

        explanation = (data.get("explanation") or "Verified against live web search results.").strip()
        correct_fact = (data.get("correct_fact") or "No correction provided.").strip()

        # Guardrail: never present an ungrounded model response as a Verified fact
        if not sources and status == "Verified":
            status = "False"
            confidence = "Low"
            explanation = (
                "No live web search citations were returned to ground this claim. "
                "Ungrounded model output is not marked as Verified."
            )
            correct_fact = (
                "Live web evidence could not be retrieved for this claim; verify manually "
                "or confirm web search grounding is enabled on your API account."
            )

        return VerificationResult(
            claim_id=claim.claim_id,
            page_number=claim.page_number,
            category=claim.category,
            claim_text=claim.claim_text,
            exact_quote=claim.exact_quote,
            status=status,
            confidence=confidence,
            explanation=explanation,
            correct_fact=correct_fact,
            sources=sources,
        )
    except Exception as exc:
        safe_err = re.sub(r"(AIza[0-9A-Za-z\-_]+|sk-or-[0-9A-Za-z\-_]+|sk-[0-9A-Za-z\-_]+)", "[REDACTED]", str(exc))
        return VerificationResult(
            claim_id=claim.claim_id,
            page_number=claim.page_number,
            category=claim.category,
            claim_text=claim.claim_text,
            exact_quote=claim.exact_quote,
            status="False",
            confidence="Low",
            explanation=f"Verification could not complete due to an API error: {safe_err}",
            correct_fact="Unable to retrieve live web evidence for this claim.",
            sources=[],
        )


def verify_claims_parallel(
    claims: List[ExtractedClaim],
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    max_workers: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    provider: str = "gemini",
) -> List[VerificationResult]:
    """Verify multiple claims concurrently while preserving original claim order."""
    if not claims:
        return []

    results_map: Dict[int, VerificationResult] = {}
    completed = 0
    total = len(claims)

    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
        future_to_id = {
            executor.submit(verify_single_claim, claim, api_key, model_name, provider): claim.claim_id
            for claim in claims
        }
        for future in as_completed(future_to_id):
            cid = future_to_id[future]
            results_map[cid] = future.result()
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return [results_map[c.claim_id] for c in claims]
