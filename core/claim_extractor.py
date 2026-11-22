"""Structured factual claim extraction from PDF text using Gemini or OpenRouter."""

from dataclasses import dataclass
from typing import List
import json
import re

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class ExtractedClaimSchema(BaseModel):
    claim_text: str = Field(
        description="Self-contained, atomic factual claim restated clearly with full entity/context."
    )
    exact_quote: str = Field(
        description="Short verbatim snippet from the PDF containing this claim."
    )
    page_number: int = Field(
        description="1-indexed page number where the claim appears."
    )
    category: str = Field(
        description="One of: Statistic, Date/Timeline, Financial, Technical, Corporate/Entity."
    )


class ClaimExtractionResponse(BaseModel):
    claims: List[ExtractedClaimSchema]


@dataclass
class ExtractedClaim:
    claim_id: int
    claim_text: str
    exact_quote: str
    page_number: int
    category: str


EXTRACTION_SYSTEM_PROMPT = """You are a meticulous fact-checking analyst.
Your task is to extract specific, objective, verifiable factual claims from marketing or business PDFs.

Focus strictly on:
- Statistics, percentages, market sizes, user counts, growth rates, and survey findings
- Dates, founding years, launch timelines, and historical milestones
- Financial figures (revenue, valuations, acquisitions, pricing, funding rounds)
- Technical specifications, benchmarks, or regulatory/scientific facts
- Specific corporate facts (headquarters, founders, ownership, partnerships)

Rules:
1. Ignore vague marketing fluff, subjective opinions, future projections clearly labeled as guesses, and generic calls to action.
2. Make every `claim_text` completely self-contained (resolve pronouns like "it" or "the company" to the actual entity name from surrounding text).
3. Keep each claim atomic (one verifiable assertion per item). If a sentence has two distinct stats, split them into two claims.
4. Preserve exact numbers, currencies, units, and dates exactly as stated in the PDF.
5. Do not invent claims that are not in the document.
"""


def _parse_json_payload(raw_text: str) -> dict:
    """Parse JSON safely even if wrapped in markdown code fences."""
    cleaned = (raw_text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        braces = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if braces:
            cleaned = braces.group(0)
    return json.loads(cleaned or '{"claims": []}')


def _extract_via_gemini(
    user_prompt: str,
    api_key: str,
    model_name: str,
) -> dict:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ClaimExtractionResponse,
        ),
    )
    return _parse_json_payload(response.text or '{"claims": []}')


def _extract_via_openrouter(
    user_prompt: str,
    api_key: str,
    model_name: str,
) -> dict:
    schema_hint = (
        '\n\nReturn ONLY a valid JSON object matching this exact schema (no markdown, no extra text):\n'
        '{"claims": [{"claim_text": "...", "exact_quote": "...", "page_number": 1, '
        '"category": "Statistic|Date/Timeline|Financial|Technical|Corporate/Entity"}]}'
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Fact-Check Agent",
    }
    payload = {
        "model": model_name,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt + schema_hint},
        ],
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter extraction request failed (HTTP {resp.status_code}). "
                "Check your OpenRouter API key, credits, and model availability."
            )
        data = resp.json()

    choices = data.get("choices") or []
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    return _parse_json_payload(content)


def extract_claims(
    document_text: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    max_claims: int = 20,
    provider: str = "gemini",
) -> List[ExtractedClaim]:
    """
    Extract verifiable factual claims from formatted PDF text using the selected LLM provider.
    """
    user_prompt = (
        f"Extract up to {max_claims} of the most important, concrete, verifiable factual claims "
        f"(prioritizing stats, dates, financial/technical figures, and factual assertions) "
        f"from the following PDF document.\n\n{document_text}"
    )

    norm_provider = (provider or "gemini").strip().lower()
    if norm_provider == "gemini":
        parsed = _extract_via_gemini(user_prompt, api_key=api_key, model_name=model_name)
    elif norm_provider == "openrouter":
        parsed = _extract_via_openrouter(user_prompt, api_key=api_key, model_name=model_name)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    raw_claims = parsed.get("claims", [])
    results: List[ExtractedClaim] = []
    seen = set()
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        claim_text = (item.get("claim_text") or "").strip()
        if not claim_text:
            continue
        norm = claim_text.lower()
        if norm in seen:
            continue
        seen.add(norm)
        results.append(
            ExtractedClaim(
                claim_id=len(results) + 1,
                claim_text=claim_text,
                exact_quote=(item.get("exact_quote") or claim_text).strip(),
                page_number=int(item.get("page_number") or 1),
                category=(item.get("category") or "Statistic").strip(),
            )
        )
        if len(results) >= max_claims:
            break

    return results
