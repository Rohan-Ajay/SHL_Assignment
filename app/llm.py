import json
import logging
import os
from app.config import LLM_TIMEOUT_SECONDS
from app.models import Assessment

SYSTEM_PROMPT = """You are an SHL assessment recommender.
You will be given a list of candidate assessments. 
You MUST only recommend assessments from that exact list.
Use the EXACT name and URL from the candidate list - do not modify, abbreviate or invent names or URLs.
Return only JSON with keys: reply, recommendations, end_of_conversation.
recommendations is a list of objects with keys: name, url, test_type.
Copy name, url and test_type exactly as they appear in the candidate list.
end_of_conversation is true only when you have provided a final shortlist.
"""

def select_with_llm(messages: list[dict], candidates: list[Assessment]) -> dict | None:
    provider = os.getenv("LLM_PROVIDER", "").casefold()
    logging.warning(f"[LLM] provider='{provider}'")
    if provider == "openai":
        return _call_openai(messages, candidates)
    if provider == "groq":
        return _call_groq(messages, candidates)
    logging.warning("[LLM] no matching provider, returning None")
    return None


def _build_candidate_payload(candidates: list[Assessment]) -> str:
    return json.dumps([
        {
            "name": item.name,
            "url": item.url,
            "test_type": item.test_type,
            "duration_minutes": item.duration_minutes,
            "remote_testing": item.remote_testing,
            "adaptive_irt": item.adaptive_irt,
            "description": item.description,
        }
        for item in candidates
    ])


def _parse_response(content: str) -> dict | None:
    try:
        clean = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(clean)
    except Exception:
        return None


def _call_openai(messages: list[dict], candidates: list[Assessment]) -> dict | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Candidates:\n{_build_candidate_payload(candidates)}"},
                *messages,
            ],
        )
        return _parse_response(response.choices[0].message.content or "{}")
    except Exception as e:
        logging.warning(f"[LLM] OpenAI error: {e}")
        return None


def _call_groq(messages: list[dict], candidates: list[Assessment]) -> dict | None:
    api_key = os.getenv("GROQ_API_KEY")
    logging.warning(f"[LLM] _call_groq entered, key present: {bool(api_key)}")
    if not api_key:
        logging.warning("[LLM] no GROQ_API_KEY")
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Candidates:\n{_build_candidate_payload(candidates)}"},
                *messages,
            ],
        )
        result = _parse_response(response.choices[0].message.content or "{}")
        logging.warning(f"[LLM] Groq result: {result}")
        return result
    except Exception as e:
        logging.warning(f"[LLM] Groq error: {e}")
        return None