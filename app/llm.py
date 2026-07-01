import json
import os

from app.config import LLM_TIMEOUT_SECONDS
from app.models import Assessment


SYSTEM_PROMPT = """You are an SHL assessment recommender.
Only use assessments from the provided candidate list.
Never invent names or URLs.
If there is insufficient information, ask one clarifying question.
If the user asks about anything outside SHL assessments, refuse politely.
Return only JSON with keys reply, recommendations, and end_of_conversation.
"""


def select_with_llm(messages: list[dict], candidates: list[Assessment]) -> dict | None:
    if os.getenv("LLM_PROVIDER", "").casefold() != "openai":
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)
        candidate_payload = [
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
        ]
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Candidates:\n{json.dumps(candidate_payload)}"},
                *messages,
            ],
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception:
        return None
