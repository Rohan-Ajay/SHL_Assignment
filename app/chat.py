from app.catalog import validate_names
from app.config import MAX_RECOMMENDATIONS, RETRIEVAL_CANDIDATES
from app.intent import classify_intent, enough_context, extract_compare_names, extract_slots
from app.llm import select_with_llm
from app.models import Assessment, ChatResponse, Message, Recommendation
from app.retriever import HybridRetriever


TECHNICAL_QUERY_TERMS = {
    "backend",
    "coding",
    "developer",
    "engineer",
    "java",
    "javascript",
    "programming",
    "python",
    "software",
    "technical",
}
TECHNICAL_ITEM_TERMS = {
    "backend",
    "coding",
    "developer",
    "engineering",
    "java",
    "javascript",
    "programming",
    "python",
    "software",
}


def refuse_response() -> ChatResponse:
    return ChatResponse(
        reply="I can help with SHL assessment recommendations and comparisons, but I cannot assist with that request.",
        recommendations=[],
    )


def clarify_response() -> ChatResponse:
    return ChatResponse(
        reply="What role, seniority level, and key skills should the SHL assessment cover?",
        recommendations=[],
    )


def chitchat_response() -> ChatResponse:
    return ChatResponse(
        reply="Tell me the role, seniority, skills, and any duration limit, and I will recommend SHL assessments.",
        recommendations=[],
    )


def no_catalog_response() -> ChatResponse:
    return ChatResponse(
        reply="The catalog is empty. Run `python scripts/scrape_catalog.py` to create `data/catalog.json`, then restart the API.",
        recommendations=[],
    )


def low_confidence_response() -> ChatResponse:
    return ChatResponse(
        reply="I could not find a confident catalog match. Please share the role, seniority, target skills, and any duration limit.",
        recommendations=[],
    )


def handle_chat(messages: list[Message], catalog: list[Assessment], retriever: HybridRetriever) -> ChatResponse:
    if not catalog:
        return no_catalog_response()

    intent = classify_intent(messages)
    if intent == "off_topic_or_injection":
        return refuse_response()
    if intent == "chitchat":
        return chitchat_response()
    if intent == "compare":
        return handle_compare(messages, catalog)

    slots = extract_slots(messages)
    if not enough_context(slots):
        return clarify_response()

    results = filter_results_for_slots(retriever.search(slots.query_text(), k=RETRIEVAL_CANDIDATES), slots)
    if not results or not has_retrieval_confidence(results):
        return low_confidence_response()

    candidates = [result.assessment for result in results]
    llm_payload = select_with_llm([message.model_dump() for message in messages], candidates)
    if llm_payload:
        names = [item.get("name", "") for item in llm_payload.get("recommendations", [])]
        valid = validate_names(names, catalog)[:MAX_RECOMMENDATIONS]
        if valid:
            return ChatResponse(
                reply=str(llm_payload.get("reply") or "Here are SHL assessments that match your request."),
                recommendations=[to_recommendation(item, slots.query_text()) for item in valid],
            )

    selected = candidates[:MAX_RECOMMENDATIONS]
    return ChatResponse(
        reply="Here are SHL assessments that best match the role requirements you provided.",
        recommendations=[to_recommendation(item, slots.query_text()) for item in selected],
    )


def handle_compare(messages: list[Message], catalog: list[Assessment]) -> ChatResponse:
    names = extract_compare_names(messages, [item.name for item in catalog])
    matched = validate_names(names, catalog)
    if len(matched) < 2:
        return ChatResponse(
            reply="Which SHL assessments would you like me to compare? Please provide two or more assessment names.",
            recommendations=[],
        )

    summary_parts = []
    for item in matched[:MAX_RECOMMENDATIONS]:
        duration = f"{item.duration_minutes} minutes" if item.duration_minutes else "duration not listed"
        types = ", ".join(item.test_type) if item.test_type else "type not listed"
        summary_parts.append(f"{item.name}: {duration}, test type {types}.")

    return ChatResponse(
        reply=" ".join(summary_parts),
        recommendations=[to_recommendation(item, "comparison request") for item in matched[:MAX_RECOMMENDATIONS]],
    )


def filter_results_for_slots(results, slots):
    filtered = []
    for result in results:
        item = result.assessment
        if slots.max_duration_minutes and item.duration_minutes and item.duration_minutes > slots.max_duration_minutes:
            continue
        if slots.test_types and item.test_type and not set(slots.test_types).intersection(item.test_type):
            continue
        filtered.append(result)
    return filtered


def query_requires_technical_fit(slots) -> bool:
    if "P" in slots.test_types:
        return False
    text = slots.query_text().casefold()
    return any(term in text for term in TECHNICAL_QUERY_TERMS)


def item_has_technical_fit(item: Assessment) -> bool:
    text = item.searchable_text().casefold()
    return any(term in text for term in TECHNICAL_ITEM_TERMS)


def has_retrieval_confidence(results) -> bool:
    return len(results) > 0


def to_recommendation(item: Assessment, query: str) -> Recommendation:
    reason = build_reason(item, query)
    return Recommendation(
        name=item.name,
        url=item.url,
        test_type=item.test_type,
        duration_minutes=item.duration_minutes,
        remote_testing=item.remote_testing,
        adaptive_irt=item.adaptive_irt,
        reason=reason,
    )


def build_reason(item: Assessment, query: str) -> str:
    if item.duration_minutes:
        return f"Matches the request context and is listed as a {item.duration_minutes}-minute assessment."
    if item.description:
        return item.description[:220].strip()
    return f"Matches the request context: {query}."
