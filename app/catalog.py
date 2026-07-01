import json
import re
from pathlib import Path
from difflib import get_close_matches
from urllib.parse import urlparse

from app.config import CATALOG_PATH
from app.models import Assessment


CATEGORY_SLUGS = {
    "assessment-and-development-centers",
    "behavioral-assessments",
    "business-skills",
    "call-center-simulations",
    "coding-simulations",
    "cognitive-assessments",
    "job-focused-assessments",
    "language-evaluation",
    "personality-assessment",
    "skills-and-simulations",
    "technical-skills",
}

LANDING_NAME_PATTERNS = [
    r"\bworld-class talent assessments\b",
    r"\bgo digital and deliver\b",
    r"\bassessment tests for deep insights\b",
    r"\bassessments that predict\b",
    r"\bfast, simple technical skill assessment\b",
]


def load_catalog(path: Path = CATALOG_PATH) -> list[Assessment]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return clean_catalog([Assessment(**item) for item in raw])


def clean_catalog(catalog: list[Assessment]) -> list[Assessment]:
    seen_urls: set[str] = set()
    seen_names: set[str] = set()
    cleaned: list[Assessment] = []
    for item in catalog:
        item.test_type = list(dict.fromkeys(item.test_type))
        if not is_individual_test_solution(item):
            continue

        url_key = item.url.rstrip("/").casefold()
        name_key = item.name.casefold()
        if url_key in seen_urls or name_key in seen_names:
            continue

        seen_urls.add(url_key)
        seen_names.add(name_key)
        cleaned.append(item)
    return cleaned


def is_individual_test_solution(item: Assessment) -> bool:
    parsed = urlparse(item.url)
    if parsed.netloc and parsed.netloc != "www.shl.com":
        return False

    path_parts = [part for part in parsed.path.split("/") if part]
    if path_parts[:2] != ["products", "assessments"]:
        return False
    if len(path_parts) <= 2:
        return False
    if path_parts[-1] in CATEGORY_SLUGS:
        return False
    if any(part in {"resources", "solutions", "training-services"} for part in path_parts):
        return False

    folded_name = item.name.casefold()
    if any(re.search(pattern, folded_name) for pattern in LANDING_NAME_PATTERNS):
        return False

    return bool(item.name.strip() and item.url.strip())


def catalog_by_name(catalog: list[Assessment]) -> dict[str, Assessment]:
    return {item.name.casefold(): item for item in catalog}


def match_assessment_name(name: str, catalog: list[Assessment]) -> Assessment | None:
    by_name = catalog_by_name(catalog)
    direct = by_name.get(name.casefold())
    if direct:
        return direct

    candidates = {item.name: item for item in catalog}
    matches = get_close_matches(name, candidates.keys(), n=1, cutoff=0.86)
    if not matches:
        return None
    return candidates[matches[0]]


def validate_names(names: list[str], catalog: list[Assessment]) -> list[Assessment]:
    seen: set[str] = set()
    valid: list[Assessment] = []
    for name in names:
        item = match_assessment_name(name, catalog)
        if item and item.name.casefold() not in seen:
            valid.append(item)
            seen.add(item.name.casefold())
    return valid
