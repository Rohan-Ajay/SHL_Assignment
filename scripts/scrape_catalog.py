#!/usr/bin/env python
import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/"
ASSESSMENTS_ROOT = "https://www.shl.com/products/assessments/"
SITEMAP_URL = "https://www.shl.com/sitemap.xml"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
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


@dataclass
class CatalogItem:
    name: str
    url: str
    test_type: list[str] = field(default_factory=list)
    duration_minutes: int | None = None
    remote_testing: bool | None = None
    adaptive_irt: bool | None = None
    description: str = ""
    embedding_text: str = ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sleep", type=float, default=0.75)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-sitemap-chunks", type=int, default=6)
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 SHLAssessmentRecommender/0.1"})
    robots = load_robots(session)

    listing_urls = discover_listing_pages(session, robots, args.sleep, args.max_pages)
    detail_urls = discover_detail_urls(session, robots, listing_urls, args.sleep, args.max_pages, args.max_sitemap_chunks)
    items = [scrape_detail(session, robots, url, args.sleep) for url in sorted(detail_urls)]
    items = dedupe_items([item for item in items if item])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump([asdict(item) for item in items], handle, indent=2, ensure_ascii=False)
    print(f"Wrote {len(items)} assessments to {args.output}")


def load_robots(session: requests.Session) -> RobotFileParser:
    robots = RobotFileParser()
    robots.set_url(urljoin(BASE_URL, "/robots.txt"))
    try:
        robots.read()
    except Exception:
        response = session.get(urljoin(BASE_URL, "/robots.txt"), timeout=20)
        robots.parse(response.text.splitlines())
    return robots


def can_fetch(robots: RobotFileParser, url: str) -> bool:
    return robots.can_fetch("*", url)


def get(session: requests.Session, robots: RobotFileParser, url: str, sleep: float) -> BeautifulSoup:
    if not can_fetch(robots, url):
        raise RuntimeError(f"Blocked by robots.txt: {url}")
    time.sleep(sleep)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def discover_listing_pages(
    session: requests.Session,
    robots: RobotFileParser,
    sleep: float,
    max_pages: int | None,
) -> list[str]:
    pages: list[str] = [CATALOG_URL, ASSESSMENTS_ROOT]
    seen: set[str] = set()
    pending = [CATALOG_URL, ASSESSMENTS_ROOT]

    while pending:
        url = pending.pop(0)
        if url in seen:
            continue
        seen.add(url)
        pages.append(url)
        if max_pages and len(pages) >= max_pages:
            break

        soup = get(session, robots, url, sleep)
        for anchor in soup.select("a[href]"):
            href = urljoin(BASE_URL, anchor["href"])
            if is_assessment_url(href) and href not in seen:
                pending.append(href)

    return list(dict.fromkeys(pages))


def discover_detail_urls(
    session: requests.Session,
    robots: RobotFileParser,
    listing_urls: list[str],
    sleep: float,
    max_pages: int | None,
    max_sitemap_chunks: int,
) -> set[str]:
    detail_urls: set[str] = discover_sitemap_assessment_urls(session, robots, sleep, max_pages, max_sitemap_chunks)
    for listing_url in listing_urls:
        soup = get(session, robots, listing_url, sleep)
        for anchor in soup.select("a[href]"):
            href = urljoin(BASE_URL, anchor["href"])
            text = anchor.get_text(" ", strip=True)
            if text and is_assessment_url(href):
                detail_urls.add(normalize_url(href))
    return detail_urls


def discover_sitemap_assessment_urls(
    session: requests.Session,
    robots: RobotFileParser,
    sleep: float,
    max_pages: int | None,
    max_sitemap_chunks: int,
) -> set[str]:
    urls: set[str] = set()
    try:
        index_text = get_text(session, robots, SITEMAP_URL, sleep)
    except Exception:
        return urls

    sitemap_urls = [url for url in re.findall(r"<loc>(.*?)</loc>", index_text) if "l=en_US" in url]
    for sitemap_url in sitemap_urls[:max_sitemap_chunks]:
        if max_pages and len(urls) >= max_pages:
            break
        try:
            sitemap_text = get_text(session, robots, sitemap_url, sleep)
        except Exception:
            continue
        for url in re.findall(r"<loc>(.*?)</loc>", sitemap_text):
            if is_assessment_url(url):
                urls.add(normalize_url(url))
                if max_pages and len(urls) >= max_pages:
                    break
    return urls


def scrape_detail(
    session: requests.Session,
    robots: RobotFileParser,
    url: str,
    sleep: float,
) -> CatalogItem | None:
    soup = get(session, robots, url, sleep)
    page_text = soup.get_text(" ", strip=True)
    if not looks_like_assessment_page(url, page_text):
        return None

    name = extract_name(soup)
    if not name:
        return None

    description = extract_description(soup)
    focused_text = " ".join([url, name, description])
    test_type = extract_test_types(focused_text)
    duration = extract_duration(page_text)
    remote_testing = extract_bool(focused_text, ["remote testing", "remote proctoring", "remote-work", "remote work"])
    adaptive_irt = extract_bool(focused_text, ["adaptive", "irt"])
    embedding_text = " ".join(
        [name, " ".join(test_type), str(duration or ""), description, "remote" if remote_testing else ""]
    ).strip()

    return CatalogItem(
        name=name,
        url=url,
        test_type=test_type,
        duration_minutes=duration,
        remote_testing=remote_testing,
        adaptive_irt=adaptive_irt,
        description=description,
        embedding_text=embedding_text,
    )


def extract_name(soup: BeautifulSoup) -> str:
    heading = soup.find(["h1", "h2"])
    if heading:
        return clean_name(heading.get_text(" ", strip=True))
    title = soup.find("title")
    return clean_name(title.get_text(" ", strip=True).split("|")[0].strip()) if title else ""


def extract_description(soup: BeautifulSoup) -> str:
    candidates = []
    for selector in ["main p", ".content p", "article p", "p"]:
        for paragraph in soup.select(selector):
            text = paragraph.get_text(" ", strip=True)
            if len(text) > 80:
                candidates.append(text)
        if candidates:
            break
    return " ".join(candidates[:4])[:1500]


def extract_test_types(text: str) -> list[str]:
    found = []
    folded = text.casefold()
    inferred = {
        "A": ["cognitive", "aptitude", "ability", "reasoning", "verify"],
        "C": ["competency", "behavioral", "behavioural", "situational judgment", "sjt"],
        "E": ["language", "english"],
        "K": ["skills", "simulation", "coding", "technical", "business skills", "knowledge"],
        "P": ["personality", "opq", "motivation", "mq"],
        "S": ["simulation", "call center", "contact center"],
    }
    for code in ["A", "B", "C", "D", "E", "K", "P", "S"]:
        if any(word in folded for word in inferred.get(code, [])):
            found.append(code)
    return found


def extract_duration(text: str) -> int | None:
    match = re.search(r"(\d{1,3})\s*(?:minutes|mins|min)\b", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_bool(text: str, phrases: list[str]) -> bool | None:
    folded = text.casefold()
    if any(phrase in folded for phrase in phrases):
        if re.search(r"\b(no|not|without)\b.{0,24}(?:" + "|".join(map(re.escape, phrases)) + ")", folded):
            return False
        return True
    return None


def get_text(session: requests.Session, robots: RobotFileParser, url: str, sleep: float) -> str:
    if not can_fetch(robots, url):
        raise RuntimeError(f"Blocked by robots.txt: {url}")
    time.sleep(sleep)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def is_assessment_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != "www.shl.com":
        return False
    path = parsed.path.rstrip("/") + "/"
    if "/products/assessments/" not in path:
        return False
    excluded = ["/resources/", "/training-services/", "/solutions/"]
    return not any(part in path for part in excluded)


def normalize_url(url: str) -> str:
    parsed = urlparse(urljoin(BASE_URL, url))
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/"


def looks_like_assessment_page(url: str, text: str) -> bool:
    folded = f"{url} {text}".casefold()
    signals = [
        "assessment",
        "assessments",
        "test",
        "tests",
        "skills",
        "simulation",
        "personality",
        "cognitive",
        "aptitude",
        "behavioral",
        "behavioural",
        "opq",
        "verify",
    ]
    return is_assessment_url(url) and any(signal in folded for signal in signals)


def clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"\s*\|\s*SHL.*$", "", name)
    return name


def dedupe_items(items: list[CatalogItem]) -> list[CatalogItem]:
    seen: set[str] = set()
    seen_names: set[str] = set()
    deduped: list[CatalogItem] = []
    for item in items:
        item.test_type = list(dict.fromkeys(item.test_type))
        if not is_individual_test_solution(item):
            continue
        key = item.url.rstrip("/").casefold()
        name_key = item.name.casefold()
        if key in seen or name_key in seen_names:
            continue
        seen.add(key)
        seen_names.add(name_key)
        deduped.append(item)
    return deduped


def is_individual_test_solution(item: CatalogItem) -> bool:
    parsed = urlparse(item.url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if path_parts[:2] != ["products", "assessments"]:
        return False
    if len(path_parts) <= 2:
        return False
    if path_parts[-1] in CATEGORY_SLUGS:
        return False

    folded_name = item.name.casefold()
    if any(re.search(pattern, folded_name) for pattern in LANDING_NAME_PATTERNS):
        return False

    return bool(item.name.strip() and item.url.strip())


if __name__ == "__main__":
    main()
