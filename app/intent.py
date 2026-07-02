import re
from dataclasses import dataclass, field

from app.models import Message


OFF_TOPIC_PATTERNS = [
    r"\bignore (all )?(previous|prior|above) instructions\b",
    r"\bdisregard (all )?(previous|prior|above) instructions\b",
    r"\bwhat is the weather\b",
    r"\blegal advice\b",
    r"\bsalary negotiation\b",
    r"\bwrite (a )?contract\b",
]

COMPARE_PATTERNS = [r"\bcompare\b", r"\bdifference between\b", r"\bversus\b", r"\bvs\.?\b"]
REFINE_PATTERNS = [r"\bactually\b", r"\binstead\b", r"\badd\b", r"\bremove\b", r"\bmake it\b", r"\bunder \d+"]

SKILL_WORDS = [
    "java",
    "python",
    "javascript",
    "sql",
    "sales",
    "customer service",
    "personality",
    "cognitive",
    "reasoning",
    "leadership",
    "manager",
    "graduate",
    "developer",
    "engineer",
    "analyst",
    "finance",
    "excel",
]

TYPE_MAP = {
    "ability": "A",
    "biodata": "B",
    "competency": "C",
    "development": "D",
    "english": "E",
    "knowledge": "K",
    "personality": "P",
    "simulation": "S",
}


@dataclass
class Slots:
    role: str | None = None
    seniority: str | None = None
    skills: list[str] = field(default_factory=list)
    test_types: list[str] = field(default_factory=list)
    max_duration_minutes: int | None = None
    remote_testing: bool | None = None
    adaptive_irt: bool | None = None

    def query_text(self) -> str:
        parts = [
            self.role or "",
            self.seniority or "",
            " ".join(self.skills),
            " ".join(self.test_types),
            f"under {self.max_duration_minutes} minutes" if self.max_duration_minutes else "",
            "remote testing" if self.remote_testing else "",
            "adaptive irt" if self.adaptive_irt else "",
        ]
        return " ".join(part for part in parts if part).strip()


def transcript(messages: list[Message]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def latest_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def classify_intent(messages: list[Message]) -> str:
    latest = latest_user_text(messages).casefold()
    if not latest:
        return "clarify_needed"
    if any(re.search(pattern, latest) for pattern in OFF_TOPIC_PATTERNS):
        return "off_topic_or_injection"
    if any(re.search(pattern, latest) for pattern in COMPARE_PATTERNS):
        return "compare"
    if any(re.search(pattern, latest) for pattern in REFINE_PATTERNS) and len(messages) > 1:
        return "refine"
    if any(word in latest for word in ["hi", "hello", "thanks"]) and not any(skill in latest for skill in SKILL_WORDS):
        return "chitchat"
    return "recommend"


def extract_slots(messages: list[Message]) -> Slots:
    raw_text = transcript(messages)
    text = raw_text.casefold()
    slots = Slots()

    duration_matches = re.findall(r"(?:under|less than|within|max(?:imum)?|below)\s+(\d{1,3})\s*(?:min|mins|minutes)?", text)
    if duration_matches:
        slots.max_duration_minutes = int(duration_matches[-1])

    for seniority in ["entry", "junior", "graduate", "mid", "senior", "lead", "manager"]:
        if seniority in text:
            slots.seniority = seniority
            break

    for skill in SKILL_WORDS:
        if skill in text and skill not in slots.skills:
            slots.skills.append(skill)

    for phrase, code in TYPE_MAP.items():
        if phrase in text and code not in slots.test_types:
            slots.test_types.append(code)
    for code in ["A", "B", "C", "D", "E", "K", "P", "S"]:
        if re.search(rf"\b{code}\b", raw_text) and code not in slots.test_types:
            slots.test_types.append(code)

    if re.search(r"\bremote\b|\bonline\b", text):
        slots.remote_testing = True
    if re.search(r"\badaptive\b|\birt\b", text):
        slots.adaptive_irt = True

    role_match = re.search(
        r"(?:assessment|test)\s+for\s+(?:an|a|the)?\s*([a-z0-9 +#.-]{3,60}?)(?:\s+(?:assessment|test|under|with|who|that)|[,.]|$)",
        text,
    ) or re.search(
        r"(?:for|hiring|hire|need)\s+(?:an|a|the)?\s*([a-z0-9 +#.-]{3,60}?)(?:\s+(?:assessment|test|under|with|who|that)|[,.]|$)",
        text,
    )
    if role_match:
        slots.role = role_match.group(1).strip()

    return slots


def enough_context(slots: Slots) -> bool:
    return bool(slots.skills or slots.role or slots.test_types)


def extract_compare_names(messages: list[Message], known_names: list[str]) -> list[str]:
    latest = latest_user_text(messages)
    latest_folded = latest.casefold()
    matches = [name for name in known_names if name.casefold() in latest_folded]
    if len(matches) >= 2:
        return matches

    pieces = re.split(r"\b(?:compare|and|versus|vs\.?|difference between)\b", latest, flags=re.IGNORECASE)
    return [piece.strip(" ?.,:;\"'") for piece in pieces if len(piece.strip()) > 2]
