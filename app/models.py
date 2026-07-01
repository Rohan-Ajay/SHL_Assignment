from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: list[str] = []
    duration_minutes: int | None = None
    remote_testing: bool | None = None
    adaptive_irt: bool | None = None
    reason: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = []
    end_of_conversation: bool = False


class Assessment(BaseModel):
    name: str
    url: str
    test_type: list[str] = []
    duration_minutes: int | None = None
    remote_testing: bool | None = None
    adaptive_irt: bool | None = None
    description: str = ""
    embedding_text: str = ""

    def searchable_text(self) -> str:
        pieces = [
            self.name,
            " ".join(self.test_type),
            str(self.duration_minutes or ""),
            self.description,
            self.embedding_text,
        ]
        return " ".join(piece for piece in pieces if piece).strip()
