from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_DIR = DATA_DIR / "index"

MAX_RECOMMENDATIONS = 10
RETRIEVAL_CANDIDATES = 15
LLM_TIMEOUT_SECONDS = 18
