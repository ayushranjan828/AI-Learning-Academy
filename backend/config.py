"""Environment-driven configuration. Loaded once at import time."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of backend/
ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")


def _path(env_key: str, default: str) -> Path:
    raw = os.getenv(env_key, default)
    p = Path(raw)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()


def _env(*keys: str, default: str = "") -> str:
    """First non-empty value among `keys` (accepts the VITE_-prefixed names too)."""
    for key in keys:
        val = os.getenv(key, "").strip().strip('"').strip("'")
        if val:
            return val
    return default


AZURE_OPENAI_API_KEY = _env("AZURE_OPENAI_API_KEY", "VITE_AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = _env("AZURE_OPENAI_ENDPOINT", "VITE_AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = _env(
    "AZURE_OPENAI_API_VERSION", "VITE_AZURE_OPENAI_API_VERSION", default="2024-10-21"
)
AZURE_OPENAI_DEPLOYMENT = _env("AZURE_OPENAI_DEPLOYMENT", "VITE_AZURE_OPENAI_DEPLOYMENT")

# On Azure the `model` argument is the *deployment name*. Both tiers default to the one
# chat deployment; point them at separate deployments to split large/small work.
LLM_MODEL_LARGE = _env("LLM_MODEL_LARGE", default=AZURE_OPENAI_DEPLOYMENT)
LLM_MODEL_SMALL = _env("LLM_MODEL_SMALL", default=AZURE_OPENAI_DEPLOYMENT)
# Without an Azure embedding deployment, fall back to Chroma's bundled local model
# (all-MiniLM-L6-v2, ONNX). Re-seed with --reset whenever this changes.
EMBED_MODEL = _env("AZURE_OPENAI_EMBED_DEPLOYMENT", default="local")

CHROMA_PERSIST_DIR = _path("CHROMA_PERSIST_DIR", "./data/chroma_db")
RUNS_DIR = _path("RUNS_DIR", "./data/runs")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "learning_materials")

# Archive of human-approved programmes. The SQLite default is convenient locally but lives
# on the same disposable disk as everything else — point this at Postgres on any host with
# an ephemeral filesystem, or the archive is lost with the container.
DATABASE_URL = os.getenv("DATABASE_URL", "").strip() or f"sqlite:///{ROOT_DIR / 'data' / 'archive.db'}"
# SQLAlchemy still uses the legacy `postgres://` scheme as a hard error; several hosts
# (Render, Heroku) hand out exactly that. Normalise rather than make the user edit it.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Cap on the *automated* quality-gate loop — stops agents revising unattended forever.
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "3"))
# Human rejections get their own budget: a person pressing "reject" is deliberate and
# self-rate-limiting, so it must not be blocked by an exhausted automated budget.
MAX_HUMAN_REJECTIONS = int(os.getenv("MAX_HUMAN_REJECTIONS", "2"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# LLM call resilience
LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "2"))
LLM_BACKOFF_SECONDS = float(os.getenv("LLM_BACKOFF_SECONDS", "2"))
LLM_TIMEOUT_MS = int(os.getenv("LLM_TIMEOUT_MS", "180000"))
# Rate limits (HTTP 429) need their own, much more patient policy: a capacity error is
# transient and retryable, unlike a bad request. The default 2 attempts x 2s gave up after
# ~2 seconds and failed a whole run on a 429.
LLM_RATE_LIMIT_ATTEMPTS = int(os.getenv("LLM_RATE_LIMIT_ATTEMPTS", "5"))
LLM_RATE_LIMIT_BACKOFF_SECONDS = float(os.getenv("LLM_RATE_LIMIT_BACKOFF_SECONDS", "10"))
LLM_RATE_LIMIT_MAX_BACKOFF_SECONDS = float(os.getenv("LLM_RATE_LIMIT_MAX_BACKOFF_SECONDS", "90"))

FRONTEND_DIR = ROOT_DIR / "frontend"

# Re-embed sample_materials/ at startup when the collection is empty. Required on hosts
# with an ephemeral filesystem (e.g. Render's free tier), where the persisted Chroma
# store is discarded on every deploy and restart.
SEED_ON_BOOT = os.getenv("SEED_ON_BOOT", "false").strip().lower() in {"1", "true", "yes"}


def ensure_dirs() -> None:
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def missing_api_key() -> bool:
    return (
        not AZURE_OPENAI_API_KEY
        or AZURE_OPENAI_API_KEY == "your_key_here"
        or not AZURE_OPENAI_ENDPOINT
    )
