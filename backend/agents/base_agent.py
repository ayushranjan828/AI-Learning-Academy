"""Shared Azure OpenAI helper: JSON-mode chat completion, schema validation, retry, cost log.

Every agent goes through `call_json()` so that:
  * the orchestrator never has to parse free text,
  * every token spent lands in `state.cost_log`,
  * transient API errors are retried instead of killing the run.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from backend import config, cost_tracker
from backend.schemas import ProgrammeState

from openai import APIConnectionError, AzureOpenAI, InternalServerError

log = logging.getLogger("agents")

T = TypeVar("T", bound=BaseModel)

_client: AzureOpenAI | None = None


class AgentError(RuntimeError):
    """Raised when an agent cannot produce valid output after all retries."""


def client() -> AzureOpenAI:
    global _client
    if _client is None:
        if config.missing_api_key():
            raise AgentError("AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT not set — add them to .env")
        # Retries are handled here (with rate-limit-aware backoff), not by the SDK.
        _client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            timeout=config.LLM_TIMEOUT_MS / 1000,
            max_retries=0,
        )
    return _client


def _schema_text(model: Type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), indent=2)


def _is_rate_limit(exc: Exception) -> bool:
    """Detect HTTP 429 across SDK error shapes (status attr, or the code in the message)."""
    for attr in ("status_code", "status"):
        if getattr(exc, attr, None) == 429:
            return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "capacity exceeded" in text


def _describe(exc: BaseException | None) -> str:
    """Error text with the underlying cause chain — the SDK's 'Connection error.' alone
    hides whether it was DNS, TLS, a proxy or a refused socket."""
    parts: list[str] = []
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts.append(f"{type(exc).__name__}: {exc}".strip(": "))
        exc = exc.__cause__ or exc.__context__
    return " <- ".join(parts) or "unknown error"


def _is_transient(exc: Exception) -> bool:
    """Network drops, timeouts and 5xx are worth waiting out just like a 429."""
    if isinstance(exc, (APIConnectionError, InternalServerError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


def _sleep_for(exc: Exception, attempt: int, rate_limited: bool) -> float:
    """Exponential backoff for rate limits, linear for everything else."""
    if not rate_limited:
        return config.LLM_BACKOFF_SECONDS * attempt
    delay = config.LLM_RATE_LIMIT_BACKOFF_SECONDS * (2 ** (attempt - 1))
    return min(delay, config.LLM_RATE_LIMIT_MAX_BACKOFF_SECONDS)


def _usage(resp: Any) -> tuple[int, int]:
    usage = getattr(resp, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt, completion


def call_json(
    state: ProgrammeState,
    *,
    agent: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    output_model: Type[T],
    temperature: float = 0.3,
) -> T:
    """Call Azure OpenAI in JSON mode and return a validated `output_model` instance.

    Retries on API errors *and* on schema-invalid responses (the retry prompt carries
    the validation error back to the model).
    """
    schema_instruction = (
        f"{system_prompt}\n\n"
        "Respond with a SINGLE JSON object and nothing else — no markdown fences, no prose.\n"
        "It must validate against this JSON Schema:\n"
        f"{_schema_text(output_model)}"
    )
    messages = [
        {"role": "system", "content": schema_instruction},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    # Budget starts at the normal allowance and is raised to the patient rate-limit
    # allowance the first time we actually see a 429.
    budget = max(1, config.LLM_MAX_ATTEMPTS)
    attempt = 0

    while attempt < budget:
        attempt += 1
        raw: Any = ""
        rate_limited = False
        transient = False
        try:
            resp = client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            prompt_tokens, completion_tokens = _usage(resp)
            cost_tracker.log_call(
                state.cost_log,
                agent=agent,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                revision=state.revision_count,
            )
            raw = resp.choices[0].message.content
            if isinstance(raw, list):  # some SDK versions return content chunks
                raw = "".join(getattr(c, "text", "") or "" for c in raw)
            payload = json.loads(_strip_fences(raw))
            return output_model(**payload)

        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = exc
            log.warning("%s: invalid JSON on attempt %s/%s: %s", agent, attempt, budget, exc)
            messages.append({"role": "assistant", "content": str(raw)[:4000]})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response did not match the schema. Error:\n"
                        f"{str(exc)[:1500]}\n"
                        "Return corrected JSON only."
                    ),
                }
            )
        except Exception as exc:  # API/network/rate-limit errors
            last_error = exc
            rate_limited = _is_rate_limit(exc)
            transient = _is_transient(exc)
            if rate_limited or transient:
                budget = max(budget, config.LLM_RATE_LIMIT_ATTEMPTS)
            log.warning(
                "%s: %s on attempt %s/%s: %s",
                agent,
                "RATE LIMIT" if rate_limited else "TRANSIENT" if transient else "API error",
                attempt,
                budget,
                _describe(exc),
            )

        if attempt < budget:
            delay = _sleep_for(last_error, attempt, rate_limited or transient)
            log.info("%s: retrying in %.1fs", agent, delay)
            time.sleep(delay)

    raise AgentError(f"{agent} failed after {attempt} attempts: {_describe(last_error)}")


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: -3]
        t = t.replace("```json", "").replace("```", "")
    return t.strip()


def embed(state: ProgrammeState, texts: list[str], agent: str = "content") -> list[list[float]]:
    """Embed texts with the Azure embedding deployment, retrying and logging cost like chat calls."""
    if config.EMBED_MODEL == "local":
        return _embed_local(texts)
    budget = max(1, config.LLM_MAX_ATTEMPTS)
    last_error: Exception | None = None
    attempt = 0
    while attempt < budget:
        attempt += 1
        rate_limited = False
        transient = False
        try:
            resp = client().embeddings.create(model=config.EMBED_MODEL, input=texts)
            prompt_tokens, completion_tokens = _usage(resp)
            cost_tracker.log_call(
                state.cost_log,
                agent=agent,
                model=config.EMBED_MODEL,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                revision=state.revision_count,
            )
            return [list(d.embedding) for d in resp.data]
        except Exception as exc:
            last_error = exc
            rate_limited = _is_rate_limit(exc)
            transient = _is_transient(exc)
            if rate_limited or transient:
                budget = max(budget, config.LLM_RATE_LIMIT_ATTEMPTS)
            log.warning(
                "embed: %s on attempt %s/%s: %s",
                "RATE LIMIT" if rate_limited else "TRANSIENT" if transient else "error",
                attempt,
                budget,
                _describe(exc),
            )
            if attempt < budget:
                delay = _sleep_for(exc, attempt, rate_limited or transient)
                log.info("embed: retrying in %.1fs", delay)
                time.sleep(delay)
    raise AgentError(f"embedding failed after {attempt} attempts: {_describe(last_error)}")


_local_ef: Any = None


def _embed_local(texts: list[str]) -> list[list[float]]:
    """Chroma's bundled ONNX model — no API call, no cost to log."""
    global _local_ef
    if _local_ef is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        _local_ef = DefaultEmbeddingFunction()
    return [[float(x) for x in v] for v in _local_ef(texts)]


def feedback_block(state: ProgrammeState, owner: str) -> str:
    """Render revision feedback so a re-run agent knows exactly what to fix.

    Human feedback is repeated on EVERY subsequent lap, not just the one it arrived on:
    `revision_feedback` is overwritten by each new quality rejection, so without this the
    human's requirement would be silently dropped after a single revision.
    """
    blocks: list[str] = []

    # Only attribute feedback to the Quality Agent on an actual automated lap. On a human
    # rejection revision_count is 0, and misattributing the human's instruction to the
    # reviewer (as "REVISION 0") both reads as nonsense and hides who is really asking.
    if state.revision_feedback and state.revision_count >= 1:
        lines = "\n".join(f"- {item}" for item in state.revision_feedback)
        blocks.append(
            f"REVISION {state.revision_count} — the Quality Agent rejected the previous "
            f"draft and routed the fix to the {owner} agent. Address every point below "
            f"and say nothing about the revision itself in your output:\n{lines}"
        )

    if state.human_feedback:
        blocks.append(
            "STANDING REQUIREMENT from the human reviewer — this overrides agent "
            "preferences and must remain satisfied in every future revision:\n"
            f"- {state.human_feedback}"
        )

    return ("\n\n" + "\n\n".join(blocks)) if blocks else ""
