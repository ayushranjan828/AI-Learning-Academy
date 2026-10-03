"""Token/cost accounting.

Prices are USD per 1M tokens and are ESTIMATES for teaching purposes — verify against
Azure OpenAI pricing before using these numbers for anything that matters.
"""

from __future__ import annotations

from typing import Any

PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    # Keyed by *deployment* name — rename these to match your Azure deployments.
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "text-embedding-3-large": {"input": 0.13, "output": 0.00},
}
_FALLBACK = {"input": 1.00, "output": 3.00}


def price_for(model: str) -> dict[str, float]:
    return PRICING_USD_PER_MTOK.get(model, _FALLBACK)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = price_for(model)
    cost = (prompt_tokens / 1_000_000) * p["input"] + (
        completion_tokens / 1_000_000
    ) * p["output"]
    return round(cost, 6)


def log_call(
    cost_log: list[dict],
    *,
    agent: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    revision: int,
) -> dict:
    entry = {
        "agent": agent,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": estimate_cost(model, prompt_tokens, completion_tokens),
        "revision": revision,
        "calls": 1,
    }
    cost_log.append(entry)
    return entry


def _blank(**kw: Any) -> dict[str, Any]:
    base = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "calls": 0,
    }
    base.update(kw)
    return base


def _accumulate(bucket: dict[str, Any], entry: dict) -> None:
    bucket["prompt_tokens"] += entry.get("prompt_tokens", 0)
    bucket["completion_tokens"] += entry.get("completion_tokens", 0)
    bucket["total_tokens"] += entry.get("total_tokens", 0)
    bucket["cost_usd"] = round(bucket["cost_usd"] + entry.get("cost_usd", 0.0), 6)
    bucket["calls"] += entry.get("calls", 1)


def summarise(cost_log: list[dict]) -> dict[str, Any]:
    """Roll the raw call log up by agent, by model and by revision."""
    total = _blank()
    by_agent: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    by_revision: dict[str, dict] = {}

    for entry in cost_log:
        _accumulate(total, entry)
        _accumulate(by_agent.setdefault(entry["agent"], _blank(agent=entry["agent"])), entry)
        _accumulate(by_model.setdefault(entry["model"], _blank(model=entry["model"])), entry)
        rev = f"revision_{entry.get('revision', 0)}"
        _accumulate(by_revision.setdefault(rev, _blank(revision=entry.get("revision", 0))), entry)

    return {
        "total": total,
        "by_agent": list(by_agent.values()),
        "by_model": list(by_model.values()),
        "by_revision": sorted(by_revision.values(), key=lambda r: r["revision"]),
        "calls": cost_log,
        "pricing_note": "Costs are estimates from a local price table (USD per 1M tokens).",
    }
