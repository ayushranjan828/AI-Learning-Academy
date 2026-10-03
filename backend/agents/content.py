"""Content Agent — RAG over the learning-materials library.

For each curriculum module: embed its objectives with the Azure embedding deployment, retrieve top-k
from ChromaDB, then ask a cheap model to decide reuse-vs-gap. One LLM call for the
whole mapping keeps cost down; retrieval is per-module for precision.
"""

from __future__ import annotations

import json

from backend import config
from backend.agents.base_agent import call_json, embed, feedback_block
from backend.rag import chroma_store
from backend.schemas import ContentPlan, ProgrammeState

AGENT = "content"

SYSTEM = """You are the Content Agent. You map an existing learning-material library
onto a curriculum and — critically — you flag what the library does NOT cover.

You will receive, per module, the top retrieval hits from a vector search with a cosine
distance (lower = more similar).

Rules:
- A hit is only reusable if it genuinely teaches that module's objectives. Topical
  adjacency is not coverage. Distances above roughly 0.30 are usually weak matches.
- For each module output recommended_materials as a list of
  {source, topic, level, why_relevant}. why_relevant must name the specific module
  objective the material covers.
- If nothing suitable was retrieved, set gap=true, leave recommended_materials empty,
  and write gap_note describing what must be authored (format, depth, level).
- Set gap=true ONLY when a stated learning objective has no usable coverage. If the
  retrieved material covers every objective — even if you would ideally want more
  practice material — gap must be false. Do not flag a gap merely because the material
  is at a different level or because you would like extra labs.
- A PARTIAL gap is allowed: list the usable material AND set gap=true, but gap_note must
  quote the specific uncovered objective. Never set gap=true without naming what is
  missing.
- Be discriminating: a plan where every module is a gap is as useless as one where none
  is. Expect a realistic mix.
- content_gaps: one plain-language line per missing asset, for the L&D backlog. Only
  include entries that correspond to a module you marked gap=true.
- reuse_summary: how many modules are fully covered, partially covered, and uncovered.
- Never invent a material source that was not in the retrieval results."""


def run(state: ProgrammeState) -> ProgrammeState:
    curriculum = state.curriculum or {}
    modules = curriculum.get("modules", [])

    library_size = chroma_store.count()
    retrieval: list[dict] = []

    if modules:
        # One batched embed call for all module queries — cheaper than one call each.
        queries = [
            f"{m.get('title', '')}. {m.get('topic', '')}. "
            f"Objectives: {'; '.join(m.get('learning_objectives', []))}"
            for m in modules
        ]
        vectors = embed(state, queries, agent=AGENT) if library_size else [[]] * len(queries)

        for module, vector in zip(modules, vectors):
            hits = chroma_store.query(vector, config.RAG_TOP_K) if library_size else []
            retrieval.append(
                {
                    "module_id": module.get("module_id"),
                    "module_title": module.get("title"),
                    "target_level": module.get("target_level"),
                    "learning_objectives": module.get("learning_objectives", []),
                    "retrieved": [
                        {
                            "source": h["source"],
                            "topic": h["topic"],
                            "level": h["level"],
                            "format": h["format"],
                            "distance": h["distance"],
                            "excerpt": h["excerpt"][:600],
                        }
                        for h in hits
                    ],
                }
            )

    user = (
        f"Library size: {library_size} chunk(s) in collection '{config.COLLECTION_NAME}'.\n\n"
        "Per-module retrieval results:\n"
        f"{json.dumps(retrieval, indent=2)}"
        + feedback_block(state, "content")
    )

    result = call_json(
        state,
        agent=AGENT,
        model=config.LLM_MODEL_SMALL,
        system_prompt=SYSTEM,
        user_prompt=user,
        output_model=ContentPlan,
        temperature=0.2,
    )
    plan = result.model_dump()
    plan["library_chunks_searched"] = library_size
    plan["top_k"] = config.RAG_TOP_K
    state.content_plan = plan
    return state
