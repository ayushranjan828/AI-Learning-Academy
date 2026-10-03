# AI Learning Academy — Multi-Agent System

Five specialised AI agents collaborate — and challenge each other — to design a corporate
AI-upskilling programme from a single manager request. A Quality Agent adversarially reviews
the result and can **reject** it, routing the fix back to the agent that owns it. Nothing is
final until a human approves; approved programmes are archived immutably and exportable as PDF.

```
Manager Request
     │
     ▼
 Orchestrator  (routing logic only — not an LLM)
     │
     ▼
Learner Analysis → Curriculum → Content (RAG) → Assessment → Quality
     ▲                                                          │
     └────────── REJECTED (feedback + target agent) ◄───────────┘
                                │
                          APPROVED
                                │
                                ▼
          Integrity check → Human Approval → Archive + PDF export
```

## Contents

- [Stack](#stack)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [API](#api)
- [Project layout](#project-layout)
- [Deployment (Render)](#deployment-render)
- [Troubleshooting](#troubleshooting)
- [Notes and limits](#notes-and-limits)

## Stack

| Layer | Choice |
|---|---|
| LLM | Azure OpenAI via the `openai` SDK (`AzureOpenAI`) — any chat deployment; JSON mode required |
| Embeddings | Azure embedding deployment, **or** Chroma's bundled local model (`all-MiniLM-L6-v2`, ONNX) when none is configured |
| Backend | FastAPI + uvicorn, Python 3.11 |
| Vector store | ChromaDB, local persistent client |
| Live run state | In-memory dict + one JSON file per run under `data/runs/` |
| Approved archive | SQLAlchemy — SQLite by default, Postgres via `DATABASE_URL` |
| PDF export | ReportLab (pure Python) |
| Frontend | Plain HTML/CSS/JS, `fetch()` only — no build step |

## Quick start

Requires Python 3.11 and an Azure OpenAI resource with a chat deployment.

```powershell
# 1. virtual environment + dependencies
python -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 2. configure
copy .env.example .env
#    → set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
#      AZURE_OPENAI_DEPLOYMENT (your chat deployment name)

# 3. seed the vector store (embeds backend/rag/sample_materials/*.md)
python -m backend.rag.seed_content --reset

# 4. run
python -m uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000/> for the UI and <http://127.0.0.1:8000/docs> for the API.

> **Always start the server with `python -m uvicorn`, not the bare `uvicorn` command.**
> See [Troubleshooting](#troubleshooting) — on some managed Windows machines the `uvicorn.exe`
> launcher produces a process that cannot resolve DNS, and every Azure call fails.

### Using the UI

1. Paste a manager request (or click the example) and submit. The run starts in the
   background; the progress card shows each agent as it runs, plus a full event trace.
2. Watch the **Cost & tokens** card: per-agent and per-revision breakdowns make the price of
   each Quality Agent rejection visible.
3. When the status reaches *awaiting approval*, review the curriculum, content plan and
   assessments tabs. Download any section as PDF.
4. **Approve** to archive the programme, or **Reject** with feedback (optionally naming the
   agent that should fix it) to send it back through the loop.
5. The **Archive** card lists every approved programme, searchable, with PDF export that
   works long after the live run data is gone.

## Configuration

Everything is read from `.env` (see [.env.example](.env.example)). `VITE_`-prefixed variants
of the Azure keys are accepted too.

| Variable | Default | Purpose |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | — | **Required.** |
| `AZURE_OPENAI_ENDPOINT` | — | **Required.** `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | API version string |
| `AZURE_OPENAI_DEPLOYMENT` | — | **Required.** Chat deployment name, used for both tiers unless overridden |
| `LLM_MODEL_LARGE` / `LLM_MODEL_SMALL` | = `AZURE_OPENAI_DEPLOYMENT` | Separate deployments for design/review vs extraction work |
| `AZURE_OPENAI_EMBED_DEPLOYMENT` | *(unset → local)* | Embedding deployment. Leave unset to use Chroma's free local model. Re-seed with `--reset` after changing — vector sizes differ |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector store location |
| `RUNS_DIR` | `./data/runs` | One JSON file per run |
| `DATABASE_URL` | `sqlite:///./data/archive.db` | Archive of approved programmes. Use Postgres on any host with an ephemeral disk |
| `SEED_ON_BOOT` | `false` | Re-embed sample materials at startup if the collection is empty (for ephemeral hosts) |
| `MAX_REVISIONS` | `3` | Cap on the automated quality loop |
| `MAX_HUMAN_REJECTIONS` | `2` | Separate cap on human rejections |
| `RAG_TOP_K` | `3` | Chunks retrieved per module |
| `LLM_MAX_ATTEMPTS` / `LLM_BACKOFF_SECONDS` | `2` / `2` | Retry policy for ordinary API errors |
| `LLM_RATE_LIMIT_ATTEMPTS` / `LLM_RATE_LIMIT_BACKOFF_SECONDS` / `LLM_RATE_LIMIT_MAX_BACKOFF_SECONDS` | `5` / `10` / `90` | Patient policy for 429s and transient network errors |
| `LLM_TIMEOUT_MS` | `180000` | Per-call timeout |

Cost estimates are keyed by **deployment name** in
[cost_tracker.PRICING_USD_PER_MTOK](backend/cost_tracker.py). Add an entry matching each of
your deployments, or costs fall back to a generic rate. They are estimates — verify against
Azure OpenAI pricing before quoting them.

## How it works

### Agents

| Agent | Tier | Reads | Writes |
|---|---|---|---|
| Learner Analysis | small | `manager_request` | cohorts, levels, skill gaps, headcount, region |
| Curriculum | large | `learner_analysis` + revision feedback | modules, objectives, sequencing, duration |
| Content | small + ChromaDB | `curriculum` | per-module material reuse **and flagged content gaps** |
| Assessment | large | `curriculum`, `content_plan` | quizzes, coding exercises, rubrics |
| Quality | large | everything above | `verdict`, `issues[]`, `target_agent` |

Tiering is deliberate: extraction and mapping use the small deployment, design and review
judgment use the large one. Point them at the same deployment and the split costs nothing.

Every agent goes through [base_agent.call_json()](backend/agents/base_agent.py), which
requests JSON mode, validates the response against a Pydantic model, retries on both API
errors and schema violations (feeding the validation error back to the model), and logs token
usage to `state.cost_log`. The orchestrator never parses free text.

### Revision routing

The Quality Agent returns a `target_agent` — the *earliest* agent in the chain that must
change. The orchestrator re-runs from that agent **onward**, not the whole pipeline, so a
content-only defect doesn't pay for a curriculum redesign. See
[orchestrator.run_pipeline()](backend/orchestrator.py).

A rejection needs at least one `high` or `medium` severity issue; `low` findings are advisory
and the orchestrator downgrades a rejection carrying only `low` issues to approved. Automated
rejections are capped at `MAX_REVISIONS` — past that, the run is parked as `failed` with an
escalation note rather than looping.

### Two separate revision budgets

`MAX_REVISIONS` caps the **automated** loop; `MAX_HUMAN_REJECTIONS` caps **human** rejections
independently. The automated loop often spends its full budget before approving — if Reject
drew from the same pool, the approval gate would be dead exactly when a human first sees the
draft. So a human rejection **resets** the automated budget, and the human's feedback becomes a
standing requirement that [feedback_block()](backend/agents/base_agent.py) replays on every
later lap, since `revision_feedback` is overwritten by each new quality rejection.

### Deterministic repairs beat prompt instructions

[curriculum.normalise()](backend/agents/curriculum.py) runs in **code** before the Quality
Agent sees a draft: `total_duration_hours` is set to the real sum of module hours,
prerequisites pointing at later or non-existent modules are stripped, and pathways referencing
unknown `module_id`s are cleaned. Every repair is recorded in `history` as a `normalised` event.

This exists because of an observed failure: the Quality Agent repeatedly raised a
`high`-severity "hours don't add up" issue against a curriculum whose hours *did* add up,
routed the fix to the curriculum agent, which regenerated everything and invalidated the
assessments. Four laps, no convergence, ~$0.40 spent on a hallucinated defect. Arithmetic is
not a judgment task; code guarantees it and the reviewer is told not to recount.

### Agent dependencies and integrity

Dependencies are **one-directional** — a DAG, not a mesh. Every agent writes exactly one
section and reads only upstream ones; no agent calls another.

```
learner_analysis → curriculum → content → assessment ──┐
                                                       ├→ quality
        ◄──────── target_agent (the only back-edge) ────┘
```

Single-writer sections make `run_pipeline(start_from=…)` safe. The cost is **cascade
invalidation**: re-running an upstream agent makes everything downstream stale, which is why
routing to `curriculum` is the most expensive route.

Because consistency is maintained only by execution order, a partially-completed revision could
leave `assessments` referencing a module a later curriculum re-run deleted.
[check_integrity()](backend/orchestrator.py) verifies all sections describe the same
curriculum before the human approval gate opens, and fails the run rather than presenting an
inconsistent draft for sign-off.

### RAG

One Chroma collection, `learning_materials`. [seed_content.py](backend/rag/seed_content.py)
parses the front matter (`title`, `topic`, `format`, `level`, `duration_hours`) of each
markdown file in [backend/rag/sample_materials/](backend/rag/sample_materials/), embeds it,
and stores the metadata alongside.

Embeddings always come from the app, never from Chroma's default function at query time:
either the Azure embedding deployment or, when `AZURE_OPENAI_EMBED_DEPLOYMENT` is unset, the
local MiniLM model. Switching between them changes vector dimensions, so re-seed with `--reset`.

The Content Agent embeds each module's title + objectives in **one batched call**, queries
top-k per module, then asks the model to decide reuse-vs-gap. The seed library deliberately
does **not** cover everything a realistic curriculum needs (no MLOps, evaluation/observability
or fine-tuning material), so the gap-flagging path actually fires.

### Failure recovery

- **Two retry policies.** Ordinary API errors (auth, bad request) get `LLM_MAX_ATTEMPTS` with
  linear backoff — fail fast, they won't fix themselves. Rate limits (429) **and transient
  errors** (connection failures, timeouts, 5xx) get `LLM_RATE_LIMIT_ATTEMPTS` with exponential
  backoff capped at `LLM_RATE_LIMIT_MAX_BACKOFF_SECONDS`.
- Error messages carry the full cause chain (e.g. `Connection error <- getaddrinfo failed`),
  so the UI tells you whether it was DNS, TLS, a proxy or a refused socket.
- Schema-invalid output is retried with the validation error appended to the conversation.
- An agent that still fails parks the run as `failed`, records the error in `history`, and
  surfaces it in the UI — the API request never crashes.
- At startup the server checks it can resolve the Azure host and logs an explicit error with
  the fix if it cannot.

### Approved archive

Live run state ([storage.py](backend/storage.py)) is mutable and lives on disk that an
ephemeral host discards. [archive.py](backend/archive.py) is the opposite: an immutable
snapshot taken the moment a human approves, in a real database. Approval returns `503` rather
than pretending to archive when the database is unreachable. SQLite locally, Postgres in
production — one code path via `DATABASE_URL`.

### PDF export

[pdf_export.py](backend/pdf_export.py) renders the curriculum, content plan and assessments
to A4 PDFs. Each is disabled in the UI until the owning agent has produced its section, and the
endpoint returns `409` rather than an empty document. Every page carries the footer
*"AI-generated draft - requires human review before use"*; the header stamps run id, status,
revision count and generation time so any document traces back to its run. ReportLab's
built-in fonts are Latin-1 only, so `_clean()` maps the arrows, em-dashes and curly quotes
that agents routinely emit.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Key present, model names, vector-store size |
| GET | `/api/programmes` | Recent runs |
| POST | `/api/programmes` | Submit request → `202` + `run_id`; orchestration runs in the background |
| GET | `/api/programmes/{run_id}` | Poll status, agent states, outputs, running cost |
| GET | `/api/programmes/{run_id}/history` | Full agent-by-agent trace |
| GET | `/api/programmes/{run_id}/cost` | Cost by agent, by model, by revision |
| GET | `/api/programmes/{run_id}/export/{section}.pdf` | `curriculum`, `content_plan` or `assessments` as PDF |
| POST | `/api/programmes/{run_id}/approve` | Human sign-off → archived |
| POST | `/api/programmes/{run_id}/reject` | Human rejection + feedback → re-enters the loop |
| GET | `/api/archive?limit=&offset=&q=` | Newest-first page of approved programmes, searchable |
| GET | `/api/archive/{run_id}` | The approved programme exactly as signed off |
| GET | `/api/archive/{run_id}/export/{section}.pdf` | PDF re-rendered from the archived snapshot |

Interactive docs at `/docs`.

## Project layout

```
backend/
  main.py            FastAPI app, routes, startup checks, static frontend
  orchestrator.py    pipeline order, revision routing, caps, integrity check
  schemas.py         Pydantic models for every agent output + ProgrammeState
  config.py          .env → settings
  storage.py         live run state (dict + JSON per run)
  archive.py         durable store of approved programmes (SQLite/Postgres)
  cost_tracker.py    token → USD estimates, roll-ups
  pdf_export.py      ReportLab renderers
  agents/
    base_agent.py    Azure client, call_json(), embed(), retries, feedback_block()
    learner_analysis.py  curriculum.py  content.py  assessment.py  quality.py
  rag/
    chroma_store.py  persistent Chroma client, add/query
    seed_content.py  embed sample_materials/ into the collection
    sample_materials/  5 markdown sources with front matter
frontend/
  index.html  app.js  styles.css
data/                gitignored: chroma_db/, runs/, archive.db
render.yaml          Render blueprint
```

## Deployment (Render)

[render.yaml](render.yaml) deploys a single free-tier web service. Three things matter there:

- **One worker only.** Run state is an in-process dict and the pipeline runs via
  `BackgroundTasks` in the same process; a second worker would serve stale snapshots.
- **`DATABASE_URL` must be Postgres.** The free tier's disk is ephemeral; the default SQLite
  archive would be destroyed on every restart. `postgres://` URLs are normalised automatically.
- **`SEED_ON_BOOT=true`.** The Chroma store is likewise discarded on each deploy, so sample
  materials are re-embedded at startup when the collection is empty.

Set the Azure keys in the Render dashboard; never commit them.

## Troubleshooting

**Every agent fails with `Connection error … getaddrinfo failed`, but the endpoint works from
`curl` or a Python one-liner.**
You started the server with the `uvicorn` command (pip's `Scripts\uvicorn.exe` launcher). On
Windows machines with endpoint security such as Kaspersky, processes spawned through those
launcher stubs — `uvicorn.exe`, `pip.exe`, … — can be denied the system DNS resolver entirely
(even `localhost` won't resolve). Start it with `python -m uvicorn backend.main:app --reload`
instead, and use `python -m pip` for installs. The server logs
`cannot resolve <host> from this process` at startup when this is happening. Only IT can
whitelist the launchers.

**`chroma locked … retrying`** at startup is normal on Windows: antivirus briefly locks
Chroma's migration files; the client retries.

**Costs show `$0.00` or a wrong rate.** Add your deployment names to
`PRICING_USD_PER_MTOK` in [cost_tracker.py](backend/cost_tracker.py).

**Changed the embedding deployment and retrieval returns nothing / errors.** Vector sizes
differ between models. Run `python -m backend.rag.seed_content --reset`.

**Run ends with "Quality Agent still rejecting after N revisions".** That is the designed
`MAX_REVISIONS` escalation, not an error — a human should look at the issues in the trace.

## Notes and limits

- Agent output is a **draft**. A named human reviews and owns anything used for a business
  decision — that's what the approval gate is for.
- Single-process by design; not horizontally scalable without moving run state out of memory.
- `data/chroma_db/`, `data/runs/` and `data/archive.db` are gitignored.
