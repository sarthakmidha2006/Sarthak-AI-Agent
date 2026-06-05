# AI Persona System — "the brain"

A **digital persona** of an engineer that answers questions about that person **only** from a
verified corpus (their resume + their public GitHub) and can **schedule meetings** through real
tool calls. A single shared backend — *the brain* — serves both a **chat** channel and a **voice**
channel. Voice is simply Speech-to-Text → brain → Text-to-Speech wrapped around the *identical*
brain, so both channels share retrieval, grounding, security, and scheduling behaviour with zero
divergence.

The persona never free-associates. Every factual answer is retrieved from a hybrid RAG index,
cited back to its source, and (optionally) checked for grounding before it is returned. Untrusted
retrieved text is treated as **data, never as instructions**, and prompt-injection attempts are
detected, logged, and refused.

---

## Architecture

```mermaid
flowchart TB
    subgraph Clients
      C1[Chat client]
      C2[Voice client]
    end
    C1 -->|POST /chat text| API
    C2 -->|POST /voice audio/text| VOICE[Voice adapter\nSTT / TTS]
    VOICE -->|text| API
    subgraph FastAPI[FastAPI app]
      API[Routes: /chat /voice /availability /book /health]
      BRAIN[PersonaBrain - shared brain]
      GUARD[Prompt-injection guard]
      GROUND[Grounding / hallucination check]
      API --> BRAIN
      BRAIN --> GUARD
      BRAIN --> GROUND
    end
    BRAIN -->|retrieve| RAG
    subgraph RAG[Hybrid RAG]
      EMB[OpenAI embeddings]
      VEC[(ChromaDB vector)]
      BM[BM25 lexical]
      RRF[RRF fusion]
      RR[Reranker]
      EMB --> VEC --> RRF
      BM --> RRF --> RR
    end
    BRAIN -->|tool calls| TOOLS
    subgraph TOOLS[Tools]
      AV[check_availability]
      BK[book_meeting]
    end
    TOOLS --> DB[(SQLite\nbookings/logs)]
    BRAIN -->|chat completion + tools| OAI[OpenAI API]
    subgraph Ingest[Offline ingestion]
      R[Resume PDF]
      GH[GitHub API\nrepos/readmes/commits/source]
      R --> CH[Chunk]
      GH --> CH --> EMB
    end
    BRAIN --> LOG[(QueryLog: latency / citations / grounding)]
    LOG --> EVAL[Eval harness:\nprecision/recall/halluc./booking/latency]
```

> The standalone source for this diagram lives at [`docs/architecture.mmd`](docs/architecture.mmd),
> and a deeper component breakdown lives at [`docs/architecture.md`](docs/architecture.md).

---

## Features mapped to the 7 requirement categories

| # | Requirement category | How the system satisfies it |
|---|---|---|
| 1 | **Grounded Q&A about the persona** | Hybrid RAG (vector + BM25 + RRF + rerank) over the resume and GitHub corpus. Answers are produced only from retrieved context; if the answer is not in context the persona says it does not know. |
| 2 | **Multi-source ingestion** | `ResumeSource` parses the resume PDF with `pypdf`; `GitHubSource` pulls repos, READMEs, commit history, and selected source files via the GitHub REST API. Token-aware chunking with `tiktoken`, embeddings via OpenAI. |
| 3 | **Two channels, one brain** | `PersonaBrain.answer()` is channel-agnostic. `POST /chat` calls it with `channel="chat"`; `POST /voice` runs STT → the same `answer()` (`channel="voice"`) → TTS. No behavioural fork between channels. |
| 4 | **Meeting scheduling via tools** | Two OpenAI function tools — `check_availability` and `book_meeting` — backed by a timezone-aware calendar engine and a SQLite bookings table. The same `book_meeting` powers the `POST /book` REST route. |
| 5 | **Security & safety** | Prompt-injection guard (heuristic regex with optional LLM second opinion), context neutralization, delimiter-wrapped untrusted data, the 5 hard rules enforced in both prompt and code. See [Security](#security). |
| 6 | **Grounding / anti-hallucination** | An LLM judge checks whether the answer's factual claims are supported by the numbered context. Refusals and pure tool confirmations are trivially grounded; judge failure is **fail-open** (logged). |
| 7 | **Evaluation & observability** | Every brain answer writes a `QueryLog` (latency breakdown, citations, retrieved chunk ids, grounding, injection flag). The `eval/` harness computes precision/recall@k, hallucination rate, booking success rate, and latency percentiles. |

---

## Folder structure

```
scaler-ai-agent/
├── app/
│   ├── main.py                 # create_app(), lifespan wiring, middleware, exception handler
│   ├── config.py               # pydantic-settings Settings + get_settings()
│   ├── logging_config.py       # setup_logging()
│   ├── api/
│   │   ├── deps.py             # get_brain / get_llm / get_db / get_settings_dep
│   │   └── routes/             # chat, voice, availability, booking, health
│   ├── brain/
│   │   ├── persona.py          # PersonaBrain (the shared brain) + BrainResponse
│   │   ├── prompts.py          # system prompt, context block, message builder, citations
│   │   └── llm.py              # async AsyncOpenAI wrapper (chat/embed/transcribe/synthesize)
│   ├── rag/
│   │   ├── schemas.py          # Document, Chunk, ScoredChunk, RetrievalResult
│   │   ├── chunking.py         # tiktoken sliding-window chunking
│   │   ├── embeddings.py       # Embedder
│   │   ├── vector_store.py     # ChromaDB PersistentClient wrapper
│   │   ├── bm25_index.py       # rank-bm25 lexical index (picklable)
│   │   ├── hybrid.py           # reciprocal_rank_fusion()
│   │   ├── reranker.py         # LLMReranker / CohereReranker + get_reranker()
│   │   └── retriever.py        # HybridRetriever
│   ├── ingestion/
│   │   ├── base.py             # Source ABC + slug()
│   │   ├── resume.py           # ResumeSource
│   │   ├── github_source.py    # GitHubSource
│   │   ├── pipeline.py         # IngestionPipeline
│   │   └── run_ingest.py       # CLI: python -m app.ingestion.run_ingest
│   ├── tools/
│   │   ├── registry.py         # TOOL_SCHEMAS + dispatch_tool()
│   │   ├── availability.py     # check_availability()
│   │   └── booking.py          # book_meeting()
│   ├── security/
│   │   ├── prompt_guard.py     # PromptGuard, INJECTION_PATTERNS, neutralize_context()
│   │   └── grounding.py        # check_grounding()
│   ├── scheduling/
│   │   └── calendar.py         # Slot, get_available_slots(), is_slot_available()
│   ├── db/
│   │   ├── database.py         # engine, SessionLocal, Base, init_db(), session_scope()
│   │   ├── models.py           # Booking, AvailabilityOverride, Conversation, Message, QueryLog, EvalResult
│   │   └── seed.py             # seed_availability_overrides()
│   └── models/
│       └── api_schemas.py      # pydantic v2 request/response models
├── eval/
│   ├── dataset.py              # GoldItem, BookingScenario, load_dataset()
│   ├── metrics.py              # precision/recall@k, latency, hallucination, booking
│   ├── run_eval.py             # CLI eval harness
│   └── gold.example.json       # gold dataset schema + examples
├── tests/                      # pytest suite (no network; OpenAI mocked)
├── data/
│   ├── resume/                 # drop your resume PDF here
│   ├── chroma/                 # ChromaDB persistent store (generated)
│   └── bm25/                   # BM25 pickle index (generated)
├── docs/
│   ├── BUILD_SPEC.md           # authoritative contract spec
│   ├── architecture.md         # deeper component breakdown + lifecycles
│   └── architecture.mmd        # standalone mermaid source
├── scripts/                    # ingest.sh, run.sh
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Setup

Requires **Python 3.11+** and an **OpenAI API key**. A GitHub token is optional but strongly
recommended (it raises the REST rate limit from 60 to 5000 requests/hour during ingestion).

### 1. Create and activate a virtualenv

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
# or, equivalently:
make install
```

### 3. Configure `.env`

Copy the template and fill in your keys. Every variable below is read by `app/config.py`.

```bash
cp .env.example .env
```

At minimum set:

```dotenv
OPENAI_API_KEY=sk-...            # required for chat, embeddings, STT, TTS
GITHUB_USERNAME=your-github-handle
GITHUB_TOKEN=ghp_...             # optional but recommended (higher rate limit)
PERSONA_NAME="Ada Lovelace"      # quote values with spaces (the person this persona represents)
PERSONA_TITLE="Software Engineer"
PERSONA_EMAIL=ada@example.com
```

<details>
<summary><strong>All configuration variables (defaults shown)</strong></summary>

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `""` | OpenAI auth (required). |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat / tool-calling model. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model. |
| `OPENAI_STT_MODEL` | `whisper-1` | Speech-to-text. |
| `OPENAI_TTS_MODEL` | `tts-1` | Text-to-speech. |
| `OPENAI_TTS_VOICE` | `alloy` | TTS voice. |
| `OPENAI_REQUEST_TIMEOUT` | `60.0` | Per-request timeout (s). |
| `OPENAI_MAX_RETRIES` | `4` | Tenacity retry attempts. |
| `RERANKER_PROVIDER` | `llm` | `llm` or `cohere`. |
| `RERANKER_MODEL` | `gpt-4o-mini` | Model used when provider is `llm`. |
| `COHERE_API_KEY` | _(unset)_ | Required only for the Cohere reranker. |
| `COHERE_RERANK_MODEL` | `rerank-english-v3.0` | Cohere rerank model. |
| `GITHUB_TOKEN` | _(unset)_ | GitHub PAT (optional, raises rate limit). |
| `GITHUB_USERNAME` | `""` | GitHub handle to ingest. |
| `GITHUB_MAX_REPOS` | `8` | Max repos pulled. |
| `GITHUB_MAX_COMMITS_PER_REPO` | `25` | Commits per repo. |
| `GITHUB_MAX_SOURCE_FILES_PER_REPO` | `6` | Source files per repo. |
| `GITHUB_MAX_FILE_BYTES` | `60000` | Max bytes per source file. |
| `GITHUB_INCLUDE_FORKS` | `false` | Whether forks are ingested. |
| `GITHUB_SOURCE_EXTENSIONS` | `.py,.js,.ts,...` | File extensions to ingest. |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB store path. |
| `CHROMA_COLLECTION` | `persona_corpus` | Collection name. |
| `BM25_INDEX_PATH` | `./data/bm25/bm25_index.pkl` | BM25 pickle path. |
| `CHUNK_SIZE_TOKENS` | `512` | Chunk size in tokens. |
| `CHUNK_OVERLAP_TOKENS` | `64` | Chunk overlap in tokens. |
| `EMBEDDING_BATCH_SIZE` | `96` | Embedding batch size. |
| `TOP_K_VECTOR` | `20` | Vector candidates. |
| `TOP_K_BM25` | `20` | BM25 candidates. |
| `RRF_K` | `60` | RRF constant. |
| `RERANK_CANDIDATES` | `16` | Fused candidates sent to reranker. |
| `FINAL_CONTEXT_CHUNKS` | `6` | Chunks placed in the prompt. |
| `PERSONA_NAME` | `the candidate` | Persona identity. |
| `PERSONA_TITLE` | `Software Engineer` | Persona title. |
| `PERSONA_EMAIL` | `candidate@example.com` | Persona contact email. |
| `PERSONA_TAGLINE` | _(see config)_ | Short persona tagline. |
| `TIMEZONE` | `America/Los_Angeles` | Scheduling timezone. |
| `WORKING_DAYS` | `0,1,2,3,4` | Working days (0=Mon). |
| `WORKING_HOURS_START` | `9` | Working hours start (24h). |
| `WORKING_HOURS_END` | `17` | Working hours end (24h). |
| `SLOT_MINUTES` | `30` | Slot step (minutes). |
| `BOOKING_DEFAULT_DURATION` | `30` | Default meeting length (minutes). |
| `BOOKING_HORIZON_DAYS` | `14` | How far ahead availability is computed. |
| `DATABASE_URL` | `sqlite:///./data/persona.db` | SQLAlchemy URL. |
| `INJECTION_GUARD_ENABLED` | `true` | Enable the prompt guard. |
| `INJECTION_LLM_CLASSIFIER` | `false` | Optional LLM second-opinion classifier. |
| `GROUNDING_CHECK_ENABLED` | `true` | Enable the grounding judge. |
| `CHAT_TEMPERATURE` | `0.2` | Chat sampling temperature. |
| `MAX_TOOL_ITERATIONS` | `4` | Max brain tool-loop iterations. |
| `MAX_HISTORY_MESSAGES` | `8` | Conversation history window. |
| `API_HOST` | `0.0.0.0` | Bind host. |
| `API_PORT` | `8000` | Bind port. |
| `CORS_ORIGINS` | `*` | Allowed CORS origins. |
| `LOG_LEVEL` | `INFO` | Root log level. |

</details>

### 4. Add the resume

Drop the persona's resume PDF into `data/resume/` (the ingestion step extracts text per page with
`pypdf`). If no resume is present, ingestion logs a warning and continues with GitHub only.

```bash
cp /path/to/your_resume.pdf data/resume/
```

### 5. Set the GitHub username

Confirm `GITHUB_USERNAME` is set in `.env` (step 3). With no username, the GitHub source warns and
returns nothing — the corpus would then contain only the resume.

### 6. Ingest the corpus

This loads all sources, chunks, embeds, writes the ChromaDB vector store, and builds + pickles the
BM25 index. Use `--reset` to rebuild from scratch.

```bash
python -m app.ingestion.run_ingest --reset
# or
make ingest
# or
./scripts/ingest.sh --reset
```

Useful flags: `--sources resume,github` (default both), `--username <handle>` (override the env
value). On success it prints a JSON summary (`documents`, `chunks`, `by_source_type`, `collection`).

### 7. Run the API

```bash
uvicorn app.main:app --reload
# or
make run
# or
./scripts/run.sh
```

The server listens on `http://localhost:8000`. Interactive OpenAPI docs are at
`http://localhost:8000/docs`.

---

## API reference

Base URL: `http://localhost:8000`. All response bodies are JSON unless noted. Every request is
timed and the server returns an `X-Process-Time-ms` header.

### `POST /chat`

Grounded Q&A on the chat channel.

**Request** (`ChatRequest`): `message` (1–4000 chars, required), `session_id` (optional — a new
UUID is created if omitted), `history` (optional list of `{role, content}` turns where role is
`user` or `assistant`).

```bash
curl -s http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What did you build at your last job?",
    "session_id": null,
    "history": []
  }' | jq
```

**Response** (`ChatResponse`): `answer`, `session_id`, `citations[]`
(`{n, title, source_type, url, snippet}`), `tool_calls[]` (`{name, arguments, result}`),
`injection_flagged`, `grounded` (nullable), `latency_ms`, `prompt_tokens`, `completion_tokens`.

A scheduling request flows through the same endpoint via tool calls:

```bash
curl -s http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Can we meet next Tuesday afternoon? My name is Sam, sam@example.com"}' | jq
```

### `POST /voice`

The voice channel. Accepts **either** form:

- `multipart/form-data` with an `audio` file field (+ optional `session_id`, `speak`). The audio is
  transcribed with Whisper, the transcript is answered by the brain, and the answer is synthesized
  back to MP3.
- `application/json` matching `VoiceTextRequest` (`{message, session_id?, speak?}`) for text-only
  voice-channel testing without uploading audio.

Audio upload:

```bash
curl -s http://localhost:8000/voice \
  -F "audio=@question.wav" \
  -F "speak=true" \
  -F "session_id=" | jq '{answer, transcript, audio_format, latency_ms}'
```

JSON (text) form:

```bash
curl -s http://localhost:8000/voice \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize your top open-source project.", "speak": true}' | jq
```

**Response** (`VoiceResponse`): `answer`, `session_id`, `transcript` (set when audio was uploaded),
`audio_base64` (set when `speak` is true), `audio_format` (`mp3`), `citations[]`, `tool_calls[]`,
`injection_flagged`, `latency_ms`. Decode `audio_base64` to play the spoken reply.

### `GET /availability`

Open meeting slots in the configured timezone and working hours.

**Query params**: `date_from` (`YYYY-MM-DD`, optional), `date_to` (`YYYY-MM-DD`, optional),
`duration_minutes` (optional). Defaults span today → today + `BOOKING_HORIZON_DAYS`.

```bash
curl -s "http://localhost:8000/availability?date_from=2026-06-08&date_to=2026-06-12&duration_minutes=30" | jq
```

**Response** (`AvailabilityResponse`): `timezone`, `duration_minutes`, `slots[]`
(`{start, end}` ISO-8601), `count`.

### `POST /book`

Book a meeting directly over REST (channel `api`). Uses the same `book_meeting` logic as the brain.

**Request** (`BookRequest`): `name` (required), `email` (validated `EmailStr`), `start_time`
(ISO-8601, required), `duration_minutes` (optional), `topic` (optional).

```bash
curl -s http://localhost:8000/book \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sam Rivera",
    "email": "sam@example.com",
    "start_time": "2026-06-09T15:00:00-07:00",
    "duration_minutes": 30,
    "topic": "Backend role chat"
  }' | jq
```

**Response** (`BookResponse`): `status` (`confirmed` / `unavailable`), `message`, and on success
`booking_id`, `start_time`, `end_time`, `timezone`. When the slot is unavailable the response
includes `alternatives[]` (up to 3 nearby open slots).

### `GET /health`

Liveness + corpus stats. Also `GET /` returns simple service info.

```bash
curl -s http://localhost:8000/health | jq
```

**Response**: `{"status": "ok", "corpus_chunks": <int>, "bm25_size": <int>, "models": {...}}`.

---

## Evaluation

The `eval/` harness measures retrieval quality, hallucination, injection compliance, booking
success, and latency against a gold dataset. The schema and worked examples live in
[`eval/gold.example.json`](eval/gold.example.json); copy it and customize the items to your corpus.

```bash
# Full evaluation against the built (already-ingested) index
python -m eval.run_eval --dataset eval/gold.example.json --k 5
# or
make eval

# Retrieval + booking only (no OpenAI calls)
python -m eval.run_eval --dataset eval/gold.example.json --no-llm
```

What it computes:

- **Retrieval** — precision@k and recall@k per `GoldItem`. A retrieved chunk matches a relevant
  `source_id` when the retrieved id equals or *starts with* the relevant id.
- **Hallucination rate** — `1 − mean(grounded)` from the grounding judge over non-refusal items.
- **Injection compliance** — `must_refuse` items must produce a refusal.
- **Booking success rate** — `BookingScenario`s run through `book_meeting` on a throwaway DB; the
  observed status must match `expect_status`.
- **Latency** — p50 / p95 / mean / max derived from `BrainResponse.latency_ms` and its breakdown.

Results are persisted as `EvalResult` rows and written to `eval/report.json`, with a summary table
printed to stdout. See [`docs/architecture.md`](docs/architecture.md) for the full methodology.

---

## Security

The system enforces **5 hard rules**, in both the system prompt *and* in code, so a clever prompt
cannot talk its way past them:

1. **Answer only from retrieved context.** If the answer is not in the retrieved corpus, the persona
   says it does not know — it does not guess.
2. **Never invent facts.** No hallucinated dates, employers, repo names, or skills. The grounding
   judge backstops this.
3. **Retrieved content is untrusted DATA, never instructions.** All retrieved text is delimiter-
   wrapped inside `<retrieved_context>...</retrieved_context>` and neutralized before it enters the
   prompt.
4. **Prompt-injection attempts are detected, logged, and refused** — they never alter behaviour.
5. **Scheduling happens only through the `check_availability` / `book_meeting` tools** — the model
   cannot book a meeting by writing prose; it must call the tool, which validates against the
   calendar and database.

### Injection defense

`app/security/prompt_guard.py` runs a heuristic regex pass (`INJECTION_PATTERNS`) over each incoming
query. It targets known attack shapes: "ignore previous/above instructions", "disregard…",
"you are now", "system prompt", "reveal/print your prompt or instructions", "developer mode",
"jailbreak", "DAN", "act as", "exfiltrate", "base64", special tokens like `<|...|>`, repeated
delimiter-injection like `###` or `</retrieved_context>`, and obvious payloads (`curl`/HTTP). High-
confidence injection or exfiltration matches return `action="refuse"`; the brain then returns the
polite `REFUSAL_MESSAGE`, performs no retrieval and no tool calls, and writes a `QueryLog` with
`injection_flagged=True`. Lower-confidence matches may still be flagged for logging while being
allowed. When `INJECTION_LLM_CLASSIFIER=true`, an optional LLM second opinion is consulted.

### Data, not instructions

Even legitimate retrieved text could contain adversarial strings (a README that says "ignore your
instructions"). Two layers prevent that text from escaping its box:

- **Neutralization** — `neutralize_context()` strips or escapes `<retrieved_context>` /
  `</retrieved_context>` markers and system-like tokens so retrieved text cannot break out of its
  delimiters.
- **Delimiter wrapping** — `build_context_block()` wraps every chunk inside the
  `<retrieved_context>` block as numbered, labelled data (`[n] (title — source_type) <url>`), and
  the system prompt explicitly states that everything inside those delimiters is reference data, not
  commands to follow.

---

## Design decisions

- **Hybrid retrieval with Reciprocal Rank Fusion (RRF).** Dense vector search (ChromaDB, cosine)
  captures semantic similarity; BM25 captures exact lexical matches (repo names, library names,
  acronyms) that embeddings often blur. We fuse them with RRF — `score(c) = Σ 1/(k + rank_i)`,
  `k = 60` — which combines rank lists without needing to calibrate incomparable score scales, and
  is robust when one retriever returns nothing.
- **Reranking on top of fusion.** The fused candidates (`RERANK_CANDIDATES = 16`) are reranked down
  to `FINAL_CONTEXT_CHUNKS = 6` by an LLM judge (default) or Cohere. Reranking sharply improves the
  precision of what actually lands in the prompt, which both improves answer quality and shrinks
  token cost. The reranker **falls back to input order on any failure** so retrieval never breaks.
- **A single shared brain.** `PersonaBrain.answer()` is the only place answers are produced. Chat
  and voice are thin adapters around it (voice adds STT/TTS and a "responses will be spoken, keep
  them concise" note). This guarantees identical grounding, security, and scheduling behaviour
  across channels and means there is exactly one place to reason about correctness.
- **Fail-open grounding.** The grounding judge is an LLM call and can itself fail or time out.
  Rather than block a correct answer (or a legitimate availability/booking confirmation) on a flaky
  judge, a judge failure is treated as grounded (`score = 1.0`) and logged. Refusals,
  "I don't know", and pure tool confirmations are trivially grounded by construction.
- **Token-aware chunking and batched embeddings.** Chunking uses `tiktoken` sliding windows
  (`512`-token windows, `64`-token overlap) so chunks respect model token budgets; embeddings are
  batched (`EMBEDDING_BATCH_SIZE = 96`) to keep ingestion fast and within rate limits.
- **Everything observable.** Each answer writes a `QueryLog` row with the latency breakdown,
  citations, retrieved chunk ids, grounding result, and injection flag — which is exactly what the
  evaluation harness consumes.

---

## Limitations

- **Corpus-bound knowledge.** The persona can only answer what is in the resume + GitHub corpus. If
  the resume is sparse or the GitHub profile is thin, coverage is thin. Re-ingest after updating the
  resume or repos.
- **Ingestion is offline / point-in-time.** The index is a snapshot from the last
  `run_ingest`. New commits, repos, or resume edits are not reflected until you re-ingest.
- **GitHub ingestion is bounded.** It pulls at most `GITHUB_MAX_REPOS` repos, `…_COMMITS_PER_REPO`
  commits, and `…_SOURCE_FILES_PER_REPO` files, and skips forks by default. Very large repos and
  private repos (without a suitable token) are out of scope. Hitting the unauthenticated rate limit
  causes sources to be skipped (logged), not retried forever.
- **Single-calendar scheduling.** Availability is computed from configured working hours, confirmed
  `Booking`s, and `AvailabilityOverride`s in the local SQLite DB. There is no live sync with Google
  Calendar or external systems, and bookings are not transactional across concurrent writers.
- **The grounding judge is heuristic, not a guarantee.** It is an LLM judge and is intentionally
  fail-open, so it reduces — but does not eliminate — the possibility of an unsupported claim slipping
  through. Treat it as a safety net, not a proof.
- **Voice quality depends on OpenAI STT/TTS.** Transcription and synthesis quality (and supported
  audio formats) are bounded by the configured `whisper-1` / `tts-1` models.
- **No authentication on the API.** The endpoints are unauthenticated by default and intended to run
  behind your own gateway/ingress. Do not expose them publicly without adding auth.
- **English-centric.** Tokenization, the default Cohere rerank model (`rerank-english-v3.0`), and the
  prompts assume English.

---

## License & attribution

Built against the contract in [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md). See that file for the
authoritative module/signature/data-model definitions.
# Sarthak-AI-Agent
