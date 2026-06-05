# Production Deployment — Railway

This guide deploys the AI Persona FastAPI service to [Railway](https://railway.com)
using the repo's existing `Dockerfile`, with **persistent** Chroma, BM25, and
SQLite storage on a Railway Volume.

It is exact to this codebase:

- ASGI entrypoint: `app.main:app` (uvicorn).
- Container `WORKDIR` is `/app`; all relative paths (`./data/...`) resolve under `/app`.
- Persisted artifacts all live under `/app/data`:
  - Chroma vector store → `./data/chroma` (`CHROMA_PERSIST_DIR`)
  - BM25 index → `./data/bm25/bm25_index.pkl` (`BM25_INDEX_PATH`)
  - SQLite DB → `sqlite:///./data/persona.db` (`DATABASE_URL`)
- Health endpoint: `GET /health` → `{"status":"ok","corpus_chunks":N,...}`.
- Embeddings run **locally** (sentence-transformers); only chat/STT use Groq.

---

## 0. Key facts & one gotcha (read first)

- **Railway injects `$PORT`.** The Dockerfile's `CMD` binds `API_PORT` (8000), which
  Railway does **not** set. You must bind `$PORT` via the start command (handled by
  the included `railway.json`, or set it manually — see Step 4).
- **`/app/data` is a Docker `VOLUME`.** When you attach a Railway Volume there, it
  **shadows** anything baked into the image at that path and starts **empty**. So:
  - *Generated* stores (Chroma/BM25/SQLite) → live in the Volume at `/app/data`. ✅
  - *Source* knowledge files (`data/*.md`, `data/resume/resume.pdf`) are **not**
    copied by the current Dockerfile and would be hidden by the Volume anyway.
    Bake them into a **non-volume** path and point ingestion at it (Step 1).
- **Corpus must be ingested once** into the Volume or `/health` returns
  `corpus_chunks: 0` and answers will be ungrounded (Step 6).

---

## 1. Make source knowledge files available for ingestion (one-time Dockerfile edit)

The corpus is built from the markdown knowledge files and the resume PDF. Because
`/app/data` becomes the Volume mount, copy the *source* files into a separate,
read-only path (`/app/seed`) that the Volume does not shadow.

Add these lines to the `Dockerfile` **after** the existing `COPY app ./app` block
and **before** the `VOLUME` line:

```dockerfile
# --- Source knowledge files for ingestion (read-only; NOT under the volume) ---
# These feed MarkdownSource + ResumeSource at ingest time. They must live outside
# /app/data because that path is replaced by the mounted Railway volume.
COPY data/about.md data/experience.md data/portfolio.md data/projects.md ./seed/
COPY data/resume/resume.pdf ./seed/resume/resume.pdf
```

Then ingestion reads them via two env vars set in Step 3:
`MARKDOWN_DATA_DIR=/app/seed` and `RESUME_PATH=/app/seed/resume/resume.pdf`.

> If you keep your knowledge files only in the Volume instead, skip this edit and
> upload them into the Volume before ingesting — baking them into the image is the
> simpler, reproducible production path.

Commit the change:

```bash
git add Dockerfile railway.json docs/RAILWAY_DEPLOYMENT.md
git commit -m "Add Railway deploy config + bake ingestion source files"
```

---

## 2. Create the Railway project & service

**Option A — GitHub (recommended).**
1. Push the repo to GitHub.
2. Railway dashboard → **New Project** → **Deploy from GitHub repo** → pick this repo.
3. Railway detects the `Dockerfile` and `railway.json` and builds with the Docker builder.

**Option B — CLI.**
```bash
npm i -g @railway/cli
railway login
railway init                 # creates/links a project
railway up                   # builds & deploys from the Dockerfile
```

---

## 3. Configure environment variables

Railway dashboard → your service → **Variables** (or `railway variables --set "K=V"`).

### Required

| Variable | Value | Notes |
|---|---|---|
| `GROQ_API_KEY` | `gsk_...` | From <https://console.groq.com/keys>. Chat + Whisper STT. **Secret.** |
| `DATABASE_URL` | `sqlite:///./data/persona.db` | Resolves to `/app/data/persona.db` (in the Volume). |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | In the Volume. |
| `BM25_INDEX_PATH` | `./data/bm25/bm25_index.pkl` | In the Volume. |
| `MARKDOWN_DATA_DIR` | `/app/seed` | Source `.md` files baked in Step 1. |
| `RESUME_PATH` | `/app/seed/resume/resume.pdf` | Source resume baked in Step 1. |

> `API_HOST` / `API_PORT` are **not** needed — the start command binds `$PORT` directly.

### Recommended (models / cost / behavior)

| Variable | Suggested | Notes |
|---|---|---|
| `OPENAI_CHAT_MODEL` | `llama-3.1-8b-instant` | Groq chat model (free-tier friendly default). |
| `OPENAI_STT_MODEL` | `whisper-large-v3` | Groq STT (only used by `/voice`). |
| `OPENAI_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence-transformers id. |
| `RERANKER_PROVIDER` | `none` | Skip the token-heavy LLM reranker; use fused RRF order. |
| `GROUNDING_CHECK_ENABLED` | `true` | Verify answers against retrieved context. |
| `INJECTION_GUARD_ENABLED` | `true` | Prompt-injection guard. |
| `CORS_ORIGINS` | `["https://your-frontend.example"]` | Lock down from the `["*"]` default. JSON array. |
| `LOG_LEVEL` | `INFO` | |
| `TIMEZONE` | `Asia/Kolkata` (or yours) | Scheduling/availability. |
| `PERSONA_NAME` / `PERSONA_TITLE` / `PERSONA_EMAIL` | your values | Identity in prompts/citations. |

### Optional

| Variable | Value | Notes |
|---|---|---|
| `GITHUB_USERNAME` | handle | Enables GitHub ingestion. Omit to ingest resume + markdown only. |
| `GITHUB_TOKEN` | PAT | Raises GitHub API rate limit during ingestion. **Secret.** |
| `HF_HOME` | `/app/data/.hf_cache` | Persist the downloaded embedding model in the Volume so redeploys don't re-download (~130 MB). |
| `TOP_K_VECTOR` / `TOP_K_BM25` | `4` / `4` | Retrieval breadth (free-tier conservative). |
| `RERANK_CANDIDATES` | `0` | Extra fused candidates for reranker (0 = none). |
| `FINAL_CONTEXT_CHUNKS` | `2` | Chunks injected into the prompt. |

> **Voice (`/voice`) only:** the Piper TTS model (`PIPER_MODEL_PATH`,
> default `./data/piper/en_US-lessac-medium.onnx`) plus its `.onnx.json` must exist
> at that path inside the Volume. Upload them to the Volume, or skip if you only use `/chat`.

---

## 4. Persistent storage — attach a Volume

Railway dashboard → your service → **Settings → Volumes → + New Volume**.

- **Mount path:** `/app/data`

This single Volume persistently holds **all three** stores (Chroma, BM25, SQLite)
because they all live under `/app/data`. The app auto-creates the subdirectories on
boot (`init_db()` creates the SQLite parent dir; Chroma creates its persist dir).

> Railway allows one Volume per service mounted at one path — `/app/data` is exactly
> the right granularity here.

---

## 5. Start command & health check

If you committed `railway.json`, these are already applied. To set manually:

Railway dashboard → **Settings → Deploy**:

- **Custom Start Command:**
  ```
  uvicorn app.main:app --host 0.0.0.0 --port $PORT
  ```
- **Healthcheck Path:** `/health`
- **Healthcheck Timeout:** `300` (seconds) — the first boot warms up the local
  embedding model (downloads weights on first deploy), which can take 30–90s.
- **Restart Policy:** On Failure, max 3 retries.

Deploy (push to GitHub, or `railway up`).

---

## 6. Build the corpus (one-time, into the Volume)

Embeddings are local, so ingestion needs no Groq key — but it **writes into the
Volume**, so it must run **inside the deployed container**, not locally.

```bash
# Open a shell in the running container (CLI must be linked to the service):
railway ssh

# Inside the container — build resume + markdown (+ GitHub if configured):
python -m app.ingestion.run_ingest --reset
```

Expected tail of the JSON summary:

```json
{ "status": "ok", "documents": 5, "chunks": 7, "collection": "persona_corpus", ... }
```

This populates `/app/data/chroma` and `/app/data/bm25/...` in the Volume; it survives
restarts and redeploys. Re-run after updating knowledge files or your GitHub profile.

> No `railway ssh`? Alternative: temporarily set the Custom Start Command to
> `sh -c "python -m app.ingestion.run_ingest --reset && uvicorn app.main:app --host 0.0.0.0 --port $PORT"`,
> deploy once, then revert to the plain `uvicorn` command so you don't re-ingest on every boot.

---

## 7. Expose & verify the health endpoint

1. Railway → **Settings → Networking → Generate Domain** (gives `https://<service>.up.railway.app`).
2. Verify liveness + that the corpus is populated:

```bash
curl -s https://<your-domain>.up.railway.app/health | jq
```

Expected — note `corpus_chunks` is **> 0** (proves persistence + ingestion worked):

```json
{
  "status": "ok",
  "corpus_chunks": 7,
  "bm25_size": 7,
  "models": {
    "chat": "llama-3.1-8b-instant",
    "embedding": "BAAI/bge-small-en-v1.5",
    "stt": "whisper-large-v3",
    "tts": "piper-en_US-lessac-medium",
    "reranker_provider": "none",
    "grounding_provider": "..."
  }
}
```

3. Deeper config check:
```bash
curl -s https://<your-domain>.up.railway.app/diagnostics | jq
# groq_enabled: true, corpus_chunks > 0, embedding_backend: "local"
```

4. End-to-end smoke test:
```bash
curl -s -X POST https://<your-domain>.up.railway.app/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What did you build recently?"}' | jq '.answer, (.citations|length)'
```

### Health verification checklist
- [ ] `GET /health` → HTTP 200, `status: "ok"`.
- [ ] `corpus_chunks > 0` and `bm25_size > 0` (Volume persistence + ingestion confirmed).
- [ ] After a manual **Redeploy**, `/health` still shows the same `corpus_chunks` (Volume persists).
- [ ] `/diagnostics` → `groq_enabled: true`.
- [ ] `POST /chat` returns a grounded answer with citations.

---

## 8. Redeploys & persistence behavior

- Code changes → push to GitHub (or `railway up`); the Volume at `/app/data` is
  retained across deploys, so Chroma/BM25/SQLite (conversations, bookings, query log)
  persist.
- You only re-run Step 6 ingestion when the **source corpus** changes.
- To wipe and rebuild the corpus: `railway ssh` → `python -m app.ingestion.run_ingest --reset`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Deploy "unhealthy" / times out | App bound to 8000, not `$PORT` | Ensure start command uses `--port $PORT` (Step 5). |
| `/health` shows `corpus_chunks: 0` | Corpus not ingested into the Volume | Run Step 6 inside the container. |
| Ingestion: "No markdown files / no resume" | `MARKDOWN_DATA_DIR`/`RESUME_PATH` unset or Step 1 skipped | Bake source files + set the two paths (Steps 1, 3). |
| `corpus_chunks` resets to 0 after redeploy | No Volume, or wrong mount path | Volume must mount at exactly `/app/data` (Step 4). |
| Slow first request / health timeout on boot | Embedding model downloading | Raise healthcheck timeout; set `HF_HOME=/app/data/.hf_cache` to cache it. |
| `/chat` returns "temporarily rate limited" | Groq 429 | Lower request volume or upgrade Groq tier; retrieval still returns citations. |
| 500s on `/chat` | Missing/invalid `GROQ_API_KEY` | Set the secret in Variables (Step 3). |
