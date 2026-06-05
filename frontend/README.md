# Sarthak Midha — AI Persona (Frontend)

A premium, production-ready frontend for the AI Persona Agent. Chat with a
grounded, citation-backed persona; book meetings; and talk via voice — all in a
polished, dark, AI-native UI.

Built with **Next.js 15 (App Router) · TypeScript · Tailwind CSS · shadcn/ui ·
Framer Motion · Lucide**.

---

## Features

- **Chat** — sidebar, new chat, client-persisted history, auto-scroll, message
  timestamps, simulated streaming, typing indicator, stop button.
- **Citations** — beautiful expandable source cards, source badges (Resume /
  GitHub / Markdown / Project / Experience), and a click-to-open detail panel.
- **Persona profile** — identity, skills, AI interests, quick stats.
- **Quick questions** — one-tap suggestion chips.
- **Scheduling** — live availability from `GET /availability`, calendar-style
  slot picker, booking form, and a booking-success modal (with `/book`).
- **Voice mode** — record → `POST /voice` → transcript + answer + audio
  playback, with a recording animation.
- **Health** — `GET /health` polled on load; backend status, model names, and
  corpus size shown in the settings panel.
- **Resilient** — elegant offline / timeout / rate-limit / voice error states.
- **Responsive** — desktop, tablet, and mobile (collapsible sidebar, touch UI).

---

## Architecture

```
frontend/
├── app/
│   ├── globals.css          # Design tokens (dark theme) + utilities
│   ├── layout.tsx           # Root layout, fonts, Tooltip + Toaster providers
│   └── page.tsx             # Composition root — wires hooks + components
├── components/
│   ├── ui/                  # shadcn/ui primitives (see list below)
│   ├── chat/                # chat-area, message-list, message-bubble, input, …
│   ├── citations/           # citations-list, citation-card, detail-panel, badge
│   ├── scheduling/          # scheduling-card, slot-picker, booking-form, modal
│   ├── voice/               # voice-button, voice-modal
│   ├── profile/             # profile-card
│   ├── sidebar/             # sidebar (history + profile + settings)
│   ├── settings/            # settings-panel, health-status
│   ├── layout/              # app-shell (responsive), mobile-header
│   └── common/              # error-state, backend-status-banner
├── hooks/
│   ├── use-chat.ts          # send/stop, simulated streaming, error handling
│   ├── use-voice.ts         # MediaRecorder → /voice → playback
│   ├── use-conversations.ts # localStorage-backed conversation store
│   ├── use-health.ts        # polls /health
│   └── use-media-query.ts   # responsive helpers
├── lib/
│   ├── api.ts               # ★ API integration layer (typed, error-normalised)
│   ├── parse.ts             # defensive tool-call/scheduling parsers
│   ├── constants.ts         # persona, quick questions, source-badge config
│   ├── storage.ts           # conversation persistence
│   └── utils.ts             # cn(), formatters, base64→audio, etc.
└── types/
    └── index.ts             # ★ TS types mirroring the backend schemas
```

### shadcn/ui components used
`button`, `card`, `badge`, `input`, `textarea`, `dialog`, `sheet`,
`scroll-area`, `separator`, `avatar`, `skeleton`, `tooltip`, `sonner` (toaster).

### API integration layer (`lib/api.ts`)
All network access is centralized and matches the **real** backend routes:

| Method | Route | Used by |
|---|---|---|
| `GET`  | `/health` | `useHealth` |
| `POST` | `/chat` | `useChat` |
| `POST` | `/voice` (multipart `audio` or JSON) | `useVoice` |
| `GET`  | `/availability?date_from&date_to&duration_minutes` | `SchedulingCard` |
| `POST` | `/book` | `SchedulingCard` |

Errors are normalised into a single `ApiError` with a `kind`
(`offline` / `timeout` / `rate_limit` / `server` / `client`) so the UI renders
the right state. `/chat` is **not** SSE on the backend, so streaming is
simulated client-side for a live feel.

> History is stored in `localStorage` (the backend exposes no conversation-list
> endpoint). The backend `session_id` is threaded through chat/voice for
> server-side context.

---

## Local development

Prerequisites: **Node.js 18.18+** (or 20+) and the backend running locally.

```bash
cd frontend
cp .env.example .env.local        # set NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev                       # http://localhost:3000
```

Make sure the backend allows the frontend origin via CORS (the backend's
`CORS_ORIGINS` — `["*"]` by default — already permits this).

Scripts: `npm run dev` · `npm run build` · `npm run start` ·
`npm run lint` · `npm run typecheck`.

---

## Deployment

### Frontend → Vercel
1. Push this repo to GitHub.
2. Vercel → **New Project** → import the repo → set **Root Directory** to
   `frontend` (the Next.js app lives in a subfolder).
3. Add an environment variable:
   - **`NEXT_PUBLIC_API_URL`** = your backend URL, e.g.
     `https://your-service.up.railway.app` (no trailing slash).
4. Deploy. Vercel auto-detects Next.js (`framework: nextjs` is also pinned in
   `vercel.json`).

> `NEXT_PUBLIC_*` vars are inlined at build time — after changing it, trigger a
> redeploy.

### Backend → Railway / Render
Deploy the FastAPI app separately (see the repo's `docs/RAILWAY_DEPLOYMENT.md`).
Then point `NEXT_PUBLIC_API_URL` at its public URL and ensure the backend's
`CORS_ORIGINS` includes your Vercel domain (e.g.
`["https://your-app.vercel.app"]`) for production.

---

## Design system

Dark by default. Tokens (in `app/globals.css`):

| Token | Value |
|---|---|
| Background | `#0A0A0A` |
| Cards | `#111111` |
| Border | `#222222` |
| Accent (primary) | `#7C3AED` |
| Secondary | `#A855F7` |

Glassmorphism (`.glass`), soft shadows (`shadow-soft`), an accent glow
(`shadow-glow`), subtle gradient backdrop, and tasteful Framer Motion
throughout (message appearance, citation expansion, sidebar, modals, loading).
