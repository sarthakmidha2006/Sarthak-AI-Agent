/**
 * API integration layer.
 *
 * Single source of truth for talking to the FastAPI backend. Every network
 * call goes through `request()`, which normalises errors into `ApiError` so
 * the UI can render elegant, kind-specific states (offline / timeout /
 * rate-limit / server).
 *
 * Real backend routes (verified against app/api/routes/*):
 *   GET  /health
 *   POST /chat
 *   POST /voice            (multipart `audio` OR application/json)
 *   GET  /availability     (query: date_from, date_to, duration_minutes)
 *   POST /book
 */
import {
  ApiError,
  type AvailabilityResponse,
  type BookRequest,
  type BookResponse,
  type ChatRequest,
  type ChatResponse,
  type HealthResponse,
  type VoiceResponse,
} from "@/types";

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

const DEFAULT_TIMEOUT_MS = 45_000;

interface RequestOptions extends Omit<RequestInit, "body"> {
  body?: BodyInit | null;
  timeoutMs?: number;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, headers, ...rest } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...rest,
      headers,
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("The request timed out. Please try again.", {
        kind: "timeout",
      });
    }
    // Network failure → backend unreachable / CORS / DNS.
    throw new ApiError("Cannot reach the backend. Is it running?", {
      kind: "offline",
    });
  }
  clearTimeout(timer);

  if (!res.ok) {
    const detail = await safeJson(res);
    if (res.status === 429) {
      throw new ApiError(
        "The AI service is rate limited. Please try again shortly.",
        { status: 429, kind: "rate_limit", detail },
      );
    }
    if (res.status >= 500) {
      throw new ApiError("The server hit an error. Please try again.", {
        status: res.status,
        kind: "server",
        detail,
      });
    }
    throw new ApiError(
      typeof (detail as { detail?: string })?.detail === "string"
        ? (detail as { detail: string }).detail
        : `Request failed (${res.status}).`,
      { status: res.status, kind: "client", detail },
    );
  }

  return (await res.json()) as T;
}

async function safeJson(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch {
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------
export const api = {
  health(signal?: AbortSignal): Promise<HealthResponse> {
    return request<HealthResponse>("/health", { method: "GET", signal, timeoutMs: 8000 });
  },

  chat(payload: ChatRequest): Promise<ChatResponse> {
    return request<ChatResponse>("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  /** Voice via recorded audio (multipart). Backend transcribes + answers + (optionally) speaks. */
  voiceAudio(audio: Blob, opts: { sessionId?: string | null; speak?: boolean } = {}): Promise<VoiceResponse> {
    const form = new FormData();
    const filename = audio.type.includes("wav") ? "audio.wav" : "audio.webm";
    form.append("audio", audio, filename);
    if (opts.sessionId) form.append("session_id", opts.sessionId);
    form.append("speak", String(opts.speak ?? true));
    return request<VoiceResponse>("/voice", { method: "POST", body: form });
  },

  /** Voice via text (json) — useful for "speak this answer" without a mic. */
  voiceText(message: string, opts: { sessionId?: string | null; speak?: boolean } = {}): Promise<VoiceResponse> {
    return request<VoiceResponse>("/voice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: opts.sessionId ?? null,
        speak: opts.speak ?? true,
      }),
    });
  },

  availability(
    params: { dateFrom?: string; dateTo?: string; durationMinutes?: number } = {},
  ): Promise<AvailabilityResponse> {
    const q = new URLSearchParams();
    if (params.dateFrom) q.set("date_from", params.dateFrom);
    if (params.dateTo) q.set("date_to", params.dateTo);
    if (params.durationMinutes) q.set("duration_minutes", String(params.durationMinutes));
    const qs = q.toString();
    return request<AvailabilityResponse>(`/availability${qs ? `?${qs}` : ""}`, {
      method: "GET",
    });
  },

  book(payload: BookRequest): Promise<BookResponse> {
    return request<BookResponse>("/book", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
};
