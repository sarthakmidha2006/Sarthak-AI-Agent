/**
 * Wire types — these mirror the FastAPI backend's pydantic models
 * (app/models/api_schemas.py) EXACTLY. Do not drift from the backend.
 */

// ----------------------------------------------------------------------------
// Chat
// ----------------------------------------------------------------------------
export type Role = "user" | "assistant";

export interface HistoryTurn {
  role: Role;
  content: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string | null;
  history?: HistoryTurn[] | null;
}

/** A grounding citation surfaced alongside an answer. */
export interface Citation {
  n: number;
  title: string;
  source_type: string;
  url?: string | null;
  snippet: string;
}

/** A tool invocation + result exposed to the client. */
export interface ToolCallView {
  name: string;
  arguments: Record<string, unknown>;
  result: Record<string, unknown>;
}

export interface ChatResponse {
  answer: string;
  session_id: string;
  citations: Citation[];
  tool_calls: ToolCallView[];
  injection_flagged: boolean;
  grounded: boolean | null;
  latency_ms: number;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
}

// ----------------------------------------------------------------------------
// Voice
// ----------------------------------------------------------------------------
export interface VoiceTextRequest {
  message: string;
  session_id?: string | null;
  speak?: boolean;
}

export interface VoiceResponse {
  answer: string;
  session_id: string;
  transcript?: string | null;
  audio_base64?: string | null;
  audio_format: string; // "mp3"
  citations: Citation[];
  tool_calls: ToolCallView[];
  injection_flagged: boolean;
  latency_ms: number;
}

// ----------------------------------------------------------------------------
// Scheduling
// ----------------------------------------------------------------------------
export interface SlotView {
  start: string; // ISO-8601
  end: string; // ISO-8601
}

export interface AvailabilityResponse {
  timezone: string;
  duration_minutes: number;
  slots: SlotView[];
  count: number;
}

export interface BookRequest {
  name: string;
  email: string;
  start_time: string; // ISO-8601
  duration_minutes?: number | null;
  topic?: string | null;
}

export interface BookResponse {
  status: string; // "confirmed" | "unavailable" | ...
  message: string;
  booking_id?: string | null;
  start_time?: string | null;
  end_time?: string | null;
  timezone?: string | null;
  alternatives?: SlotView[] | null;
}

// ----------------------------------------------------------------------------
// Health / diagnostics
// ----------------------------------------------------------------------------
export interface HealthModels {
  chat: string;
  embedding: string;
  stt: string;
  tts: string;
  reranker_provider: string;
  grounding_provider?: string;
}

export interface HealthResponse {
  status: string; // "ok"
  corpus_chunks: number;
  bm25_size: number;
  models: HealthModels;
}

// ----------------------------------------------------------------------------
// Client-side domain models (not from the backend)
// ----------------------------------------------------------------------------
export type MessageStatus = "sending" | "streaming" | "complete" | "error";

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: number; // epoch ms
  status?: MessageStatus;
  citations?: Citation[];
  toolCalls?: ToolCallView[];
  grounded?: boolean | null;
  latencyMs?: number;
  /** Parsed scheduling payload if a tool surfaced availability. */
  availability?: AvailabilityResponse | null;
  /** True when the turn is scheduling-related → render the scheduler card. */
  scheduling?: boolean;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}

export type BackendStatus = "online" | "offline" | "checking";

/** Canonical source-badge kinds derived from backend `source_type`. */
export type SourceKind =
  | "resume"
  | "github"
  | "markdown"
  | "project"
  | "experience"
  | "about"
  | "unknown";

/** Standard shape for surfaced API errors. */
export class ApiError extends Error {
  status: number;
  kind: "offline" | "timeout" | "rate_limit" | "server" | "client";
  detail?: unknown;

  constructor(
    message: string,
    opts: { status?: number; kind: ApiError["kind"]; detail?: unknown } = {
      kind: "server",
    },
  ) {
    super(message);
    this.name = "ApiError";
    this.status = opts.status ?? 0;
    this.kind = opts.kind;
    this.detail = opts.detail;
  }
}
