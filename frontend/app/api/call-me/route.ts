/**
 * POST /api/call-me — outbound AI callback request.
 *
 * This route handler is the ONLY place provider logic lives. The browser calls
 * it same-origin (no CORS, no exposed secrets); it normalizes the result to
 * `{ success, message? }`. To switch telephony providers later, implement
 * `placeCallback()` below — the UI and `lib/call-service.ts` stay untouched.
 *
 * Default behaviour forwards to the FastAPI backend's `/call-me` endpoint
 * (configurable via env). Swap the body of `placeCallback()` for a direct
 * Vapi / Twilio / Retell outbound-call API call whenever you're ready.
 */

import { NextResponse } from "next/server";

interface IncomingBody {
  name?: unknown;
  phone?: unknown;
}

interface CallResult {
  success: boolean;
  message?: string;
}

// Backend that actually triggers the call. Prefer a server-only var; fall back
// to the public API base so a single Railway URL config keeps working.
const BACKEND_BASE = (
  process.env.CALL_ME_BACKEND_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  ""
).replace(/\/$/, "");

const E164 = /^\+[1-9]\d{7,14}$/;

/**
 * Trigger the outbound call. Replace this implementation to plug in a provider
 * directly (e.g. Vapi `POST /call`, Twilio `calls.create`, Retell outbound) —
 * read provider keys from server env here, never from the client.
 */
async function placeCallback(payload: { name?: string; phone: string }): Promise<CallResult> {
  if (!BACKEND_BASE) {
    // No backend wired yet — fail loudly server-side, gracefully client-side.
    return { success: false, message: "Calling service is not configured." };
  }

  const res = await fetch(`${BACKEND_BASE}/call-me`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    // Outbound trigger should be quick; don't hang the request.
    signal: AbortSignal.timeout(15_000),
  });

  const data = (await res.json().catch(() => null)) as Partial<CallResult> | null;
  if (!res.ok || !data?.success) {
    return {
      success: false,
      message: data?.message ?? "Unable to place call.",
    };
  }
  return { success: true, message: data.message };
}

export async function POST(request: Request): Promise<NextResponse<CallResult>> {
  let body: IncomingBody;
  try {
    body = (await request.json()) as IncomingBody;
  } catch {
    return NextResponse.json({ success: false, message: "Invalid request." }, { status: 400 });
  }

  const phone = typeof body.phone === "string" ? body.phone.trim() : "";
  const name = typeof body.name === "string" ? body.name.trim() : undefined;

  if (!E164.test(phone)) {
    return NextResponse.json(
      { success: false, message: "A valid phone number is required." },
      { status: 400 },
    );
  }

  try {
    const result = await placeCallback({ name, phone });
    return NextResponse.json(result, { status: result.success ? 200 : 502 });
  } catch {
    return NextResponse.json(
      { success: false, message: "Unable to place call. Please try again." },
      { status: 502 },
    );
  }
}
