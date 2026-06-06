/**
 * Call-Me service — the UI's single, provider-agnostic entry point for
 * requesting an outbound AI callback.
 *
 * The UI only ever calls {@link requestCallback}. The actual telephony provider
 * (Vapi / Twilio / Retell outbound) is wired behind the same-origin
 * `POST /api/call-me` route handler, so providers can be swapped server-side
 * without touching any component. Phone numbers are normalized to E.164 here
 * before they ever leave the browser.
 */

import type { CallbackRequest, CallbackResult } from "@/types";

/** Same-origin Next.js route handler (keeps secrets server-side, avoids CORS). */
const CALL_ME_ENDPOINT = "/api/call-me";
const REQUEST_TIMEOUT_MS = 20_000;

/** A dialable country with its E.164 calling code. India first (primary audience). */
export interface CountryCode {
  /** ISO-3166 alpha-2, used as a stable key. */
  iso: string;
  name: string;
  /** E.164 calling code including the leading "+". */
  dial: string;
  flag: string;
}

export const COUNTRY_CODES: CountryCode[] = [
  { iso: "IN", name: "India", dial: "+91", flag: "🇮🇳" },
  { iso: "US", name: "United States", dial: "+1", flag: "🇺🇸" },
  { iso: "GB", name: "United Kingdom", dial: "+44", flag: "🇬🇧" },
  { iso: "CA", name: "Canada", dial: "+1", flag: "🇨🇦" },
  { iso: "AU", name: "Australia", dial: "+61", flag: "🇦🇺" },
  { iso: "AE", name: "United Arab Emirates", dial: "+971", flag: "🇦🇪" },
  { iso: "SG", name: "Singapore", dial: "+65", flag: "🇸🇬" },
  { iso: "DE", name: "Germany", dial: "+49", flag: "🇩🇪" },
  { iso: "FR", name: "France", dial: "+33", flag: "🇫🇷" },
  { iso: "IE", name: "Ireland", dial: "+353", flag: "🇮🇪" },
  { iso: "NL", name: "Netherlands", dial: "+31", flag: "🇳🇱" },
  { iso: "NZ", name: "New Zealand", dial: "+64", flag: "🇳🇿" },
];

export const DEFAULT_COUNTRY = COUNTRY_CODES[0];

/**
 * Compose an E.164 number from a dial code and a (possibly messy) national
 * number. Strips spaces, dashes, parens, and a leading 0 (common trunk prefix).
 */
export function toE164(dialCode: string, national: string): string {
  const digits = national.replace(/[^\d]/g, "").replace(/^0+/, "");
  const code = dialCode.replace(/[^\d+]/g, "");
  return `${code}${digits}`;
}

/** Strict E.164 check: "+" then 8–15 digits, no leading zero after "+". */
export function isValidE164(value: string): boolean {
  return /^\+[1-9]\d{7,14}$/.test(value);
}

/**
 * Request an outbound AI callback. Always resolves to a {@link CallbackResult}
 * (never throws) so the UI can render success/error states declaratively.
 */
export async function requestCallback(payload: CallbackRequest): Promise<CallbackResult> {
  if (!isValidE164(payload.phone)) {
    return { success: false, message: "Please enter a valid phone number." };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(CALL_ME_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: payload.name?.trim() || undefined,
        phone: payload.phone,
      }),
      signal: controller.signal,
    });

    const data = (await res.json().catch(() => null)) as CallbackResult | null;

    if (!res.ok || !data?.success) {
      return {
        success: false,
        message: data?.message ?? "We couldn't place the call right now. Please try again.",
      };
    }
    return { success: true, message: data.message };
  } catch (err) {
    const aborted = err instanceof DOMException && err.name === "AbortError";
    return {
      success: false,
      message: aborted
        ? "The request timed out. Please try again."
        : "We couldn't reach the calling service. Please try again.",
    };
  } finally {
    clearTimeout(timer);
  }
}
