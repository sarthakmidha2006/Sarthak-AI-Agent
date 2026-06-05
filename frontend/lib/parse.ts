/**
 * Defensive parsers for the backend's tool-call payloads. The brain surfaces
 * `tool_calls: [{ name, arguments, result }]`; the exact result shape can vary,
 * so we extract scheduling info best-effort and never throw.
 */
import { AVAILABILITY_TOOL_NAMES, BOOKING_TOOL_NAMES } from "@/lib/constants";
import type { AvailabilityResponse, SlotView, ToolCallView } from "@/types";

function asRecord(v: unknown): Record<string, unknown> {
  return v && typeof v === "object" ? (v as Record<string, unknown>) : {};
}

function toSlots(value: unknown): SlotView[] {
  if (!Array.isArray(value)) return [];
  const out: SlotView[] = [];
  for (const item of value) {
    const rec = asRecord(item);
    const start = rec.start ?? rec.start_time;
    const end = rec.end ?? rec.end_time;
    if (typeof start === "string" && typeof end === "string") {
      out.push({ start, end });
    }
  }
  return out;
}

/** True if any tool call is scheduling-related (availability or booking). */
export function isSchedulingTurn(toolCalls: ToolCallView[] | undefined): boolean {
  if (!toolCalls?.length) return false;
  return toolCalls.some(
    (t) =>
      AVAILABILITY_TOOL_NAMES.includes(t.name) || BOOKING_TOOL_NAMES.includes(t.name),
  );
}

/**
 * Pull an availability payload out of tool-call results, if present. Returns
 * null when no slots can be recovered (the UI then fetches /availability live).
 */
export function extractAvailability(
  toolCalls: ToolCallView[] | undefined,
): AvailabilityResponse | null {
  if (!toolCalls?.length) return null;
  for (const call of toolCalls) {
    if (!AVAILABILITY_TOOL_NAMES.includes(call.name)) continue;
    const result = asRecord(call.result);
    const slots = toSlots(result.slots ?? result.availability ?? result.available_slots);
    if (slots.length === 0) continue;
    return {
      timezone: typeof result.timezone === "string" ? result.timezone : "",
      duration_minutes:
        typeof result.duration_minutes === "number" ? result.duration_minutes : 30,
      slots,
      count: slots.length,
    };
  }
  return null;
}
