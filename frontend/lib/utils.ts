import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { SourceKind } from "@/types";

/** Tailwind-aware className combiner (shadcn convention). */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Stable-ish id without extra deps. */
export function uid(prefix = "id"): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`;
}

/** Format an epoch-ms timestamp as a short time (e.g. "2:04 PM"). */
export function formatTime(ms: number): string {
  return new Date(ms).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

/** Format an ISO datetime to a readable "Wed, Jun 10 · 2:00 PM". */
export function formatSlot(iso: string, timeZone?: string): { date: string; time: string } {
  const d = new Date(iso);
  const date = d.toLocaleDateString([], {
    weekday: "short",
    month: "short",
    day: "numeric",
    timeZone,
  });
  const time = d.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
    timeZone,
  });
  return { date, time };
}

/** Group slots by their calendar day (for the slot picker). */
export function groupSlotsByDay<T extends { start: string }>(
  slots: T[],
  timeZone?: string,
): { day: string; slots: T[] }[] {
  const groups = new Map<string, T[]>();
  for (const slot of slots) {
    const day = new Date(slot.start).toLocaleDateString([], {
      weekday: "long",
      month: "short",
      day: "numeric",
      timeZone,
    });
    const bucket = groups.get(day) ?? [];
    bucket.push(slot);
    groups.set(day, bucket);
  }
  return Array.from(groups.entries()).map(([day, s]) => ({ day, slots: s }));
}

/** Map a backend `source_type` onto a canonical badge kind. */
export function sourceKind(sourceType: string): SourceKind {
  const s = (sourceType || "").toLowerCase();
  if (s.includes("resume")) return "resume";
  if (s.includes("github")) return "github";
  if (s.includes("project")) return "project";
  if (s.includes("experience")) return "experience";
  if (s.includes("about")) return "about";
  if (s.includes("markdown") || s.endsWith(".md")) return "markdown";
  return "unknown";
}

/** Convert a base64 (mp3) payload into a playable object URL. */
export function base64ToAudioUrl(base64: string, mime = "audio/mpeg"): string {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: mime });
  return URL.createObjectURL(blob);
}

/** Derive a conversation title from the first user message. */
export function deriveTitle(text: string): string {
  const trimmed = text.trim().replace(/\s+/g, " ");
  if (!trimmed) return "New chat";
  return trimmed.length > 42 ? `${trimmed.slice(0, 42)}…` : trimmed;
}

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
