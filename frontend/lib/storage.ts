/**
 * Conversation persistence. The backend has no "list conversations" endpoint,
 * so chat history is stored client-side in localStorage. Reads/writes are
 * defensive and SSR-safe (guard against `window` being undefined).
 */
import type { Conversation } from "@/types";

const KEY = "ai-persona:conversations:v1";

export const storage = {
  load(): Conversation[] {
    if (typeof window === "undefined") return [];
    try {
      const raw = window.localStorage.getItem(KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw) as Conversation[];
      if (!Array.isArray(parsed)) return [];
      return parsed;
    } catch {
      return [];
    }
  },

  save(conversations: Conversation[]): void {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(KEY, JSON.stringify(conversations));
    } catch {
      // Quota or serialization failure: non-fatal.
    }
  },

  clear(): void {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.removeItem(KEY);
    } catch {
      /* noop */
    }
  },
};
