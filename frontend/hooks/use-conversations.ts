"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { storage } from "@/lib/storage";
import { deriveTitle, uid } from "@/lib/utils";
import type { ChatMessage, Conversation } from "@/types";

export interface UseConversations {
  conversations: Conversation[];
  activeId: string | null;
  active: Conversation | null;
  newChat: () => string;
  selectChat: (id: string) => void;
  deleteChat: (id: string) => void;
  /** Replace messages for the active conversation (and refresh title/timestamps). */
  setMessages: (updater: (prev: ChatMessage[]) => ChatMessage[]) => void;
}

function createConversation(): Conversation {
  const now = Date.now();
  return { id: uid("conv"), title: "New chat", createdAt: now, updatedAt: now, messages: [] };
}

/** Owns the conversation list + active conversation, persisted to localStorage. */
export function useConversations(): UseConversations {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const hydrated = useRef(false);

  // Hydrate once on mount.
  useEffect(() => {
    const loaded = storage.load();
    if (loaded.length > 0) {
      setConversations(loaded);
      setActiveId(loaded[0].id);
    } else {
      const fresh = createConversation();
      setConversations([fresh]);
      setActiveId(fresh.id);
    }
    hydrated.current = true;
  }, []);

  // Persist on change (after hydration to avoid clobbering with [] on first render).
  useEffect(() => {
    if (hydrated.current) storage.save(conversations);
  }, [conversations]);

  const newChat = useCallback(() => {
    const fresh = createConversation();
    setConversations((prev) => [fresh, ...prev]);
    setActiveId(fresh.id);
    return fresh.id;
  }, []);

  const selectChat = useCallback((id: string) => setActiveId(id), []);

  const deleteChat = useCallback(
    (id: string) => {
      setConversations((prev) => {
        const next = prev.filter((c) => c.id !== id);
        if (next.length === 0) {
          const fresh = createConversation();
          setActiveId(fresh.id);
          return [fresh];
        }
        setActiveId((cur) => (cur === id ? next[0].id : cur));
        return next;
      });
    },
    [],
  );

  const setMessages = useCallback(
    (updater: (prev: ChatMessage[]) => ChatMessage[]) => {
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeId) return c;
          const messages = updater(c.messages);
          const firstUser = messages.find((m) => m.role === "user");
          return {
            ...c,
            messages,
            title: c.title === "New chat" && firstUser ? deriveTitle(firstUser.content) : c.title,
            updatedAt: Date.now(),
          };
        }),
      );
    },
    [activeId],
  );

  const active = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? null,
    [conversations, activeId],
  );

  return { conversations, activeId, active, newChat, selectChat, deleteChat, setMessages };
}
