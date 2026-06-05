"use client";

import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { extractAvailability, isSchedulingTurn } from "@/lib/parse";
import { uid } from "@/lib/utils";
import { ApiError, type ChatMessage, type HistoryTurn } from "@/types";

interface UseChatArgs {
  sessionId: string | null;
  setSessionId: (id: string) => void;
  messages: ChatMessage[];
  setMessages: (updater: (prev: ChatMessage[]) => ChatMessage[]) => void;
}

/** Simulated streaming: reveal the final answer progressively for a live feel. */
async function streamInto(
  fullText: string,
  onChunk: (partial: string) => void,
  shouldStop: () => boolean,
) {
  const words = fullText.split(/(\s+)/); // keep whitespace tokens
  let acc = "";
  for (let i = 0; i < words.length; i++) {
    if (shouldStop()) {
      onChunk(fullText);
      return;
    }
    acc += words[i];
    onChunk(acc);
    // Slightly variable cadence feels more natural than a fixed tick.
    await new Promise((r) => setTimeout(r, 12 + Math.random() * 18));
  }
}

export function useChat({ sessionId, setSessionId, messages, setMessages }: UseChatArgs) {
  const [isSending, setIsSending] = useState(false);
  const cancelled = useRef(false);

  const send = useCallback(
    async (text: string) => {
      const content = text.trim();
      if (!content || isSending) return;
      cancelled.current = false;
      setIsSending(true);

      // History snapshot (completed turns only) for backend context.
      const history: HistoryTurn[] = messages
        .filter((m) => m.status !== "error")
        .map((m) => ({ role: m.role, content: m.content }));

      const userMsg: ChatMessage = {
        id: uid("msg"),
        role: "user",
        content,
        createdAt: Date.now(),
        status: "complete",
      };
      const assistantId = uid("msg");
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: Date.now(),
        status: "streaming",
      };
      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      try {
        const res = await api.chat({
          message: content,
          session_id: sessionId,
          history,
        });
        if (res.session_id) setSessionId(res.session_id);

        // Reveal text progressively.
        await streamInto(
          res.answer,
          (partial) =>
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: partial } : m)),
            ),
          () => cancelled.current,
        );

        const availability = extractAvailability(res.tool_calls);
        const scheduling =
          isSchedulingTurn(res.tool_calls) || /schedul|meeting|availab|book/i.test(content);

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: res.answer,
                  status: "complete",
                  citations: res.citations,
                  toolCalls: res.tool_calls,
                  grounded: res.grounded,
                  latencyMs: res.latency_ms,
                  availability,
                  scheduling,
                }
              : m,
          ),
        );
      } catch (err) {
        const message =
          err instanceof ApiError ? err.message : "Something went wrong. Please try again.";
        if (err instanceof ApiError && err.kind === "rate_limit") {
          toast.warning("Rate limited", { description: message });
        } else {
          toast.error("Request failed", { description: message });
        }
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, status: "error", content: m.content || "", error: message }
              : m,
          ),
        );
      } finally {
        setIsSending(false);
      }
    },
    [isSending, messages, sessionId, setMessages, setSessionId],
  );

  const stop = useCallback(() => {
    cancelled.current = true;
  }, []);

  return { send, stop, isSending };
}
