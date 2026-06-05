"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowDown } from "lucide-react";
import { MessageBubble } from "@/components/chat/message-bubble";
import { Button } from "@/components/ui/button";
import type { ChatMessage } from "@/types";

interface MessageListProps {
  messages: ChatMessage[];
}

/** Scrollable transcript with sticky auto-scroll + "jump to latest" affordance. */
export function MessageList({ messages }: MessageListProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState(true);

  // Auto-scroll while pinned to the bottom (handles streaming growth too).
  useEffect(() => {
    if (pinned) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pinned]);

  const handleScroll = () => {
    const el = viewportRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    setPinned(distance < 120);
  };

  const jumpToLatest = () => {
    setPinned(true);
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="relative h-full">
      <div
        ref={viewportRef}
        onScroll={handleScroll}
        className="scrollbar-thin h-full overflow-y-auto px-4 py-6 sm:px-6"
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-6">
          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}
          <div ref={bottomRef} className="h-px" />
        </div>
      </div>

      <AnimatePresence>
        {!pinned && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2"
          >
            <Button
              size="sm"
              variant="outline"
              onClick={jumpToLatest}
              className="pointer-events-auto rounded-full shadow-soft"
            >
              <ArrowDown className="h-3.5 w-3.5" /> Latest
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
