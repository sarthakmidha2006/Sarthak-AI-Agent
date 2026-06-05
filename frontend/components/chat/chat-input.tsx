"use client";

import { useLayoutEffect, useRef, useState } from "react";
import { ArrowUp, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { VoiceButton } from "@/components/voice/voice-button";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSend: (text: string) => void;
  onStop?: () => void;
  isSending: boolean;
  disabled?: boolean;
  onVoice: () => void;
  voiceActive?: boolean;
}

const MAX_CHARS = 4000;

/** Auto-growing composer with send/stop, char count, and voice trigger. */
export function ChatInput({
  onSend,
  onStop,
  isSending,
  disabled,
  onVoice,
  voiceActive,
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => {
    const text = value.trim();
    if (!text || isSending || disabled) return;
    onSend(text);
    setValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-border bg-background/80 px-4 py-3 backdrop-blur-xl sm:px-6">
      <div className="mx-auto max-w-3xl">
        <div
          className={cn(
            "glass flex items-end gap-2 rounded-2xl p-2 shadow-soft transition-colors",
            "focus-within:border-primary/40 focus-within:shadow-glow",
          )}
        >
          <VoiceButton onClick={onVoice} active={voiceActive} disabled={disabled} />

          <Textarea
            ref={ref}
            value={value}
            onChange={(e) => setValue(e.target.value.slice(0, MAX_CHARS))}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={disabled}
            placeholder={disabled ? "Backend offline…" : "Message Sarthak's AI persona…"}
            className="max-h-[200px] flex-1 border-0 bg-transparent px-1.5 py-2 focus-visible:ring-0"
          />

          {isSending ? (
            <Button
              size="icon"
              variant="secondary"
              onClick={onStop}
              aria-label="Stop generating"
              className="shrink-0 rounded-xl"
            >
              <Square className="h-4 w-4 fill-current" />
            </Button>
          ) : (
            <Button
              size="icon"
              onClick={submit}
              disabled={!value.trim() || disabled}
              aria-label="Send message"
              className="shrink-0 rounded-xl"
            >
              <ArrowUp className="h-4 w-4" strokeWidth={2.5} />
            </Button>
          )}
        </div>
        <div className="mt-1.5 flex items-center justify-between px-2 text-[11px] text-muted-foreground">
          <span>Enter to send · Shift+Enter for newline</span>
          {value.length > MAX_CHARS * 0.8 && (
            <span>
              {value.length}/{MAX_CHARS}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
