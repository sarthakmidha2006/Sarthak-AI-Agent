"use client";

import { MessageList } from "@/components/chat/message-list";
import { EmptyState } from "@/components/chat/empty-state";
import { ChatInput } from "@/components/chat/chat-input";
import { QuickQuestions } from "@/components/chat/quick-questions";
import type { ChatMessage } from "@/types";

interface ChatAreaProps {
  messages: ChatMessage[];
  isSending: boolean;
  disabled: boolean;
  voiceActive: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onVoice: () => void;
}

/** Main conversation surface: transcript (or zero-state) + composer. */
export function ChatArea({
  messages,
  isSending,
  disabled,
  voiceActive,
  onSend,
  onStop,
  onVoice,
}: ChatAreaProps) {
  const hasMessages = messages.length > 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1">
        {hasMessages ? (
          <MessageList messages={messages} />
        ) : (
          <EmptyState onPick={onSend} disabled={disabled || isSending} />
        )}
      </div>

      {/* Inline quick-questions strip once a conversation is underway */}
      {hasMessages && !isSending && (
        <div className="mx-auto w-full max-w-3xl px-4 pb-1 sm:px-6">
          <QuickQuestions onPick={onSend} disabled={disabled} />
        </div>
      )}

      <ChatInput
        onSend={onSend}
        onStop={onStop}
        isSending={isSending}
        disabled={disabled}
        onVoice={onVoice}
        voiceActive={voiceActive}
      />
    </div>
  );
}
