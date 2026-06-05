"use client";

import { motion } from "framer-motion";
import { AlertCircle, ShieldCheck, Sparkles, User2 } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { CitationsList } from "@/components/citations/citations-list";
import { SchedulingCard } from "@/components/scheduling/scheduling-card";
import { FormattedText } from "@/components/chat/formatted-text";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { PERSONA } from "@/lib/constants";
import { cn, formatTime } from "@/lib/utils";
import type { ChatMessage } from "@/types";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isStreaming = message.status === "streaming";
  const isError = message.status === "error";
  const showTyping = isStreaming && !message.content;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
      className={cn("flex w-full gap-3", isUser ? "flex-row-reverse" : "flex-row")}
    >
      <Avatar className="mt-0.5 h-8 w-8">
        <AvatarFallback
          className={cn(
            "text-xs",
            isUser
              ? "bg-white/10 text-foreground"
              : "bg-gradient-to-br from-primary to-secondary text-white",
          )}
        >
          {isUser ? <User2 className="h-4 w-4" /> : PERSONA.initials}
        </AvatarFallback>
      </Avatar>

      <div className={cn("flex min-w-0 max-w-[85%] flex-col", isUser ? "items-end" : "items-start")}>
        <div className="mb-1 flex items-center gap-2 text-xs text-muted-foreground">
          <span className="font-medium text-foreground/70">
            {isUser ? "You" : PERSONA.name}
          </span>
          <span>{formatTime(message.createdAt)}</span>
        </div>

        <div
          className={cn(
            "rounded-2xl px-4 py-3 text-sm",
            isUser
              ? "rounded-tr-sm bg-primary text-primary-foreground shadow-glow"
              : "rounded-tl-sm border border-border bg-card",
            isError && "border-destructive/40 bg-destructive/[0.06]",
          )}
        >
          {showTyping ? (
            <TypingIndicator />
          ) : isError ? (
            <div className="flex items-start gap-2 text-sm text-foreground/90">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
              <span>{message.error ?? "This response failed. Please try again."}</span>
            </div>
          ) : isUser ? (
            <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
          ) : (
            <FormattedText text={message.content} />
          )}
          {isStreaming && message.content && (
            <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-secondary align-middle" />
          )}
        </div>

        {/* Assistant metadata: grounding + latency */}
        {!isUser && !isError && message.status === "complete" && (
          <div className="mt-1.5 flex items-center gap-3 text-[11px] text-muted-foreground">
            {message.grounded != null && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span
                    className={cn(
                      "inline-flex items-center gap-1",
                      message.grounded ? "text-emerald-400" : "text-amber-400",
                    )}
                  >
                    {message.grounded ? (
                      <ShieldCheck className="h-3 w-3" />
                    ) : (
                      <Sparkles className="h-3 w-3" />
                    )}
                    {message.grounded ? "Grounded" : "Unverified"}
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  {message.grounded
                    ? "Answer is supported by the cited corpus."
                    : "Answer could not be fully verified against the corpus."}
                </TooltipContent>
              </Tooltip>
            )}
            {message.latencyMs != null && <span>{(message.latencyMs / 1000).toFixed(1)}s</span>}
          </div>
        )}

        {/* Citations */}
        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="w-full">
            <CitationsList citations={message.citations} />
          </div>
        )}

        {/* Scheduling widget */}
        {!isUser && message.status === "complete" && message.scheduling && (
          <div className="w-full">
            <SchedulingCard initial={message.availability} />
          </div>
        )}
      </div>
    </motion.div>
  );
}
