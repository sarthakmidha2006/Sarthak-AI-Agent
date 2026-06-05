"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Loader2, Mic, Square, Volume2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/common/error-state";
import { cn } from "@/lib/utils";
import type { VoiceStatus } from "@/hooks/use-voice";
import type { VoiceResponse } from "@/types";

interface VoiceModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  status: VoiceStatus;
  error: string | null;
  last: VoiceResponse | null;
  audioUrl: string | null;
  isSupported: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
}

const STATUS_COPY: Record<VoiceStatus, string> = {
  idle: "Tap the mic and start speaking",
  requesting: "Requesting microphone…",
  recording: "Listening… tap to stop",
  processing: "Transcribing & thinking…",
  error: "Voice error",
};

export function VoiceModal({
  open,
  onOpenChange,
  status,
  error,
  last,
  audioUrl,
  isSupported,
  onStart,
  onStop,
  onReset,
}: VoiceModalProps) {
  const recording = status === "recording";
  const processing = status === "processing" || status === "requesting";

  const handleClose = (next: boolean) => {
    if (!next) {
      if (recording) onStop();
      onReset();
    }
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-md">
        <DialogHeader className="items-center text-center">
          <DialogTitle>Voice mode</DialogTitle>
          <DialogDescription>{STATUS_COPY[status]}</DialogDescription>
        </DialogHeader>

        {!isSupported ? (
          <ErrorState
            title="Not supported"
            message="This browser can't record audio. Try Chrome or Safari on a device with a microphone."
            compact
          />
        ) : status === "error" ? (
          <ErrorState title="Voice error" message={error ?? "Please try again."} onRetry={onStart} compact />
        ) : (
          <div className="flex flex-col items-center gap-6 py-4">
            {/* Mic orb with recording animation */}
            <button
              onClick={recording ? onStop : processing ? undefined : onStart}
              disabled={processing}
              className="relative flex h-28 w-28 items-center justify-center rounded-full"
              aria-label={recording ? "Stop recording" : "Start recording"}
            >
              <AnimatePresence>
                {recording && (
                  <>
                    {[0, 1, 2].map((i) => (
                      <motion.span
                        key={i}
                        className="absolute inset-0 rounded-full bg-secondary/25"
                        initial={{ scale: 1, opacity: 0.6 }}
                        animate={{ scale: 1.6, opacity: 0 }}
                        transition={{ duration: 1.6, repeat: Infinity, delay: i * 0.4 }}
                      />
                    ))}
                  </>
                )}
              </AnimatePresence>
              <span
                className={cn(
                  "relative flex h-20 w-20 items-center justify-center rounded-full text-white shadow-glow transition-colors",
                  recording
                    ? "bg-gradient-to-br from-rose-500 to-secondary"
                    : "bg-gradient-to-br from-primary to-secondary",
                )}
              >
                {processing ? (
                  <Loader2 className="h-7 w-7 animate-spin" />
                ) : recording ? (
                  <Square className="h-6 w-6 fill-current" />
                ) : (
                  <Mic className="h-7 w-7" />
                )}
              </span>
            </button>

            {/* Transcript + answer */}
            <AnimatePresence>
              {last && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="w-full space-y-3"
                >
                  {last.transcript && (
                    <div className="rounded-lg border border-border bg-background/50 p-3">
                      <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                        You said
                      </p>
                      <p className="text-sm text-foreground/90">{last.transcript}</p>
                    </div>
                  )}
                  <div className="rounded-lg border border-primary/25 bg-primary/[0.06] p-3">
                    <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-primary">
                      Sarthak
                    </p>
                    <p className="text-sm text-foreground/90">{last.answer}</p>
                  </div>
                  {audioUrl && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full"
                      onClick={() => new Audio(audioUrl).play().catch(() => void 0)}
                    >
                      <Volume2 className="h-4 w-4" /> Replay audio
                    </Button>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
