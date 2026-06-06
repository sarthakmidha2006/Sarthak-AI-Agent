"use client";

import { motion } from "framer-motion";
import { PhoneCall } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { QuickQuestions } from "@/components/chat/quick-questions";
import { PERSONA } from "@/lib/constants";

interface EmptyStateProps {
  onPick: (query: string) => void;
  onCallMe: () => void;
  disabled?: boolean;
}

/** Welcome / zero-state shown for a fresh conversation. */
export function EmptyState({ onPick, onCallMe, disabled }: EmptyStateProps) {
  return (
    <div className="flex h-full items-center justify-center px-4 py-10">
      <div className="mx-auto w-full max-w-2xl text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: "spring", stiffness: 200, damping: 18 }}
          className="mx-auto mb-5 w-fit"
        >
          <Avatar className="h-16 w-16 shadow-glow">
            <AvatarFallback className="text-lg">{PERSONA.initials}</AvatarFallback>
          </Avatar>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          Chat with <span className="text-gradient">{PERSONA.name}</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mx-auto mt-2 max-w-md text-sm text-muted-foreground"
        >
          {PERSONA.tagline} Ask anything about his work, projects, and experience — or
          book a meeting.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="mt-7 flex justify-center"
        >
          <QuickQuestions onPick={onPick} disabled={disabled} />
        </motion.div>

        {/* "Prefer talking instead?" — outbound AI callback */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mx-auto mt-8 w-full max-w-md"
        >
          <div className="flex flex-col items-center gap-3 rounded-2xl border border-border bg-card/40 p-5 text-center sm:flex-row sm:items-center sm:gap-4 sm:text-left">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-primary/15">
              <PhoneCall className="h-5 w-5 text-primary" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-foreground">Prefer talking instead?</p>
              <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
                Receive a phone call from {PERSONA.name}&apos;s AI representative and ask about
                projects, experience, skills, or interview availability.
              </p>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={onCallMe}
              className="w-full shrink-0 sm:w-auto"
            >
              <PhoneCall className="h-4 w-4" />
              Request a Call
            </Button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
