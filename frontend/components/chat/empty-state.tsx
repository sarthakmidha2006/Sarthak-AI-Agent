"use client";

import { motion } from "framer-motion";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { QuickQuestions } from "@/components/chat/quick-questions";
import { PERSONA } from "@/lib/constants";

interface EmptyStateProps {
  onPick: (query: string) => void;
  disabled?: boolean;
}

/** Welcome / zero-state shown for a fresh conversation. */
export function EmptyState({ onPick, disabled }: EmptyStateProps) {
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
      </div>
    </div>
  );
}
