"use client";

import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
import { QUICK_QUESTIONS } from "@/lib/constants";

interface QuickQuestionsProps {
  onPick: (query: string) => void;
  disabled?: boolean;
}

/** Suggestion chips that send a prefilled query when clicked. */
export function QuickQuestions({ onPick, disabled }: QuickQuestionsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {QUICK_QUESTIONS.map((q, i) => (
        <motion.button
          key={q.label}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.04 }}
          whileHover={{ y: -2 }}
          whileTap={{ scale: 0.97 }}
          disabled={disabled}
          onClick={() => onPick(q.query)}
          className="group inline-flex items-center gap-1.5 rounded-full border border-border bg-card/60 px-3.5 py-2 text-sm text-foreground/80 transition-colors hover:border-primary/40 hover:bg-primary/[0.06] hover:text-foreground disabled:opacity-50"
        >
          {q.label}
          <ArrowUpRight className="h-3.5 w-3.5 text-muted-foreground transition-colors group-hover:text-primary" />
        </motion.button>
      ))}
    </div>
  );
}
