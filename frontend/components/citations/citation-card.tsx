"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, ExternalLink, Maximize2 } from "lucide-react";
import { SourceBadge } from "@/components/citations/source-badge";
import { cn } from "@/lib/utils";
import type { Citation } from "@/types";

interface CitationCardProps {
  citation: Citation;
  /** Open the full details panel. */
  onOpen: (citation: Citation) => void;
}

/** Compact, expandable citation card with an inline snippet preview. */
export function CitationCard({ citation, onOpen }: CitationCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={cn(
        "group rounded-lg border border-border bg-card/60 p-3 transition-colors hover:border-white/15",
      )}
    >
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 flex h-5 min-w-5 items-center justify-center rounded-md bg-primary/15 px-1 text-[11px] font-semibold text-primary">
          {citation.n}
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <p className="truncate text-sm font-medium text-foreground">{citation.title}</p>
            <button
              onClick={() => onOpen(citation)}
              aria-label="Open citation details"
              className="rounded-md p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-white/5 hover:text-foreground group-hover:opacity-100"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="mt-1.5 flex items-center gap-2">
            <SourceBadge sourceType={citation.source_type} />
            {citation.url && (
              <a
                href={citation.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-secondary"
              >
                <ExternalLink className="h-3 w-3" /> Open
              </a>
            )}
          </div>

          <button
            onClick={() => setExpanded((v) => !v)}
            className="mt-2 inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground hover:text-foreground"
          >
            <ChevronDown
              className={cn("h-3 w-3 transition-transform", expanded && "rotate-180")}
            />
            {expanded ? "Hide snippet" : "Show snippet"}
          </button>

          <AnimatePresence initial={false}>
            {expanded && (
              <motion.p
                key="snippet"
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.22 }}
                className="overflow-hidden text-xs leading-relaxed text-muted-foreground"
              >
                <span className="mt-2 block border-l-2 border-primary/30 pl-3">
                  {citation.snippet}
                </span>
              </motion.p>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
}
