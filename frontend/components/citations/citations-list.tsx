"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, ExternalLink } from "lucide-react";
import { SOURCE_BADGES } from "@/lib/constants";
import { cn, sourceKind } from "@/lib/utils";
import type { Citation } from "@/types";

interface CitationsListProps {
  citations: Citation[];
}

/**
 * Perplexity / Notion-AI style sources block.
 *
 * Collapsed by default to a single compact row ("📚 Sources (N)"). Expanding
 * reveals a wrapped strip of numbered source *pills* (no large cards); clicking
 * a pill reveals that citation's snippet inline. All citation data — number,
 * title, source type, snippet, and url — remains accessible.
 */
export function CitationsList({ citations }: CitationsListProps) {
  const [expanded, setExpanded] = useState(false);
  const [activeN, setActiveN] = useState<number | null>(null);

  if (!citations.length) return null;

  const active = citations.find((c) => c.n === activeN) ?? null;
  const togglePill = (c: Citation) => setActiveN((cur) => (cur === c.n ? null : c.n));

  return (
    <div className="mt-2">
      {/* Collapsed trigger — the only thing shown by default. */}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-md px-1.5 py-1 text-xs font-medium",
          "text-muted-foreground transition-colors hover:text-foreground",
        )}
      >
        <span aria-hidden="true">📚</span>
        <span>Sources</span>
        <span className="text-muted-foreground/60">({citations.length})</span>
        <ChevronDown
          className={cn("h-3.5 w-3.5 transition-transform duration-200", expanded && "rotate-180")}
        />
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            key="sources"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="overflow-hidden"
          >
            {/* Compact, numbered source pills. */}
            <div className="mt-2 flex flex-wrap gap-1.5">
              {citations.map((c) => {
                const cfg = SOURCE_BADGES[sourceKind(c.source_type)];
                const Icon = cfg.icon;
                const isActive = c.n === activeN;
                return (
                  <button
                    key={`${c.n}-${c.title}`}
                    type="button"
                    onClick={() => togglePill(c)}
                    title={c.title}
                    aria-pressed={isActive}
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-full border px-2 py-1 text-[11px] font-medium transition-colors",
                      isActive
                        ? "border-primary/40 bg-primary/10 text-foreground"
                        : "border-border bg-card/50 text-muted-foreground hover:border-white/15 hover:text-foreground",
                    )}
                  >
                    <span className="flex h-4 min-w-4 items-center justify-center rounded-full bg-primary/15 px-1 text-[10px] font-semibold leading-none text-primary">
                      {c.n}
                    </span>
                    <Icon className="h-3 w-3 shrink-0 opacity-80" />
                    <span className="max-w-[160px] truncate">{c.title}</span>
                  </button>
                );
              })}
            </div>

            {/* Inline snippet for the selected pill. */}
            <AnimatePresence initial={false} mode="wait">
              {active && (
                <motion.div
                  key={active.n}
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.18, ease: "easeOut" }}
                  className="overflow-hidden"
                >
                  <div className="mt-2 rounded-lg border border-border bg-card/40 p-3">
                    <div className="mb-1.5 flex items-center justify-between gap-2">
                      <span className="truncate text-xs font-semibold text-foreground">
                        {active.title}
                      </span>
                      {active.url ? (
                        <a
                          href={active.url}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex shrink-0 items-center gap-1 text-[11px] text-muted-foreground transition-colors hover:text-secondary"
                        >
                          <ExternalLink className="h-3 w-3" /> Open
                        </a>
                      ) : null}
                    </div>
                    <p className="border-l-2 border-primary/30 pl-3 text-xs leading-relaxed text-muted-foreground">
                      {active.snippet}
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
