"use client";

import { useState } from "react";
import { BookText } from "lucide-react";
import { CitationCard } from "@/components/citations/citation-card";
import { CitationDetailPanel } from "@/components/citations/citation-detail-panel";
import type { Citation } from "@/types";

interface CitationsListProps {
  citations: Citation[];
}

/** "Sources" block rendered beneath an assistant answer. */
export function CitationsList({ citations }: CitationsListProps) {
  const [selected, setSelected] = useState<Citation | null>(null);
  const [open, setOpen] = useState(false);

  if (!citations.length) return null;

  const handleOpen = (c: Citation) => {
    setSelected(c);
    setOpen(true);
  };

  return (
    <div className="mt-3">
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        <BookText className="h-3.5 w-3.5" />
        Sources
        <span className="text-muted-foreground/60">· {citations.length}</span>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {citations.map((c) => (
          <CitationCard key={`${c.n}-${c.title}`} citation={c} onOpen={handleOpen} />
        ))}
      </div>
      <CitationDetailPanel citation={selected} open={open} onOpenChange={setOpen} />
    </div>
  );
}
