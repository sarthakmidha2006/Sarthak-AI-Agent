"use client";

import { ExternalLink } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { SourceBadge } from "@/components/citations/source-badge";
import { Button } from "@/components/ui/button";
import type { Citation } from "@/types";

interface CitationDetailPanelProps {
  citation: Citation | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** Full citation details surfaced when a citation card is clicked. */
export function CitationDetailPanel({ citation, open, onOpenChange }: CitationDetailPanelProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        {citation && (
          <>
            <DialogHeader>
              <div className="mb-1 flex items-center gap-2">
                <span className="flex h-6 min-w-6 items-center justify-center rounded-md bg-primary/15 px-1.5 text-xs font-semibold text-primary">
                  {citation.n}
                </span>
                <SourceBadge sourceType={citation.source_type} />
              </div>
              <DialogTitle className="text-left text-base">{citation.title}</DialogTitle>
              <DialogDescription className="text-left">
                Source excerpt used to ground this answer.
              </DialogDescription>
            </DialogHeader>

            <div className="max-h-[50vh] overflow-y-auto rounded-lg border border-border bg-background/50 p-4 text-sm leading-relaxed text-foreground/90 scrollbar-thin">
              {citation.snippet}
            </div>

            {citation.url && (
              <Button asChild variant="outline" className="w-full">
                <a href={citation.url} target="_blank" rel="noreferrer">
                  <ExternalLink className="h-4 w-4" /> Open original source
                </a>
              </Button>
            )}
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
