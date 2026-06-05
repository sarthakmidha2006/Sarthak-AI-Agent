"use client";

import { Activity, Database, Cpu, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { BackendStatus, HealthResponse } from "@/types";

interface HealthStatusProps {
  status: BackendStatus;
  health: HealthResponse | null;
  error: string | null;
  onRefresh: () => void;
}

const DOT: Record<BackendStatus, string> = {
  online: "bg-emerald-400",
  offline: "bg-destructive",
  checking: "bg-amber-400",
};

/** Backend health readout for the settings panel. */
export function HealthStatus({ status, health, error, onRefresh }: HealthStatusProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
            <span
              className={cn(
                "absolute inline-flex h-full w-full rounded-full opacity-60",
                status === "online" && "animate-ping",
                DOT[status],
              )}
            />
            <span className={cn("relative inline-flex h-2.5 w-2.5 rounded-full", DOT[status])} />
          </span>
          <span className="text-sm font-medium capitalize">
            {status === "checking" ? "Checking…" : status}
          </span>
        </div>
        <Button variant="ghost" size="icon-sm" onClick={onRefresh} aria-label="Refresh health">
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>

      {status === "offline" && (
        <p className="rounded-lg border border-destructive/25 bg-destructive/[0.06] p-3 text-xs text-muted-foreground">
          {error ?? "Backend unreachable. Check NEXT_PUBLIC_API_URL and that the server is running."}
        </p>
      )}

      <div className="space-y-2">
        <StatRow
          icon={<Database className="h-4 w-4 text-primary" />}
          label="Corpus chunks"
          value={health ? String(health.corpus_chunks) : null}
        />
        <StatRow
          icon={<Activity className="h-4 w-4 text-primary" />}
          label="BM25 index"
          value={health ? String(health.bm25_size) : null}
        />
        <StatRow
          icon={<Cpu className="h-4 w-4 text-primary" />}
          label="Chat model"
          value={health?.models.chat ?? null}
          mono
        />
        <StatRow
          icon={<Cpu className="h-4 w-4 text-secondary" />}
          label="Embeddings"
          value={health?.models.embedding ?? null}
          mono
        />
        <StatRow
          icon={<Cpu className="h-4 w-4 text-secondary" />}
          label="Reranker"
          value={health?.models.reranker_provider ?? null}
          mono
        />
      </div>
    </div>
  );
}

function StatRow({
  icon,
  label,
  value,
  mono,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | null;
  mono?: boolean;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card/50 px-3 py-2">
      <span className="flex items-center gap-2 text-xs text-muted-foreground">
        {icon}
        {label}
      </span>
      {value === null ? (
        <Skeleton className="h-4 w-16" />
      ) : (
        <span className={cn("max-w-[55%] truncate text-xs text-foreground", mono && "font-mono")}>
          {value}
        </span>
      )}
    </div>
  );
}
