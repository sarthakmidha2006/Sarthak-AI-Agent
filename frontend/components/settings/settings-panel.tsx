"use client";

import { Info, Trash2 } from "lucide-react";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { HealthStatus } from "@/components/settings/health-status";
import { API_BASE_URL } from "@/lib/api";
import type { BackendStatus, HealthResponse } from "@/types";

interface SettingsPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  status: BackendStatus;
  health: HealthResponse | null;
  error: string | null;
  onRefresh: () => void;
  onClearHistory: () => void;
}

/** Right-side settings drawer: backend status, models, corpus, actions. */
export function SettingsPanel({
  open,
  onOpenChange,
  status,
  health,
  error,
  onRefresh,
  onClearHistory,
}: SettingsPanelProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="flex w-96 flex-col p-0">
        <div className="border-b border-border px-5 py-4">
          <h2 className="text-base font-semibold">Settings</h2>
          <p className="text-xs text-muted-foreground">Backend status & preferences</p>
        </div>

        <div className="scrollbar-thin flex-1 space-y-6 overflow-y-auto p-5">
          <section>
            <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Backend
            </h3>
            <HealthStatus status={status} health={health} error={error} onRefresh={onRefresh} />
          </section>

          <Separator />

          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Connection
            </h3>
            <div className="flex items-start gap-2 rounded-lg border border-border bg-card/50 p-3 text-xs text-muted-foreground">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>
                API endpoint:
                <br />
                <span className="break-all font-mono text-foreground/80">{API_BASE_URL}</span>
              </span>
            </div>
          </section>

          <Separator />

          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Data
            </h3>
            <Button variant="outline" className="w-full justify-start text-destructive" onClick={onClearHistory}>
              <Trash2 className="h-4 w-4" /> Clear conversation history
            </Button>
            <p className="mt-1.5 text-[11px] text-muted-foreground">
              Conversations are stored locally in your browser only.
            </p>
          </section>
        </div>
      </SheetContent>
    </Sheet>
  );
}
