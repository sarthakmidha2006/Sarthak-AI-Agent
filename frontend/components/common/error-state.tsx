"use client";

import { AlertTriangle, RefreshCw, WifiOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  icon?: "warning" | "offline";
  compact?: boolean;
  className?: string;
}

/** Reusable, elegant error surface for failed fetches / actions. */
export function ErrorState({
  title = "Something went wrong",
  message,
  onRetry,
  icon = "warning",
  compact = false,
  className,
}: ErrorStateProps) {
  const Icon = icon === "offline" ? WifiOff : AlertTriangle;
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-xl border border-destructive/25 bg-destructive/[0.06] text-center",
        compact ? "gap-2 p-4" : "gap-3 p-8",
        className,
      )}
    >
      <span className="flex h-10 w-10 items-center justify-center rounded-full bg-destructive/15 text-destructive">
        <Icon className="h-5 w-5" />
      </span>
      <div>
        <p className="text-sm font-semibold text-foreground">{title}</p>
        <p className="mt-0.5 text-xs text-muted-foreground">{message}</p>
      </div>
      {onRetry && (
        <Button size="sm" variant="outline" onClick={onRetry}>
          <RefreshCw className="h-3.5 w-3.5" /> Try again
        </Button>
      )}
    </div>
  );
}
