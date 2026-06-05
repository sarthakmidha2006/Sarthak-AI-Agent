"use client";

import { AnimatePresence, motion } from "framer-motion";
import { RefreshCw, WifiOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { BackendStatus } from "@/types";

interface BackendStatusBannerProps {
  status: BackendStatus;
  onRetry: () => void;
}

/** Slim banner shown when the backend is unreachable. */
export function BackendStatusBanner({ status, onRetry }: BackendStatusBannerProps) {
  return (
    <AnimatePresence>
      {status === "offline" && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="overflow-hidden border-b border-destructive/30 bg-destructive/10"
        >
          <div className="flex items-center justify-center gap-3 px-4 py-2 text-xs text-destructive-foreground">
            <span className="flex items-center gap-1.5 font-medium">
              <WifiOff className="h-3.5 w-3.5" />
              Backend offline — responses are unavailable.
            </span>
            <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={onRetry}>
              <RefreshCw className="h-3 w-3" /> Retry
            </Button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
