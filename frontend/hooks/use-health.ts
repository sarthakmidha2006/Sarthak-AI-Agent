"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { BackendStatus, HealthResponse } from "@/types";

interface UseHealth {
  status: BackendStatus;
  health: HealthResponse | null;
  error: string | null;
  refresh: () => Promise<void>;
}

const POLL_MS = 30_000;

/** Polls GET /health on mount and on an interval; exposes backend status. */
export function useHealth(): UseHealth {
  const [status, setStatus] = useState<BackendStatus>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setStatus((s) => (s === "online" ? s : "checking"));
    try {
      const data = await api.health();
      setHealth(data);
      setStatus(data.status === "ok" ? "online" : "offline");
      setError(null);
    } catch (err) {
      setStatus("offline");
      setError(err instanceof Error ? err.message : "Health check failed");
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  return { status, health, error, refresh };
}
