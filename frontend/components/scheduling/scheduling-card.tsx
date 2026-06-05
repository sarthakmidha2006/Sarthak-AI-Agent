"use client";

import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CalendarDays, Loader2, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorState } from "@/components/common/error-state";
import { SlotPicker } from "@/components/scheduling/slot-picker";
import { BookingForm } from "@/components/scheduling/booking-form";
import { BookingSuccessModal } from "@/components/scheduling/booking-success-modal";
import { api } from "@/lib/api";
import { ApiError, type AvailabilityResponse, type BookResponse, type SlotView } from "@/types";

interface SchedulingCardProps {
  /** Pre-parsed availability from tool calls, if any (else fetched live). */
  initial?: AvailabilityResponse | null;
}

type Phase = "loading" | "ready" | "form" | "booking" | "error";

/** End-to-end scheduling widget embedded in an assistant message. */
export function SchedulingCard({ initial }: SchedulingCardProps) {
  const [phase, setPhase] = useState<Phase>(initial?.slots.length ? "ready" : "loading");
  const [data, setData] = useState<AvailabilityResponse | null>(initial ?? null);
  const [selected, setSelected] = useState<SlotView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [booking, setBooking] = useState<BookResponse | null>(null);
  const [successOpen, setSuccessOpen] = useState(false);

  const loadAvailability = useCallback(async () => {
    setPhase("loading");
    setError(null);
    try {
      const res = await api.availability();
      setData(res);
      setPhase("ready");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't load availability.");
      setPhase("error");
    }
  }, []);

  useEffect(() => {
    if (!initial?.slots.length) void loadAvailability();
  }, [initial, loadAvailability]);

  const handleBook = async (form: { name: string; email: string; topic?: string }) => {
    if (!selected) return;
    setPhase("booking");
    try {
      const res = await api.book({
        name: form.name,
        email: form.email,
        start_time: selected.start,
        duration_minutes: data?.duration_minutes,
        topic: form.topic,
      });
      setBooking(res);
      if (res.status === "confirmed") {
        setSuccessOpen(true);
        setPhase("ready");
      } else {
        // Slot taken → surface alternatives the backend returned.
        toast.warning("Slot unavailable", { description: res.message });
        if (res.alternatives?.length && data) {
          setData({ ...data, slots: res.alternatives, count: res.alternatives.length });
        }
        setSelected(null);
        setPhase("ready");
      }
    } catch (err) {
      toast.error("Booking failed", {
        description: err instanceof ApiError ? err.message : "Please try again.",
      });
      setPhase("form");
    }
  };

  return (
    <Card className="mt-3 overflow-hidden border-primary/20 bg-card/80">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/15 text-primary">
            <CalendarDays className="h-4 w-4" />
          </span>
          Schedule a meeting
        </div>
        {phase !== "loading" && phase !== "booking" && (
          <Button variant="ghost" size="icon-sm" onClick={loadAvailability} aria-label="Refresh slots">
            <RefreshCw className="h-4 w-4" />
          </Button>
        )}
      </div>

      <div className="p-4">
        <AnimatePresence mode="wait">
          {phase === "loading" && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="grid grid-cols-2 gap-2 sm:grid-cols-3"
            >
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </motion.div>
          )}

          {phase === "error" && (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <ErrorState
                title="Couldn't load availability"
                message={error ?? "Please try again."}
                onRetry={loadAvailability}
                compact
              />
            </motion.div>
          )}

          {(phase === "ready" || phase === "form" || phase === "booking") && data && (
            <motion.div key="content" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {data.slots.length === 0 ? (
                <p className="py-4 text-center text-sm text-muted-foreground">
                  No open slots in the current window.
                </p>
              ) : phase === "ready" ? (
                <>
                  <SlotPicker
                    slots={data.slots}
                    timezone={data.timezone}
                    selected={selected}
                    onSelect={setSelected}
                  />
                  <Button
                    className="mt-4 w-full"
                    disabled={!selected}
                    onClick={() => setPhase("form")}
                  >
                    {selected ? "Continue" : "Select a time"}
                  </Button>
                </>
              ) : (
                selected && (
                  <BookingForm
                    slot={selected}
                    timezone={data.timezone}
                    submitting={phase === "booking"}
                    onSubmit={handleBook}
                    onBack={() => setPhase("ready")}
                  />
                )
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <BookingSuccessModal booking={booking} open={successOpen} onOpenChange={setSuccessOpen} />
    </Card>
  );
}
