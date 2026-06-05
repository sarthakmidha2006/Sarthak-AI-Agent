"use client";

import { motion } from "framer-motion";
import { CalendarCheck, Check, Clock, Globe } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { formatSlot } from "@/lib/utils";
import type { BookResponse } from "@/types";

interface BookingSuccessModalProps {
  booking: BookResponse | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** Celebratory confirmation shown after a successful booking. */
export function BookingSuccessModal({ booking, open, onOpenChange }: BookingSuccessModalProps) {
  const slot =
    booking?.start_time != null
      ? formatSlot(booking.start_time, booking.timezone ?? undefined)
      : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-sm text-center">
        <DialogHeader className="items-center">
          <motion.div
            initial={{ scale: 0, rotate: -20 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: "spring", stiffness: 260, damping: 16 }}
            className="mb-2 flex h-14 w-14 items-center justify-center rounded-full bg-gradient-to-br from-primary to-secondary shadow-glow"
          >
            <Check className="h-7 w-7 text-white" strokeWidth={3} />
          </motion.div>
          <DialogTitle className="text-center">Meeting confirmed</DialogTitle>
        </DialogHeader>

        <p className="text-sm text-muted-foreground">
          {booking?.message ?? "Your meeting has been booked."}
        </p>

        {slot && (
          <div className="mt-2 space-y-2 rounded-xl border border-border bg-background/50 p-4 text-left text-sm">
            <div className="flex items-center gap-2">
              <CalendarCheck className="h-4 w-4 text-primary" />
              <span>{slot.date}</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-primary" />
              <span>{slot.time}</span>
            </div>
            {booking?.timezone && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Globe className="h-4 w-4" />
                <span>{booking.timezone}</span>
              </div>
            )}
          </div>
        )}

        {booking?.booking_id && (
          <p className="mt-2 text-[11px] text-muted-foreground">
            Confirmation ID: <span className="font-mono">{booking.booking_id}</span>
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}
