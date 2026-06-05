"use client";

import { motion } from "framer-motion";
import { Clock, Globe } from "lucide-react";
import { cn, formatSlot, groupSlotsByDay } from "@/lib/utils";
import type { SlotView } from "@/types";

interface SlotPickerProps {
  slots: SlotView[];
  timezone: string;
  selected: SlotView | null;
  onSelect: (slot: SlotView) => void;
}

/** Calendar-style grid of bookable slots, grouped by day. */
export function SlotPicker({ slots, timezone, selected, onSelect }: SlotPickerProps) {
  const groups = groupSlotsByDay(slots, timezone || undefined);

  return (
    <div className="space-y-4">
      {timezone && (
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Globe className="h-3.5 w-3.5" /> Times shown in {timezone}
        </div>
      )}

      {groups.map(({ day, slots: daySlots }) => (
        <div key={day}>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {day}
          </p>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {daySlots.map((slot) => {
              const { time } = formatSlot(slot.start, timezone || undefined);
              const isSelected = selected?.start === slot.start;
              return (
                <motion.button
                  key={slot.start}
                  whileTap={{ scale: 0.96 }}
                  onClick={() => onSelect(slot)}
                  className={cn(
                    "flex items-center justify-center gap-1.5 rounded-lg border px-3 py-2.5 text-sm font-medium transition-all",
                    isSelected
                      ? "border-primary bg-primary/15 text-foreground shadow-glow"
                      : "border-border bg-card/60 text-foreground/80 hover:border-white/20 hover:bg-white/[0.04]",
                  )}
                >
                  <Clock className="h-3.5 w-3.5 opacity-70" />
                  {time}
                </motion.button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
