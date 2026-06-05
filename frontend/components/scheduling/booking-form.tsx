"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { formatSlot } from "@/lib/utils";
import type { SlotView } from "@/types";

interface BookingFormProps {
  slot: SlotView;
  timezone: string;
  submitting: boolean;
  onSubmit: (data: { name: string; email: string; topic?: string }) => void;
  onBack: () => void;
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/** Collects attendee details for the selected slot. */
export function BookingForm({ slot, timezone, submitting, onSubmit, onBack }: BookingFormProps) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [topic, setTopic] = useState("");
  const [errors, setErrors] = useState<{ name?: string; email?: string }>({});

  const { date, time } = formatSlot(slot.start, timezone || undefined);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const next: typeof errors = {};
    if (!name.trim()) next.name = "Please enter your name.";
    if (!EMAIL_RE.test(email)) next.email = "Please enter a valid email.";
    setErrors(next);
    if (Object.keys(next).length > 0) return;
    onSubmit({ name: name.trim(), email: email.trim(), topic: topic.trim() || undefined });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div className="rounded-lg border border-primary/30 bg-primary/[0.07] px-3 py-2 text-sm">
        <span className="text-muted-foreground">Booking</span>{" "}
        <span className="font-medium text-foreground">
          {date} · {time}
        </span>
      </div>

      <div className="space-y-1.5">
        <label htmlFor="name" className="text-xs font-medium text-muted-foreground">
          Your name
        </label>
        <Input
          id="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Ada Lovelace"
          autoComplete="name"
        />
        {errors.name && <p className="text-xs text-destructive">{errors.name}</p>}
      </div>

      <div className="space-y-1.5">
        <label htmlFor="email" className="text-xs font-medium text-muted-foreground">
          Email
        </label>
        <Input
          id="email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="ada@example.com"
          autoComplete="email"
        />
        {errors.email && <p className="text-xs text-destructive">{errors.email}</p>}
      </div>

      <div className="space-y-1.5">
        <label htmlFor="topic" className="text-xs font-medium text-muted-foreground">
          Topic <span className="opacity-60">(optional)</span>
        </label>
        <Input
          id="topic"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="What would you like to discuss?"
        />
      </div>

      <div className="flex gap-2 pt-1">
        <Button type="button" variant="ghost" onClick={onBack} disabled={submitting}>
          Back
        </Button>
        <Button type="submit" className="flex-1" disabled={submitting}>
          {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : "Book meeting"}
        </Button>
      </div>
    </form>
  );
}
