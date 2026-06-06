"use client";

import { useEffect, useId, useState, type FormEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronDown, Loader2, PhoneCall } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import {
  COUNTRY_CODES,
  DEFAULT_COUNTRY,
  isValidE164,
  requestCallback,
  toE164,
  type CountryCode,
} from "@/lib/call-service";
import { cn } from "@/lib/utils";

interface CallMeModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type Status = "idle" | "submitting" | "success";

/**
 * "Call Me" — request an outbound AI callback. Self-contained: validation +
 * E.164 formatting happen here, the call is placed via `lib/call-service`
 * (provider-agnostic). No provider/telecom logic lives in this component.
 */
export function CallMeModal({ open, onOpenChange }: CallMeModalProps) {
  const nameId = useId();
  const phoneId = useId();
  const errId = useId();

  const [name, setName] = useState("");
  const [country, setCountry] = useState<CountryCode>(DEFAULT_COUNTRY);
  const [phone, setPhone] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  // Reset to a clean slate whenever the modal is closed.
  useEffect(() => {
    if (!open) {
      const t = setTimeout(() => {
        setName("");
        setCountry(DEFAULT_COUNTRY);
        setPhone("");
        setStatus("idle");
        setError(null);
      }, 200); // after the close animation
      return () => clearTimeout(t);
    }
  }, [open]);

  const submitting = status === "submitting";
  const canSubmit = phone.trim().length > 0 && !submitting;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);

    const e164 = toE164(country.dial, phone);
    if (!isValidE164(e164)) {
      setError("Please enter a valid phone number.");
      return;
    }

    setStatus("submitting");
    const result = await requestCallback({ name, phone: e164 });

    if (result.success) {
      setStatus("success");
      toast.success("Call requested", {
        description: "Sarthak's AI representative will call you shortly.",
      });
    } else {
      setStatus("idle");
      setError(result.message ?? "Something went wrong. Please try again.");
      toast.error("Couldn't request the call", {
        description: result.message ?? "Please try again.",
      });
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <div className="mx-auto mb-1 flex h-11 w-11 items-center justify-center rounded-full bg-primary/15 sm:mx-0">
            <PhoneCall className="h-5 w-5 text-primary" />
          </div>
          <DialogTitle>Request a call</DialogTitle>
          <DialogDescription>
            Sarthak&apos;s AI representative will call you to talk through projects,
            experience, skills, or interview availability.
          </DialogDescription>
        </DialogHeader>

        <AnimatePresence mode="wait" initial={false}>
          {status === "success" ? (
            <motion.div
              key="success"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="flex flex-col items-center gap-3 py-4 text-center"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/15">
                <Check className="h-6 w-6 text-emerald-400" />
              </div>
              <div className="space-y-1">
                <p className="text-base font-semibold text-foreground">You&apos;re all set</p>
                <p className="text-sm text-muted-foreground">
                  Great! Sarthak&apos;s AI representative will call you shortly.
                </p>
              </div>
              <Button className="mt-1 w-full" onClick={() => onOpenChange(false)}>
                Done
              </Button>
            </motion.div>
          ) : (
            <motion.form
              key="form"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              onSubmit={handleSubmit}
              className="space-y-4"
              noValidate
            >
              {/* Name (optional) */}
              <div className="space-y-1.5">
                <label htmlFor={nameId} className="text-sm font-medium text-foreground/80">
                  Name <span className="font-normal text-muted-foreground">(optional)</span>
                </label>
                <Input
                  id={nameId}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your name"
                  autoComplete="name"
                  disabled={submitting}
                />
              </div>

              {/* Phone (required) */}
              <div className="space-y-1.5">
                <label htmlFor={phoneId} className="text-sm font-medium text-foreground/80">
                  Phone number
                </label>
                <div className="flex gap-2">
                  {/* Country code selector */}
                  <div className="relative shrink-0">
                    <select
                      aria-label="Country code"
                      value={country.iso}
                      disabled={submitting}
                      onChange={(e) =>
                        setCountry(
                          COUNTRY_CODES.find((c) => c.iso === e.target.value) ?? DEFAULT_COUNTRY,
                        )
                      }
                      className={cn(
                        "h-10 appearance-none rounded-md border border-border bg-background/60 pl-3 pr-8 text-sm text-foreground",
                        "transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                        "disabled:cursor-not-allowed disabled:opacity-50",
                      )}
                    >
                      {COUNTRY_CODES.map((c) => (
                        <option key={c.iso} value={c.iso} className="bg-card text-foreground">
                          {c.flag} {c.dial} · {c.name}
                        </option>
                      ))}
                    </select>
                    <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  </div>

                  <Input
                    id={phoneId}
                    type="tel"
                    inputMode="tel"
                    autoComplete="tel-national"
                    value={phone}
                    onChange={(e) => {
                      setPhone(e.target.value);
                      if (error) setError(null);
                    }}
                    placeholder="98765 43210"
                    aria-invalid={error ? true : undefined}
                    aria-describedby={error ? errId : undefined}
                    disabled={submitting}
                    autoFocus
                    className="flex-1"
                  />
                </div>
                {error && (
                  <p id={errId} role="alert" className="text-xs text-destructive">
                    {error}
                  </p>
                )}
              </div>

              <Button type="submit" className="w-full" disabled={!canSubmit}>
                {submitting ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Requesting…
                  </>
                ) : (
                  <>
                    <PhoneCall className="h-4 w-4" />
                    Request Call
                  </>
                )}
              </Button>

              <p className="text-center text-[11px] text-muted-foreground/70">
                Phone availability may vary by region.
              </p>
            </motion.form>
          )}
        </AnimatePresence>
      </DialogContent>
    </Dialog>
  );
}
