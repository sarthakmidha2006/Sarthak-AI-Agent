"use client";

import { motion } from "framer-motion";
import { Brain, GraduationCap, Sparkles } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { PERSONA } from "@/lib/constants";

/** AI-persona profile card: identity, skills, interests, quick stats. */
export function ProfileCard() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="glass rounded-2xl p-5 shadow-soft"
    >
      <div className="flex items-center gap-3">
        <Avatar className="h-14 w-14 shadow-glow">
          <AvatarFallback className="text-base">{PERSONA.initials}</AvatarFallback>
        </Avatar>
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold tracking-tight">{PERSONA.name}</h2>
          <div className="mt-0.5 flex flex-wrap gap-1.5">
            {PERSONA.roles.map((r) => (
              <Badge key={r} variant="secondary" className="text-[11px]">
                {r}
              </Badge>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-3 flex items-center gap-1.5 text-xs text-muted-foreground">
        <GraduationCap className="h-3.5 w-3.5" />
        {PERSONA.org}
      </div>

      <Separator className="my-4" />

      {/* Quick stats */}
      <div className="grid grid-cols-3 gap-2">
        {PERSONA.stats.map((s) => (
          <div
            key={s.label}
            className="rounded-lg border border-border bg-card/50 p-2.5 text-center"
          >
            <p className="text-sm font-semibold text-gradient">{s.value}</p>
            <p className="mt-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
              {s.label}
            </p>
          </div>
        ))}
      </div>

      <Separator className="my-4" />

      {/* Skills */}
      <div>
        <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          <Sparkles className="h-3.5 w-3.5" /> Skills
        </p>
        <div className="flex flex-wrap gap-1.5">
          {PERSONA.skills.map((s) => (
            <Badge key={s} variant="muted" className="text-[11px]">
              {s}
            </Badge>
          ))}
        </div>
      </div>

      <Separator className="my-4" />

      {/* AI interests */}
      <div>
        <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          <Brain className="h-3.5 w-3.5" /> AI Interests
        </p>
        <ul className="space-y-1.5">
          {PERSONA.interests.map((it) => (
            <li key={it} className="flex items-start gap-2 text-sm text-foreground/80">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-secondary" />
              {it}
            </li>
          ))}
        </ul>
      </div>
    </motion.div>
  );
}
