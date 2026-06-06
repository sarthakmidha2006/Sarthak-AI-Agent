"use client";

import { Menu, PhoneCall, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { PERSONA } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { BackendStatus } from "@/types";

interface MobileHeaderProps {
  status: BackendStatus;
  onOpenSidebar: () => void;
  onOpenSettings: () => void;
  onOpenCallMe: () => void;
}

const DOT: Record<BackendStatus, string> = {
  online: "bg-emerald-400",
  offline: "bg-destructive",
  checking: "bg-amber-400",
};

/** Top app bar for mobile/tablet — sidebar toggle + identity + settings. */
export function MobileHeader({
  status,
  onOpenSidebar,
  onOpenSettings,
  onOpenCallMe,
}: MobileHeaderProps) {
  return (
    <header className="flex items-center justify-between border-b border-border bg-background/80 px-3 py-2.5 backdrop-blur-xl md:hidden">
      <Button variant="ghost" size="icon" onClick={onOpenSidebar} aria-label="Open menu">
        <Menu className="h-5 w-5" />
      </Button>

      <div className="flex items-center gap-2">
        <Avatar className="h-7 w-7">
          <AvatarFallback className="text-[11px]">{PERSONA.initials}</AvatarFallback>
        </Avatar>
        <span className="text-sm font-semibold">{PERSONA.name}</span>
        <span className={cn("h-2 w-2 rounded-full", DOT[status])} />
      </div>

      <div className="flex items-center">
        <Button variant="ghost" size="icon" onClick={onOpenCallMe} aria-label="Request a call">
          <PhoneCall className="h-5 w-5" />
        </Button>
        <Button variant="ghost" size="icon" onClick={onOpenSettings} aria-label="Settings">
          <Settings className="h-5 w-5" />
        </Button>
      </div>
    </header>
  );
}
