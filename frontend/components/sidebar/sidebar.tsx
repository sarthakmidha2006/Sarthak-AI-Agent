"use client";

import { motion } from "framer-motion";
import { MessageSquarePlus, MessageSquare, Settings, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ProfileCard } from "@/components/profile/profile-card";
import { cn } from "@/lib/utils";
import type { BackendStatus, Conversation } from "@/types";

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  status: BackendStatus;
  onNewChat: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onOpenSettings: () => void;
}

const DOT: Record<BackendStatus, string> = {
  online: "bg-emerald-400",
  offline: "bg-destructive",
  checking: "bg-amber-400",
};

export function Sidebar({
  conversations,
  activeId,
  status,
  onNewChat,
  onSelect,
  onDelete,
  onOpenSettings,
}: SidebarProps) {
  return (
    <div className="flex h-full flex-col bg-card/40">
      <div className="space-y-4 p-4">
        <ProfileCard />
        <Button onClick={onNewChat} className="w-full">
          <MessageSquarePlus className="h-4 w-4" /> New chat
        </Button>
      </div>

      <div className="px-4 pb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        History
      </div>
      <div className="scrollbar-thin flex-1 overflow-y-auto px-2">
        {conversations.length === 0 ? (
          <p className="px-2 py-4 text-sm text-muted-foreground">No conversations yet.</p>
        ) : (
          <ul className="space-y-1">
            {conversations.map((c) => {
              const isActive = c.id === activeId;
              return (
                <li key={c.id}>
                  <motion.div
                    layout
                    className={cn(
                      "group flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm transition-colors",
                      isActive
                        ? "bg-primary/12 text-foreground"
                        : "text-foreground/70 hover:bg-white/[0.04] hover:text-foreground",
                    )}
                  >
                    <button
                      onClick={() => onSelect(c.id)}
                      className="flex min-w-0 flex-1 items-center gap-2 text-left"
                    >
                      <MessageSquare
                        className={cn("h-4 w-4 shrink-0", isActive ? "text-primary" : "text-muted-foreground")}
                      />
                      <span className="truncate">{c.title}</span>
                    </button>
                    <button
                      onClick={() => onDelete(c.id)}
                      aria-label="Delete conversation"
                      className="rounded-md p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-white/5 hover:text-destructive group-hover:opacity-100"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </motion.div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className="border-t border-border p-3">
        <button
          onClick={onOpenSettings}
          className="flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-sm text-foreground/70 transition-colors hover:bg-white/[0.04] hover:text-foreground"
        >
          <span className="flex items-center gap-2">
            <Settings className="h-4 w-4" /> Settings
          </span>
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className={cn("h-2 w-2 rounded-full", DOT[status])} />
            {status === "checking" ? "…" : status}
          </span>
        </button>
      </div>
    </div>
  );
}
