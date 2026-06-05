"use client";

import { Toaster as SonnerToaster } from "sonner";

/** App-wide toast surface. Styled to match the dark, glassy theme. */
export function Toaster() {
  return (
    <SonnerToaster
      theme="dark"
      position="top-center"
      toastOptions={{
        classNames: {
          toast:
            "group border border-border bg-card text-card-foreground shadow-glow rounded-xl",
          description: "text-muted-foreground",
          actionButton: "bg-primary text-primary-foreground",
          cancelButton: "bg-white/5 text-muted-foreground",
          error: "border-destructive/40",
        },
      }}
    />
  );
}
