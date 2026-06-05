"use client";

import { motion } from "framer-motion";
import { Mic } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface VoiceButtonProps {
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
}

/** Mic button in the composer that opens the voice modal. */
export function VoiceButton({ onClick, active, disabled }: VoiceButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          type="button"
          size="icon"
          variant="ghost"
          onClick={onClick}
          disabled={disabled}
          aria-label="Voice mode"
          className={cn("relative shrink-0 rounded-xl", active && "text-secondary")}
        >
          {active && (
            <motion.span
              className="absolute inset-0 rounded-xl bg-secondary/20"
              animate={{ scale: [1, 1.25, 1], opacity: [0.6, 0, 0.6] }}
              transition={{ duration: 1.4, repeat: Infinity }}
            />
          )}
          <Mic className="h-4 w-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>Talk to the persona</TooltipContent>
    </Tooltip>
  );
}
