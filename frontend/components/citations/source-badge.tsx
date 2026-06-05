import { Badge } from "@/components/ui/badge";
import { SOURCE_BADGES } from "@/lib/constants";
import { cn, sourceKind } from "@/lib/utils";

interface SourceBadgeProps {
  sourceType: string;
  className?: string;
  showIcon?: boolean;
}

/** Colored pill identifying a citation's source kind (Resume/GitHub/etc.). */
export function SourceBadge({ sourceType, className, showIcon = true }: SourceBadgeProps) {
  const kind = sourceKind(sourceType);
  const cfg = SOURCE_BADGES[kind];
  const Icon = cfg.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium",
        cfg.className,
        className,
      )}
    >
      {showIcon && <Icon className="h-3 w-3" />}
      {cfg.label}
    </span>
  );
}

export { Badge };
