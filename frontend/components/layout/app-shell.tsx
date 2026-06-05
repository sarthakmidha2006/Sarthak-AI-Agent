"use client";

import { Sheet, SheetContent } from "@/components/ui/sheet";

interface AppShellProps {
  sidebar: React.ReactNode;
  children: React.ReactNode;
  mobileSidebarOpen: boolean;
  onMobileSidebarOpenChange: (open: boolean) => void;
}

/**
 * Responsive two-column layout:
 *  - Desktop (md+): persistent left sidebar + main column.
 *  - Mobile/tablet: sidebar collapses into a slide-over Sheet.
 */
export function AppShell({
  sidebar,
  children,
  mobileSidebarOpen,
  onMobileSidebarOpenChange,
}: AppShellProps) {
  return (
    <div className="flex h-[100dvh] w-full overflow-hidden">
      {/* Desktop sidebar */}
      <aside className="hidden w-80 shrink-0 border-r border-border md:block">{sidebar}</aside>

      {/* Mobile sidebar */}
      <Sheet open={mobileSidebarOpen} onOpenChange={onMobileSidebarOpenChange}>
        <SheetContent side="left" className="w-80 p-0">
          {sidebar}
        </SheetContent>
      </Sheet>

      {/* Main column */}
      <main className="flex min-w-0 flex-1 flex-col">{children}</main>
    </div>
  );
}
