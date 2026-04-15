"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useSidebar } from "@/components/ui/sidebar";
import { Pulse } from "@phosphor-icons/react";
import { ActivityDropdown } from "./components/ActivityDropdown/ActivityDropdown";
import { formatNotificationCount } from "./helpers";
import { useAgentActivityDropdown } from "./useAgentActivityDropdown";

export function AgentActivityDropdown() {
  const {
    activeExecutions,
    recentCompletions,
    recentFailures,
    isOpen,
    setIsOpen,
  } = useAgentActivityDropdown();
  const { state } = useSidebar();
  const isSidebarCollapsed = state === "collapsed";

  const activeCount = activeExecutions.length;

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <Tooltip>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <button
              className={`group relative flex size-8 items-center justify-center rounded-md transition-colors hover:bg-sidebar-accent ${isOpen ? "bg-sidebar-accent" : ""}`}
              data-testid="agent-activity-button"
              aria-label="View Agent Activity"
            >
              <Pulse className="!size-5" />

              {activeCount > 0 && (
                <div
                  data-testid="agent-activity-badge"
                  className="absolute -right-0.5 -top-0.5 flex size-3.5 items-center justify-center rounded-full bg-purple-600 text-[8px] font-medium text-white"
                >
                  {formatNotificationCount(activeCount)}
                  <div className="absolute -inset-0.5 animate-spin rounded-full border-2 border-transparent border-r-purple-200 border-t-purple-200" />
                </div>
              )}
            </button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side={isSidebarCollapsed ? "right" : "bottom"}>
          {activeCount > 0
            ? `${activeCount} active agent${activeCount > 1 ? "s" : ""}`
            : "Agent Activity"}
        </TooltipContent>
      </Tooltip>

      <PopoverContent
        className="w-80 p-0"
        side={isSidebarCollapsed ? "right" : "bottom"}
        align="start"
        sideOffset={8}
      >
        <ActivityDropdown
          activeExecutions={activeExecutions}
          recentCompletions={recentCompletions}
          recentFailures={recentFailures}
        />
      </PopoverContent>
    </Popover>
  );
}
