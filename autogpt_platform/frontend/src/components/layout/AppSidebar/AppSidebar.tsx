"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  IconAutoGPTLogo,
  IconAutoGPTLogoMinimal,
} from "@/components/__legacy__/ui/icons";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import {
  NotePencil,
  Books,
  ShoppingBag,
  PenNibStraight,
  GearSix,
  CircleNotch,
  ChatCircleDots,
  CaretDown,
  MagnifyingGlass,
} from "@phosphor-icons/react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ReactNode, useEffect, useState } from "react";
import { UsageLimits } from "@/app/(platform)/copilot/components/UsageLimits/UsageLimits";
import { NotificationToggle } from "@/app/(platform)/copilot/components/ChatSidebar/components/NotificationToggle/NotificationToggle";
import { AgentActivityDropdown } from "@/components/layout/Navbar/components/AgentActivityDropdown/AgentActivityDropdown";
import { useTallyPopup } from "@/components/molecules/TallyPoup/useTallyPopup";

interface Props {
  dynamicContent?: ReactNode;
}

export function AppSidebar({ dynamicContent }: Props) {
  const { state } = useSidebar();
  const isCollapsed = state === "collapsed";
  const pathname = usePathname();
  const { state: tallyState } = useTallyPopup();
  const { isLoggedIn } = useSupabase();

  const [loadingHref, setLoadingHref] = useState<string | null>(null);
  const [isTasksOpen, setIsTasksOpen] = useState(true);

  useEffect(() => {
    setLoadingHref(null);
  }, [pathname]);

  const homeHref = "/copilot";

  const navLinks = [
    {
      name: "Search",
      href: "/marketplace/search",
      icon: MagnifyingGlass,
      testId: "sidebar-link-search",
      showWhenCollapsed: true,
    },
    {
      name: "Workflows",
      href: "/library",
      icon: Books,
      testId: "sidebar-link-workflows",
      showWhenCollapsed: true,
    },
    {
      name: "Explore",
      href: "/marketplace",
      icon: ShoppingBag,
      testId: "sidebar-link-marketplace",
      showWhenCollapsed: false,
    },
    {
      name: "Builder",
      href: "/build",
      icon: PenNibStraight,
      testId: "sidebar-link-build",
      showWhenCollapsed: false,
    },
  ];

  function isActive(href: string) {
    if (href === homeHref) {
      return pathname === "/" || pathname.startsWith(homeHref);
    }
    return pathname.startsWith(href);
  }

  if (!isLoggedIn) return null;

  return (
    <Sidebar
      variant="sidebar"
      collapsible="icon"
      className="border-r border-zinc-200 !bg-zinc-100"
    >
      {/* Header */}
      <SidebarHeader
        className={cn(
          "!flex-row px-3",
          isCollapsed
            ? "items-center justify-center py-3"
            : "items-center py-3",
        )}
      >
        {!isCollapsed && (
          <motion.div
            className="flex w-full items-center justify-between"
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, ease: [0.25, 0.46, 0.45, 0.94] }}
          >
            <Link href={homeHref}>
              <IconAutoGPTLogo className="h-7 w-auto" />
            </Link>
            <div className="flex items-center">
              <AgentActivityDropdown />
              <Tooltip>
                <TooltipTrigger asChild>
                  <SidebarTrigger className="size-10 p-2 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground [&>svg]:!size-5" />
                </TooltipTrigger>
                <TooltipContent side={isCollapsed ? "right" : "bottom"}>
                  Close sidebar
                </TooltipContent>
              </Tooltip>
            </div>
          </motion.div>
        )}
        {isCollapsed && (
          <div className="relative flex flex-col items-center">
            <Link
              href={homeHref}
              className="transition-opacity group-hover:opacity-0"
            >
              <IconAutoGPTLogoMinimal className="h-6 w-6" />
            </Link>
            <SidebarTrigger className="absolute inset-0 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100 hover:bg-sidebar-accent [&>svg]:!size-5" />
          </div>
        )}
      </SidebarHeader>

      {/* Navigation links */}
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <div>
              <SidebarMenu className={cn(isCollapsed && "gap-3")}>
                <SidebarMenuItem>
                  <SidebarMenuButton
                    asChild
                    isActive={isActive(homeHref)}
                    tooltip="New Task"
                    className={cn(
                      "!rounded-xl py-5 data-[active=true]:!bg-zinc-200 data-[active=true]:!font-normal",
                      !isCollapsed && "gap-3",
                    )}
                  >
                    <Link
                      href="/copilot"
                      data-testid="sidebar-link-new-task"
                      onClick={() =>
                        !isActive(homeHref) && setLoadingHref(homeHref)
                      }
                    >
                      {loadingHref === homeHref && isCollapsed ? (
                        <CircleNotch className="!size-5 animate-spin text-zinc-600" />
                      ) : (
                        <NotePencil className="!size-5" weight="regular" />
                      )}
                      {!isCollapsed && <span className="flex-1">New Task</span>}
                      {loadingHref === homeHref && !isCollapsed && (
                        <CircleNotch className="!size-4 animate-spin text-zinc-600" />
                      )}
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
                {navLinks
                  .filter((link) => !isCollapsed || link.showWhenCollapsed)
                  .map((link) => (
                    <SidebarMenuItem key={link.name}>
                      <SidebarMenuButton
                        asChild
                        isActive={isActive(link.href)}
                        tooltip={link.name}
                        className={cn(
                          "!rounded-xl py-5 data-[active=true]:!bg-zinc-200 data-[active=true]:!font-normal",
                          !isCollapsed && "gap-3",
                        )}
                      >
                        <Link
                          href={link.href}
                          data-testid={link.testId}
                          onClick={() =>
                            !isActive(link.href) && setLoadingHref(link.href)
                          }
                        >
                          {loadingHref === link.href && isCollapsed ? (
                            <CircleNotch className="!size-5 animate-spin text-zinc-600" />
                          ) : (
                            <link.icon className="!size-5" weight="regular" />
                          )}
                          {!isCollapsed && (
                            <span className="flex-1">{link.name}</span>
                          )}
                          {loadingHref === link.href && !isCollapsed && (
                            <CircleNotch className="!size-4 animate-spin text-zinc-600" />
                          )}
                        </Link>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
              </SidebarMenu>
            </div>
          </SidebarGroupContent>
        </SidebarGroup>

        {dynamicContent && (
          <SidebarGroup className="flex-1 overflow-hidden">
            {!isCollapsed && (
              <>
                <button
                  onClick={() => setIsTasksOpen((prev) => !prev)}
                  className="group/tasks flex w-full items-center justify-between px-3 pb-4"
                >
                  <span className="flex items-center gap-1 text-sm font-medium text-zinc-600">
                    All tasks
                    <CaretDown
                      className={cn(
                        "!size-3.5 text-zinc-400 transition-transform",
                        !isTasksOpen && "-rotate-90",
                      )}
                      weight="bold"
                    />
                  </span>
                </button>
                <AnimatePresence initial={false}>
                  {isTasksOpen && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{
                        opacity: {
                          duration: 0.15,
                          ease: [0.25, 0.46, 0.45, 0.94],
                        },
                        height: {
                          duration: 0.2,
                          ease: [0.25, 0.46, 0.45, 0.94],
                        },
                      }}
                      className="flex-1 overflow-hidden"
                    >
                      <SidebarGroupContent className="h-full overflow-y-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
                        {dynamicContent}
                      </SidebarGroupContent>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}
          </SidebarGroup>
        )}
      </SidebarContent>

      {/* Footer */}
      <SidebarFooter className="border-t border-zinc-200 p-2">
        <div
          className={cn(
            "flex",
            isCollapsed ? "flex-col items-center gap-3" : "items-center gap-1",
          )}
        >
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="[&_button]:!flex [&_button]:!size-8 [&_button]:items-center [&_button]:justify-center [&_button]:!rounded-xl [&_button]:!p-0 [&_button]:transition-colors [&_button]:hover:bg-sidebar-accent [&_button_svg]:!size-5">
                <UsageLimits />
              </div>
            </TooltipTrigger>
            <TooltipContent side={isCollapsed ? "right" : "top"}>
              Usage
            </TooltipContent>
          </Tooltip>
          {!isCollapsed && (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="[&_button]:!flex [&_button]:!size-8 [&_button]:items-center [&_button]:justify-center [&_button]:!rounded-xl [&_button]:!p-0 [&_button]:transition-colors [&_button]:hover:bg-sidebar-accent [&_button_svg]:!size-5">
                  <NotificationToggle />
                </div>
              </TooltipTrigger>
              <TooltipContent side="top">Notifications</TooltipContent>
            </Tooltip>
          )}
          {!isCollapsed && !tallyState.isFormVisible && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="flex !size-8 items-center justify-center !rounded-xl transition-colors hover:bg-sidebar-accent"
                  data-tally-open="3yx2L0"
                  data-tally-emoji-text="👋"
                  data-tally-emoji-animation="wave"
                  data-sentry-replay-id={
                    tallyState.sentryReplayId || "not-initialized"
                  }
                  data-sentry-replay-url={
                    tallyState.replayUrl || "not-initialized"
                  }
                  data-page-url={
                    tallyState.pageUrl
                      ? tallyState.pageUrl.split("?")[0]
                      : "not-initialized"
                  }
                  data-is-authenticated={
                    tallyState.isAuthenticated === null
                      ? "unknown"
                      : String(tallyState.isAuthenticated)
                  }
                  aria-label="Give Feedback"
                >
                  <ChatCircleDots className="!size-5" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="top">Feedback</TooltipContent>
            </Tooltip>
          )}
          {!isCollapsed && <div className="flex-1" />}
          <Tooltip>
            <TooltipTrigger asChild>
              <Link
                href="/profile/settings"
                className="flex !size-8 items-center justify-center !rounded-xl transition-colors hover:bg-sidebar-accent"
                data-testid="sidebar-settings-button"
              >
                <GearSix className="!size-5" />
              </Link>
            </TooltipTrigger>
            <TooltipContent side={isCollapsed ? "right" : "top"}>
              Settings
            </TooltipContent>
          </Tooltip>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
