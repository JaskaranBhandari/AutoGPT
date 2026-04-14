"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { IconType } from "@/components/__legacy__/ui/icons";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { PreviewBanner } from "@/components/layout/Navbar/components/PreviewBanner/PreviewBanner";
import { useSidebar } from "@/components/ui/sidebar";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { environment } from "@/services/environment";
import { AccountMenu } from "./components/AccountMenu/AccountMenu";
import { FeedbackButton } from "./components/FeedbackButton";
import { AgentActivityDropdown } from "./components/AgentActivityDropdown/AgentActivityDropdown";
import { LoginButton } from "./components/LoginButton";
import { MobileNavBar } from "./components/MobileNavbar/MobileNavBar";
import { NavbarLoading } from "./components/NavbarLoading";
import { Wallet } from "./components/Wallet/Wallet";
import { getAccountMenuItems, loggedInLinks } from "./helpers";
import { useLayout } from "@/components/layout/LayoutContext";

export function Navbar() {
  const { layout } = useLayout();
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const breakpoint = useBreakpoint();
  const isSmallScreen = breakpoint === "sm" || breakpoint === "base";
  const dynamicMenuItems = getAccountMenuItems(user?.role);
  const { state: sidebarState } = useSidebar();
  const isSidebarCollapsed = sidebarState === "collapsed";
  const previewBranchName = environment.getPreviewStealingDev();
  const logoutInProgress = isLogoutInProgress();

  const { data: profile, isLoading: isProfileLoading } =
    useGetV2GetUserProfile({
      query: {
        select: okData,
        enabled: isLoggedIn && !!user && !logoutInProgress,
        queryKey: ["/api/store/profile", user?.id],
      },
    });

  const isLoadingProfile = isProfileLoading || isUserLoading;
  const isClassic = layout === "classic";

  const shouldShowPreviewBanner = Boolean(isLoggedIn && previewBranchName);

  const actualLoggedInLinks = [
    { name: "Home", href: "/copilot" },
    { name: "Agents", href: "/library" },
    ...loggedInLinks,
  ];

  if (isUserLoading) {
    return isClassic ? <NavbarLoading /> : null;
  }

  return (
    <>
      {/* Classic layout: full top navbar */}
      {isClassic && (
        <div className="sticky top-0 z-40 w-full">
          {shouldShowPreviewBanner && previewBranchName ? (
            <PreviewBanner branchName={previewBranchName} />
          ) : null}
          <nav
            className="relative inline-flex w-full items-center justify-end border-b border-zinc-100 bg-[#FAFAFA]/80 p-3 backdrop-blur-xl"
            style={{ height: NAVBAR_HEIGHT_PX }}
          >
            {isSidebarCollapsed && (
              <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
                <AutoGPTLogo className="h-auto w-[4.5rem] md:w-[5.5rem]" />
              </div>
            )}
            {isLoggedIn && !isSmallScreen ? (
              <div className="flex items-center gap-4">
                <FeedbackButton />
                <AgentActivityDropdown />
                {profile && <Wallet key={profile.username} />}
                <AccountMenu
                  userName={profile?.username}
                  userEmail={profile?.name}
                  avatarSrc={profile?.avatar_url ?? ""}
                  menuItemGroups={dynamicMenuItems}
                  isLoading={isLoadingProfile}
                />
              </div>
            ) : !isLoggedIn ? (
              <LoginButton />
            ) : null}
          </nav>
        </div>
      )}

      {/* Modern layout: only preview banner + login for unauthenticated */}
      {!isClassic && (
        <>
          {shouldShowPreviewBanner && previewBranchName ? (
            <div className="sticky top-0 z-40 w-full">
              <PreviewBanner branchName={previewBranchName} />
            </div>
          ) : null}
          {!isLoggedIn ? (
            <div className="flex w-full justify-end p-3">
              <LoginButton />
            </div>
          ) : null}
        </>
      )}

      {/* Mobile Navbar (both layouts) */}
      {isLoggedIn && isSmallScreen ? (
        <div className="fixed right-0 top-2 z-50 flex items-center gap-0">
          <Wallet />
          <MobileNavBar
            userName={profile?.username}
            menuItemGroups={[
              {
                groupName: "Navigation",
                items: actualLoggedInLinks
                  .map((link) => {
                    return {
                      icon:
                        link.href === "/marketplace"
                          ? IconType.Marketplace
                          : link.href === "/build"
                            ? IconType.Builder
                            : link.href === "/copilot"
                              ? IconType.Chat
                              : link.href === "/library"
                                ? IconType.Library
                                : link.href === "/monitor"
                                  ? IconType.Library
                                  : IconType.LayoutDashboard,
                      text: link.name,
                      href: link.href,
                    };
                  })
                  .filter((item) => item !== null) as Array<{
                  icon: IconType;
                  text: string;
                  href: string;
                }>,
              },
              ...dynamicMenuItems,
            ]}
            userEmail={profile?.name}
            avatarSrc={profile?.avatar_url ?? ""}
          />
        </div>
      ) : null}
    </>
  );
}
